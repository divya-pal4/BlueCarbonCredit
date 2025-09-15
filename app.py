from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import os
import shutil
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from torchvision.models import ResNet50_Weights
import uvicorn

# ---------------- CONFIG ----------------
UPLOAD_DIR = "uploads"
DB_FILE = "embeddings_db.npy"
IMG_DB_FILE = "image_paths_db.npy"
EMBEDDING_THRESHOLD = 0.85  
ORB_MATCH_THRESHOLD = 50   

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- CLASSIFICATION MODEL ----------------
# Load pretrained MobileNetV2 or your mangrove_mobilenetv2.pth
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_model = models.mobilenet_v2(weights=None)
class_model.classifier[1] = nn.Linear(class_model.last_channel, 2)  # 2 classes: mangrove/non-mangrove
class_model.load_state_dict(torch.load("mangrove_mobilenetv2.pth", map_location=device))
class_model.eval().to(device)

class_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def classify_image(img_path):
    img = Image.open(img_path).convert("RGB")
    x = class_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = class_model(x)
        _, pred_class = torch.max(preds, 1)
    return "Mangrove" if pred_class.item() == 1 else "Non-Mangrove"

# ---------------- ANOMALY DETECTION ----------------
resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet.fc = nn.Identity()  
resnet.eval()

def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    x = class_transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = resnet(x).squeeze().numpy()
    return emb / np.linalg.norm(emb)

def load_db():
    if os.path.exists(DB_FILE) and os.path.exists(IMG_DB_FILE):
        embeddings = np.load(DB_FILE, allow_pickle=True)
        paths = np.load(IMG_DB_FILE, allow_pickle=True)
        return embeddings.tolist(), paths.tolist()
    return [], []

def save_db(embeddings, paths):
    np.save(DB_FILE, np.array(embeddings, dtype=object))
    np.save(IMG_DB_FILE, np.array(paths, dtype=object))

def orb_similarity(img1_path, img2_path):
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)
    
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good_matches = [m for m in matches if m.distance < 60]
    return len(good_matches)

def check_duplicate(new_img_path):
    embeddings, paths = load_db()
    new_emb = get_embedding(new_img_path)

    if not embeddings:  
        embeddings.append(new_emb)
        paths.append(new_img_path)
        save_db(embeddings, paths)
        return {"status": "stored", "message": "First image stored"}

    sims = cosine_similarity([new_emb], embeddings)[0]
    max_sim = np.max(sims)
    best_idx = np.argmax(sims)

    if max_sim >= EMBEDDING_THRESHOLD:
        return {
            "status": "duplicate",
            "uploaded_image": new_img_path,
            "matched_with": paths[best_idx],
            "similarity": float(max_sim)
        }
    
    for i, path in enumerate(paths):
        good_matches = orb_similarity(new_img_path, path)
        if good_matches >= ORB_MATCH_THRESHOLD:
            return {
                "status": "duplicate_cropped",
                "uploaded_image": new_img_path,
                "matched_with": path,
                "orb_matches": good_matches
            }

    embeddings.append(new_emb)
    paths.append(new_img_path)
    save_db(embeddings, paths)
    return {
        "status": "stored",
        "message": "New unique image stored",
        "uploaded_image": new_img_path
    }

# ---------------- DENSITY CALCULATION ----------------
def calculate_density(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    density = (green_pixels / total_pixels) * 100
    return round(density, 2)

# ---------------- HEIGHT ESTIMATION ----------------
def get_height():
    return "Approx 5–7 ft (above water)"

# ---------------- FASTAPI APP ----------------
app = FastAPI()

@app.post("/analyze")
async def analyze(horizontal: UploadFile = File(...), vertical: UploadFile = File(...)):
    # Save files
    horiz_path = os.path.join(UPLOAD_DIR, horizontal.filename)
    vert_path = os.path.join(UPLOAD_DIR, vertical.filename)
    with open(horiz_path, "wb") as buffer:
        shutil.copyfileobj(horizontal.file, buffer)
    with open(vert_path, "wb") as buffer:
        shutil.copyfileobj(vertical.file, buffer)

    # Run tasks
    classification = classify_image(horiz_path)
    anomaly = check_duplicate(horiz_path)
    density = calculate_density(horiz_path)
    height = get_height()

    result = {
        "classification": classification,
        "anomaly": anomaly,
        "density_percent": density,
        "height_estimate": height
    }

    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
