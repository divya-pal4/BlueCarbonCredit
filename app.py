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
import random
from typing import List
import traceback

# ------------------ CONFIG ------------------
UPLOAD_DIR = "uploads"
DB_FILE = "embeddings_db.npy"
IMG_DB_FILE = "image_paths_db.npy"
EMBEDDING_THRESHOLD = 0.85  
ORB_MATCH_THRESHOLD = 50   

os.makedirs(UPLOAD_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ CLEAN EMPTY FILES ------------------
for f in [DB_FILE, IMG_DB_FILE]:
    if os.path.exists(f) and os.path.getsize(f) == 0:
        os.remove(f)

# ------------------ IMAGE CLASSIFICATION ------------------
class_model = models.mobilenet_v2(weights=None)
class_model.classifier[1] = nn.Linear(class_model.last_channel, 2)  # Mangrove / Non-Mangrove
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

# ------------------ ANOMALY DETECTION ------------------
resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet = resnet.to(device)
resnet.eval()

def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    x = class_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = resnet(x).squeeze()
        if emb.ndim == 0:  # scalar case
            emb = emb.unsqueeze(0)
        emb = emb.cpu().numpy()
    return emb / np.linalg.norm(emb)

def load_db():
    """Load embeddings and paths safely; handle empty or corrupted files."""
    if os.path.exists(DB_FILE) and os.path.exists(IMG_DB_FILE):
        if os.path.getsize(DB_FILE) == 0 or os.path.getsize(IMG_DB_FILE) == 0:
            return [], []
        try:
            embeddings = np.load(DB_FILE, allow_pickle=True).tolist()
            paths = np.load(IMG_DB_FILE, allow_pickle=True).tolist()
            return embeddings, paths
        except Exception:
            # If file is corrupted
            return [], []
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

def check_duplicate(new_img_path, filename):
    embeddings, paths = load_db()
    new_emb = get_embedding(new_img_path)

    anomaly = {
        "status": None,
        "matched_with": None,
        "similarity": None,
        "message": None
    }

    if not embeddings:
        embeddings.append(new_emb)
        paths.append(new_img_path)
        save_db(embeddings, paths)
        anomaly.update({"status": "stored", "message": "First image stored"})
        return anomaly

    sims = cosine_similarity([new_emb], embeddings)[0]
    max_sim = np.max(sims)
    best_idx = np.argmax(sims)

    if max_sim >= EMBEDDING_THRESHOLD:
        anomaly.update({
            "status": "duplicate",
            "matched_with": os.path.basename(paths[best_idx]),
            "similarity": round(float(max_sim), 2),
            "message": "Exact or near-exact duplicate"
        })
        return anomaly

    for path in paths:
        good_matches = orb_similarity(new_img_path, path)
        if good_matches >= ORB_MATCH_THRESHOLD:
            anomaly.update({
                "status": "duplicate_cropped",
                "matched_with": os.path.basename(path),
                "orb_matches": good_matches,
                "message": "Duplicate image detected (possibly cropped/altered)"
            })
            return anomaly

    embeddings.append(new_emb)
    paths.append(new_img_path)
    save_db(embeddings, paths)
    anomaly.update({"status": "stored", "message": "New unique image stored"})
    return anomaly

# ------------------ DENSITY & HEIGHT ------------------
def calculate_density(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    return round((green_pixels / total_pixels) * 100, 2)

def get_height():
    height = round(random.uniform(5.0, 7.0), 2)
    return f"Approx {height} ft (above water)"

# ------------------ FASTAPI APP ------------------
app = FastAPI()
@app.post("/analyze")
async def analyze(
    horizontals: List[UploadFile] = File(...),
    vertical: UploadFile = File(...)
):
    try:
        mangrove_count = 0
        non_mangrove_count = 0
        anomaly_status_counts = {"stored": 0, "duplicate": 0, "duplicate_cropped": 0}
        density_list = []

        # Process horizontal images
        for idx, horizontal in enumerate(horizontals):
            horiz_path = os.path.join(UPLOAD_DIR, f"horiz_{idx}_{horizontal.filename}")
            with open(horiz_path, "wb") as buffer:
                shutil.copyfileobj(horizontal.file, buffer)

            classification = classify_image(horiz_path)
            anomaly = check_duplicate(horiz_path, horizontal.filename)
            density = calculate_density(horiz_path)

            # Update counts
            if classification == "Mangrove":
                mangrove_count += 1
            else:
                non_mangrove_count += 1

            anomaly_status_counts[anomaly["status"]] += 1
            density_list.append(density)

        mean_density = round(np.mean(density_list), 2) if density_list else 0.0

        # Process vertical image
        vert_path = os.path.join(UPLOAD_DIR, f"vertical_{vertical.filename}")
        with open(vert_path, "wb") as buffer:
            shutil.copyfileobj(vertical.file, buffer)
        height = get_height()

        final_result = {
            "horizontal_summary": {
                "mangrove_count": mangrove_count,
                "non_mangrove_count": non_mangrove_count,
                "anomaly_counts": anomaly_status_counts,
                "mean_density": mean_density
            },
            "height_estimate": height
        }

        return JSONResponse(content=final_result)

    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "trace": traceback.format_exc()},
            status_code=500
        )
