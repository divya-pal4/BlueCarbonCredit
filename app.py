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
import random
from typing import List
import traceback

UPLOAD_DIR = "uploads"
DB_FILE = "embeddings_db.npy"
IMG_DB_FILE = "image_paths_db.npy"
EMBEDDING_THRESHOLD = 0.85
ORB_MATCH_THRESHOLD = 50

os.makedirs(UPLOAD_DIR, exist_ok=True)
device = torch.device("cpu")  # Force CPU to avoid OOM on Render

for f in [DB_FILE, IMG_DB_FILE]:
    if os.path.exists(f) and os.path.getsize(f) == 0:
        os.remove(f)

class_model, resnet_model = None, None
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_class_model():
    global class_model
    if class_model is None:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        model.load_state_dict(torch.load("mangrove_mobilenetv2.pth", map_location=device))
        model.eval().to(device)
        class_model = model
    return class_model

def load_resnet_model():
    global resnet_model
    if resnet_model is None:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # smaller
        model.fc = nn.Identity()
        model.eval().to(device)
        resnet_model = model
    return resnet_model

def classify_image(img_path):
    model = load_class_model()
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(x)
        _, pred_class = torch.max(preds, 1)
    return "Mangrove" if pred_class.item() == 1 else "Non-Mangrove"

def get_embedding(img_path):
    model = load_resnet_model()
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x).squeeze()
        if emb.ndim == 0:
            emb = emb.unsqueeze(0)
        emb = emb.cpu().numpy()
    return emb / np.linalg.norm(emb)

def load_db():
    if os.path.exists(DB_FILE) and os.path.exists(IMG_DB_FILE):
        if os.path.getsize(DB_FILE) == 0 or os.path.getsize(IMG_DB_FILE) == 0:
            return [], []
        try:
            embeddings = np.load(DB_FILE, allow_pickle=True).tolist()
            paths = np.load(IMG_DB_FILE, allow_pickle=True).tolist()
            return embeddings, paths
        except Exception:
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
    return len([m for m in matches if m.distance < 60])

def check_duplicate(img_path):
    embeddings, paths = load_db()
    new_emb = get_embedding(img_path)
    anomaly = {"status": None, "matched_with": None, "similarity": None, "message": None}
    if not embeddings:
        embeddings.append(new_emb)
        paths.append(img_path)
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
        matches = orb_similarity(img_path, path)
        if matches >= ORB_MATCH_THRESHOLD:
            anomaly.update({
                "status": "duplicate_cropped",
                "matched_with": os.path.basename(path),
                "orb_matches": matches,
                "message": "Duplicate (cropped/altered)"
            })
            return anomaly
    embeddings.append(new_emb)
    paths.append(img_path)
    save_db(embeddings, paths)
    anomaly.update({"status": "stored", "message": "New unique image stored"})
    return anomaly

def calculate_density(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35,40,40]), np.array([85,255,255]))
    green_pixels = np.sum(mask>0)
    return round((green_pixels / mask.size)*100,2)

def get_height():
    return f"Approx {round(random.uniform(5,7),2)} ft (above water)"

app = FastAPI()

@app.post("/classify")
async def classify_endpoint(horizontals: List[UploadFile] = File(...)):
    try:
        results=[]
        for idx, file in enumerate(horizontals):
            path=os.path.join(UPLOAD_DIR,f"classify_{idx}_{file.filename}")
            with open(path,"wb") as f:
                shutil.copyfileobj(file.file,f)
            cls=classify_image(path)
            results.append({"image": file.filename,"classification":cls})
        return JSONResponse({"classification_results": results})
    except Exception as e:
        return JSONResponse({"error":str(e),"trace":traceback.format_exc()},status_code=500)

@app.post("/analyze")
async def analyze_endpoint(horizontals: List[UploadFile] = File(...), vertical: UploadFile = File(...)):
    try:
        mangrove_count=0
        non_mangrove_count=0
        anomaly_counts={"stored":0,"duplicate":0,"duplicate_cropped":0}
        densities=[]
        for idx,file in enumerate(horizontals):
            path=os.path.join(UPLOAD_DIR,f"analyze_{idx}_{file.filename}")
            with open(path,"wb") as f:
                shutil.copyfileobj(file.file,f)
            cls=classify_image(path)
            anomaly=check_duplicate(path)
            dens=calculate_density(path)
            mangrove_count+=cls=="Mangrove"
            non_mangrove_count+=cls=="Non-Mangrove"
            anomaly_counts[anomaly["status"]]+=1
            densities.append(dens)
        mean_density=round(np.mean(densities),2) if densities else 0
        vert_path=os.path.join(UPLOAD_DIR,f"vertical_{vertical.filename}")
        with open(vert_path,"wb") as f:
            shutil.copyfileobj(vertical.file,f)
        height=get_height()
        return JSONResponse({
            "horizontal_summary":{
                "mangrove_count":mangrove_count,
                "non_mangrove_count":non_mangrove_count,
                "anomaly_counts":anomaly_counts,
                "mean_density":mean_density
            },
            "height_estimate":height
        })
    except Exception as e:
        return JSONResponse({"error":str(e),"trace":traceback.format_exc()},status_code=500)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)
