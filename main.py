from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
import torch
from torchvision import transforms
from PIL import Image
import pickle
import io
import os
import random

app = FastAPI()

# -----------------------------
# DATASET PATH (LOCAL + RENDER)
# -----------------------------
DATASET_PATH = "dataset"

# -----------------------------
# SERVE IMAGES
# -----------------------------
app.mount("/images", StaticFiles(directory=DATASET_PATH), name="images")

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cpu")

# -----------------------------
# LOAD FILES
# -----------------------------
with open("classes.pkl", "rb") as f:
    classes = pickle.load(f)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = torch.load("model.pth", map_location=device, weights_only=False)
model.to(device)
model.eval()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

    _, pred = torch.max(output, 1)
    return pred.item()

# -----------------------------
# ROOT
# -----------------------------
@app.get("/")
def home():
    return {"message": "Fashion API running 🚀"}

# -----------------------------
# MAIN API
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), brand: str = Form(default=None)):
    try:
        contents = await file.read()
        idx = predict_image(contents)

        if idx >= len(classes):
            return {"error": "Invalid prediction index"}

        category = classes[idx]

        # -----------------------------
        # FIND MATCHING FOLDER
        # -----------------------------
        category_folder = category.lower().replace(" ", "_")

        folder_path = os.path.join(DATASET_PATH, category_folder)

        if not os.path.exists(folder_path):
            return {
                "category": category,
                "recommendations": []
            }

        images = os.listdir(folder_path)

        # optional brand filter
        if brand:
            images = [img for img in images if brand.lower() in img.lower()]

        # random 5 images
        selected = random.sample(images, min(5, len(images)))

        # create URLs
        final_urls = [
            f"/images/{category_folder}/{img}" for img in selected
        ]

        return {
            "category": category,
            "recommendations": final_urls
        }

    except Exception as e:
        return {"error": str(e)}