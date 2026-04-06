from fastapi import FastAPI, UploadFile, File, Form
import torch
from torchvision import transforms
from PIL import Image
import pickle
import io

app = FastAPI()

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
    return {"message": "Fashion API is running 🚀"}

# -----------------------------
# MAIN API
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), brand: str = Form(default=None)):
    try:
        contents = await file.read()

        idx = predict_image(contents)

        if idx >= len(classes):
            return {"error": f"Index {idx} out of range"}

        category = classes[idx]

        # TEMP: no dataset (for deployment)
        final_urls = []

        return {
            "category": category,
            "recommendations": final_urls
        }

    except Exception as e:
        return {"error": str(e)}