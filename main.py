from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import pickle
import io
import os
import random
from collections import defaultdict
import torchvision.models as models

app = FastAPI()

# -----------------------------
# CORS (for frontend access)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# DATASET PATH (LOCAL + RENDER)
# -----------------------------
# Look for full dataset first (categorized local layout), then flat dataset, then fast dataset
if os.path.exists("archive/categorized_images"):
    DATASET_PATH = "archive/categorized_images"
elif os.path.exists("archive/images"):
    DATASET_PATH = "archive/images"
else:
    DATASET_PATH = "dataset"

# -----------------------------
# SERVE IMAGES & FRONTEND
# -----------------------------
if os.path.exists(DATASET_PATH):
    app.mount("/images", StaticFiles(directory=DATASET_PATH), name="images")

if not os.path.exists("frontend"):
    os.makedirs("frontend")
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cpu")

# -----------------------------
# LOAD FILES
# -----------------------------
# -----------------------------
# LOAD FILES & MODEL
# -----------------------------
state = torch.load("model.pth", map_location=device, weights_only=False)

try:
    with open("image_paths.pkl", "rb") as f:
        image_paths = pickle.load(f)
except Exception as e:
    print(f"[WARN] Failed to load image_paths.pkl: {e}")
    image_paths = []


# Look for classes in the state dict or fallback to classes.pkl
loaded_classes = state.get("classes") if isinstance(state, dict) and "classes" in state else None
if loaded_classes is None:
    with open("classes.pkl", "rb") as f:
        loaded_classes = pickle.load(f)

classes = list(loaded_classes)
model_state = state.get("model_state", state) if isinstance(state, dict) else state.state_dict()

# Get FC size to accommodate potentially mismatched classes.pkl
num_classes = model_state["fc.bias"].shape[0]

# Pad classes if the model has more outputs than classes in pkl
if num_classes > len(classes):
    classes += [f"UnknownClass_{i}" for i in range(len(classes), num_classes)]

if not isinstance(state, dict):
    model = state
else:
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(model_state)

model.to(device)
model.eval()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ---------------------------------------------------------
# BUILD CATEGORY → IMAGE MAPPING (handles both layouts)
# ---------------------------------------------------------
# Layout A: dataset/<Category>/<image>.jpg  (subfolders)
# Layout B: dataset/<image>.jpg             (flat)
# We detect which layout is present and build the map.

def normalize_name(name: str) -> str:
    """Normalize a category name for folder matching.
    'T-Shirts' -> 'tshirts', 'Casual Shoes' -> 'casualshoes'
    """
    return name.lower().replace("-", "").replace(" ", "").replace("_", "")


def build_category_map():
    """
    Scan dataset/ and build category -> [image filenames] mapping.
    Supports both subfolder and flat layouts.
    """
    cat_to_images = defaultdict(list)
    has_subfolders = False

    if not os.path.exists(DATASET_PATH):
        print(f"[WARN] Dataset path '{DATASET_PATH}' not found!")
        return cat_to_images, False

    # Check if dataset has subfolders (Layout A)
    entries = os.listdir(DATASET_PATH)
    subdirs = [e for e in entries if os.path.isdir(os.path.join(DATASET_PATH, e))]

    if subdirs:
        # ---- LAYOUT A: Subfolders ----
        has_subfolders = True
        print(f"[INFO] Detected subfolder layout with {len(subdirs)} category folders")

        # Build normalized lookup: normalize(folder_name) -> actual_folder_name
        folder_lookup = {}
        for folder in subdirs:
            key = normalize_name(folder)
            folder_lookup[key] = folder

        # Map each class to its matching folder
        for cls in classes:
            cls_key = normalize_name(cls)
            matched_folder = folder_lookup.get(cls_key)

            if matched_folder:
                folder_path = os.path.join(DATASET_PATH, matched_folder)
                images = [
                    f for f in os.listdir(folder_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
                ]
                cat_to_images[cls] = images
                # Store the actual folder name for URL building
                cat_to_images[f"__folder__{cls}"] = matched_folder
            else:
                print(f"[WARN] No folder match for class '{cls}' (normalized: '{cls_key}')")
    else:
        # ---- LAYOUT B: Flat ----
        print(f"[INFO] Detected flat layout with {len(entries)} files")

        # Use image_paths.pkl to map category -> filenames
        local_files = set(entries)
        for path in image_paths:
            parts = path.replace("\\", "/").split("/")
            if len(parts) >= 2:
                category = parts[-2]
                filename = parts[-1]
                if filename in local_files:
                    cat_to_images[category].append(filename)

    return cat_to_images, has_subfolders


category_image_map, SUBFOLDER_LAYOUT = build_category_map()

# Log stats
mapped_count = sum(
    len(v) for k, v in category_image_map.items() if not k.startswith("__folder__")
)
cats_with_images = sum(
    1 for k, v in category_image_map.items() if not k.startswith("__folder__") and v
)
print(f"[INFO] {len(classes)} classes | {cats_with_images} with images | {mapped_count} total mapped images")


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


def get_recommendations(category: str, brand: str = None, count: int = 4):
    """
    Get image recommendations for a predicted category.
    Returns up to `count` image URLs.
    """
    images = category_image_map.get(category, [])

    if not images:
        # Fallback: return random images from any category that has images
        all_images = []
        for k, v in category_image_map.items():
            if not k.startswith("__folder__") and v:
                if SUBFOLDER_LAYOUT:
                    folder = category_image_map.get(f"__folder__{k}", k)
                    all_images.extend([(folder, img) for img in v])
                else:
                    all_images.extend([("", img) for img in v])
        if all_images:
            selected = random.sample(all_images, min(count, len(all_images)))
            if SUBFOLDER_LAYOUT:
                return [f"/images/{folder}/{img}" for folder, img in selected]
            else:
                return [f"/images/{img}" for _, img in selected]
        return []

    # Optional brand filter
    if brand:
        filtered = [img for img in images if brand.lower() in img.lower()]
        if filtered:
            images = filtered

    # Select top N random images
    selected = random.sample(images, min(count, len(images)))

    # Build URLs based on layout
    if SUBFOLDER_LAYOUT:
        folder = category_image_map.get(f"__folder__{category}", category)
        return [f"/images/{folder}/{img}" for img in selected]
    else:
        return [f"/images/{img}" for img in selected]


# -----------------------------
# ROOT
# -----------------------------
@app.get("/", response_class=FileResponse)
def home():
    index_path = os.path.join("frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Fashion API frontend not built yet."}


# -----------------------------
# DEBUG: LIST CATEGORIES
# -----------------------------
@app.get("/categories")
def list_categories():
    """List all categories and how many images are mapped to each."""
    result = {}
    for cls in classes:
        count = len(category_image_map.get(cls, []))
        result[cls] = count
    return {
        "total_classes": len(classes),
        "layout": "subfolders" if SUBFOLDER_LAYOUT else "flat",
        "categories": result,
    }


# -----------------------------
# MAIN API
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), brand: str = Form(default=None)):
    try:
        contents = await file.read()
        idx = predict_image(contents)

        if idx >= len(classes):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid prediction index: {idx} (max: {len(classes) - 1})"
            )

        category = classes[idx]
        recommendations = get_recommendations(category, brand=brand, count=4)

        return {
            "category": category,
            "confidence_index": idx,
            "total_recommendations": len(recommendations),
            "recommendations": recommendations,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# -----------------------------
# ADVISOR DATA & ENDPOINTS
# -----------------------------
baseline_retail_prices = {
    "Luxury": {
        "Gucci": {"shirt": 700, "jeans": 800, "dresses": 2500, "leather jacket": 4500, "handbags": 3000, "tshirts": 400}
    },
    "Mid-Range": {
        "Zara": {"shirt": 50, "jeans": 60, "dresses": 80, "leather jacket": 150, "tshirts": 30},
        "H&M": {"shirt": 30, "jeans": 40, "dresses": 60, "tshirts": 20}
    }
}

brand_size_data = {
    "Zara": {
        "shirt": [
            {"size": "S", "chest": 96, "waist": 84, "tier": "Fast Fashion"},
            {"size": "M", "chest": 100, "waist": 88, "tier": "Fast Fashion"},
            {"size": "L", "chest": 104, "waist": 92, "tier": "Fast Fashion"},
            {"size": "XL", "chest": 108, "waist": 96, "tier": "Fast Fashion"},
        ],
        "jeans": [
            {"size": "28", "waist": 71, "hip": 89, "tier": "Fast Fashion"},
            {"size": "30", "waist": 76, "hip": 94, "tier": "Fast Fashion"},
            {"size": "32", "waist": 81, "hip": 99, "tier": "Fast Fashion"},
        ],
        "dresses": [
            {"size": "S", "bust": 84, "waist": 68, "hip": 92, "tier": "Fast Fashion"},
            {"size": "M", "bust": 88, "waist": 72, "hip": 96, "tier": "Fast Fashion"},
            {"size": "L", "bust": 92, "waist": 76, "hip": 100, "tier": "Fast Fashion"},
        ]
    },
    "H&M": {
        "shirt": [
            {"size": "S", "chest": 98, "waist": 86, "tier": "Fast Fashion"},
            {"size": "M", "chest": 102, "waist": 90, "tier": "Fast Fashion"},
            {"size": "L", "chest": 106, "waist": 94, "tier": "Fast Fashion"},
            {"size": "XL", "chest": 110, "waist": 98, "tier": "Fast Fashion"},
        ],
        "jeans": [
            {"size": "29", "waist": 74, "hip": 92, "tier": "Fast Fashion"},
            {"size": "31", "waist": 79, "hip": 97, "tier": "Fast Fashion"},
            {"size": "33", "waist": 84, "hip": 102, "tier": "Fast Fashion"},
        ],
         "dresses": [
            {"size": "S", "bust": 86, "waist": 70, "hip": 94, "tier": "Fast Fashion"},
            {"size": "M", "bust": 90, "waist": 74, "hip": 98, "tier": "Fast Fashion"},
            {"size": "L", "bust": 94, "waist": 78, "hip": 102, "tier": "Fast Fashion"},
        ]
    },
    "Levi's": {
        "jeans": [
            {"size": "28", "waist": 72, "hip": 90, "tier": "Premium"},
            {"size": "30", "waist": 77, "hip": 95, "tier": "Premium"},
            {"size": "32", "waist": 82, "hip": 100, "tier": "Premium"},
            {"size": "34", "waist": 87, "hip": 105, "tier": "Premium"},
        ],
        "shirt": [
            {"size": "S", "chest": 94, "waist": 82, "tier": "Premium"},
            {"size": "M", "chest": 98, "waist": 86, "tier": "Premium"},
            {"size": "L", "chest": 102, "waist": 90, "tier": "Premium"},
        ]
    },
    "Gucci": {
        "shirt": [
            {"size": "48IT", "chest": 92, "waist": 80, "tier": "Luxury"},
            {"size": "50IT", "chest": 96, "waist": 84, "tier": "Luxury"},
            {"size": "52IT", "chest": 100, "waist": 88, "tier": "Luxury"},
        ],
        "leather jacket": [
            {"size": "S", "chest": 94, "waist": 82, "tier": "Luxury"},
            {"size": "M", "chest": 98, "waist": 86, "tier": "Luxury"},
            {"size": "L", "chest": 102, "waist": 90, "tier": "Luxury"},
        ],
        "handbags": [
            {"size": "One Size", "dimensions": "30x20x10cm", "tier": "Luxury"}
        ]
    }
}

class ResaleRequest(BaseModel):
    brand: str
    category: str
    condition: str = "Good"

@app.post("/estimate-resale")
def estimate_resale(request: ResaleRequest):
    brand_lower = request.brand.lower().replace(" ", "").replace("-", "")
    category_lower = request.category.lower().replace(" ", "").replace("-", "")

    base_price = 0
    brand_tier = "Unknown"

    for tier, brands_in_tier in baseline_retail_prices.items():
        for b, items in brands_in_tier.items():
            if b.lower().replace(" ", "").replace("-", "") == brand_lower:
                brand_tier = tier
                if category_lower in [c.lower().replace(" ", "").replace("-", "") for c in items]:
                    category_key = next((c for c in items if c.lower().replace(" ", "").replace("-", "") == category_lower), request.category)
                    base_price = items.get(category_key, 0)
                break
        if base_price > 0:
            break

    if base_price == 0:
        return {"error": f"Resale value estimation not available for {request.brand} {request.category}."}

    depreciation_multipliers = {"Like-New": 0.8, "Good": 0.6, "Fair": 0.4, "Poor": 0.2}
    condition_multiplier = depreciation_multipliers.get(request.condition, 0.3)

    category_adjustment = 1.0
    if "leather" in category_lower and "jacket" in category_lower:
        category_adjustment = 1.2
    elif "handbags" in category_lower:
        category_adjustment = 1.1
    elif "tshirt" in category_lower or "shirt" in category_lower:
        category_adjustment = 0.9

    estimated_value = base_price * condition_multiplier * category_adjustment

    min_price = estimated_value * 0.9
    max_price = estimated_value * 1.1

    return {
        "brand": request.brand,
        "category": request.category,
        "tier": brand_tier,
        "min_price": round(min_price, 2),
        "max_price": round(max_price, 2)
    }

class SizeRequest(BaseModel):
    target_brand: str
    category: str
    chest: float = None
    waist: float = None
    bust: float = None
    hip: float = None
    dimensions: str = None

@app.post("/translate-size")
def translate_size(req: SizeRequest):
    user_measurements = {k: v for k, v in req.dict().items() if v is not None and k not in ["target_brand", "category"]}
    category_lower = req.category.lower().replace(" ", "").replace("-", "")
    target_brand_norm = req.target_brand.lower().replace(" ", "").replace("-", "")

    target_brand_key = next((k for k in brand_size_data if k.lower().replace(" ", "").replace("-", "") == target_brand_norm), None)

    if not target_brand_key or category_lower not in [k.lower().replace(" ", "").replace("-", "") for k in brand_size_data[target_brand_key]]:
        return {"error": f"Size data not available for brand '{req.target_brand}' or category '{req.category}'."}

    category_key_target = next((k for k in brand_size_data[target_brand_key] if k.lower().replace(" ", "").replace("-", "") == category_lower), None)

    best_target_size = None
    min_diff_score = float('inf')

    for target_item in brand_size_data[target_brand_key][category_key_target]:
        current_diff_score = 0
        measurement_keys = ["chest", "waist", "bust", "hip"]
        for key in measurement_keys:
            if key in user_measurements and key in target_item:
                current_diff_score += abs(user_measurements[key] - target_item[key]) * 2
            elif key in user_measurements or key in target_item:
                current_diff_score += 10

        if "dimensions" in user_measurements and "dimensions" in target_item:
            if user_measurements["dimensions"] == target_item["dimensions"]:
                current_diff_score = 0
            else:
                current_diff_score += 50
        elif "dimensions" in user_measurements or "dimensions" in target_item:
             current_diff_score += 25

        if current_diff_score < min_diff_score:
            min_diff_score = current_diff_score
            best_target_size = target_item["size"]

    if best_target_size:
        return {"recommended_size": best_target_size}
    return {"error": f"Could not find a suitable size in '{target_brand_key}' for '{category_key_target}'."}