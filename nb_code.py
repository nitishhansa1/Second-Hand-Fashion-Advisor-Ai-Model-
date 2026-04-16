!pip install transformers torch torchvision faiss-cpu

import torch
from transformers import AutoImageProcessor, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")
model.to(device)
model.eval()


from google.colab import files
uploaded = files.upload()

from PIL import Image

image_name = list(uploaded.keys())[0]
image = Image.open(image_name).convert("RGB")

inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)

print("Embedding shape:", embeddings.shape)

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

image_embeddings = []
image_names = []

for img_name in uploaded.keys():
    image = Image.open(img_name).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

        # ✅ 1. Take CLS token
        embedding = outputs.last_hidden_state[:, 0]

        # ✅ 2. Normalize (VERY IMPORTANT)
        embedding = F.normalize(embedding, p=2, dim=1)

    image_embeddings.append(embedding.cpu().numpy())
    image_names.append(img_name)

image_embeddings = np.vstack(image_embeddings)

print("Total embeddings:", image_embeddings.shape)

!pip install faiss-cpu -q

import faiss

dimension = 768
index = faiss.IndexFlatIP(dimension)  # use inner product for cosine similarity

index.add(image_embeddings)

print("Index size:", index.ntotal)

query_embedding = image_embeddings[0].reshape(1, -1)

distances, indices = index.search(query_embedding, k=3)

print("Similar images:")
for i in indices[0]:
    print(image_names[i])

# =====================================================
# SMART SIZE RECOMMENDATION SYSTEM (MEN + WOMEN)
# =====================================================

# ---------------------------
# Example Brand Size Charts
# (All measurements in cm)
# ---------------------------

brand_size_chart = {
    "Zara": {
        "mens": {
            "shirt": [
                {"size": "S", "chest": 96, "waist": 84},
                {"size": "M", "chest": 100, "waist": 88},
                {"size": "L", "chest": 104, "waist": 92},
                {"size": "XL", "chest": 108, "waist": 96},
            ],
            "pants": [
                {"size": "30", "waist": 76},
                {"size": "32", "waist": 81},
                {"size": "34", "waist": 86},
                {"size": "36", "waist": 91},
            ]
        },
        "womens": {
            "dress": [
                {"size": "S", "bust": 84, "waist": 66, "hip": 90},
                {"size": "M", "bust": 88, "waist": 70, "hip": 94},
                {"size": "L", "bust": 92, "waist": 74, "hip": 98},
                {"size": "XL", "bust": 96, "waist": 78, "hip": 102},
            ],
            "jeans": [
                {"size": "26", "waist": 66, "hip": 90},
                {"size": "28", "waist": 71, "hip": 95},
                {"size": "30", "waist": 76, "hip": 100},
                {"size": "32", "waist": 81, "hip": 105},
            ]
        }
    }
}


# ---------------------------
# Recommendation Function
# ---------------------------

def recommend_size(user, brand, gender, category, fit_pref="regular"):

    if brand not in brand_size_chart:
        return "Brand not found"

    if gender not in brand_size_chart[brand]:
        return "Gender category not found"

    if category not in brand_size_chart[brand][gender]:
        return "Category not found"

    sizes = brand_size_chart[brand][gender][category]
    scored_sizes = []

    for item in sizes:
        score = 0

        # MEN LOGIC
        if gender == "mens":
            if "chest" in item:
                score += 2 * abs(user.get("chest", 0) - item["chest"])
            if "waist" in item:
                score += abs(user.get("waist", 0) - item["waist"])

        # WOMEN LOGIC
        elif gender == "womens":
            if "bust" in item:
                score += 2 * abs(user.get("bust", 0) - item["bust"])
            if "waist" in item:
                score += abs(user.get("waist", 0) - item["waist"])
            if "hip" in item:
                score += abs(user.get("hip", 0) - item["hip"])

        # Fit preference adjustment
        if fit_pref == "tight":
            score += 2
        elif fit_pref == "loose":
            score -= 2

        scored_sizes.append((item["size"], score))

    # Sort by best score
    scored_sizes.sort(key=lambda x: x[1])

    best_size = scored_sizes[0][0]
    alt_size = scored_sizes[1][0] if len(scored_sizes) > 1 else None
    best_score = scored_sizes[0][1]

    # Simple confidence logic
    confidence = max(0, 100 - best_score * 1.5)

    return {
        "recommended_size": best_size,
        "alternative_size": alt_size,
        "confidence_percent": round(confidence, 1)
    }


# ---------------------------
# TEST CASES
# ---------------------------

print("---- MEN TEST ----")
user_male = {"chest": 102, "waist": 90}
print(recommend_size(user_male, "Zara", "mens", "shirt", "regular"))

print("\n---- WOMEN TEST ----")
user_female = {"bust": 89, "waist": 71, "hip": 95}
print(recommend_size(user_female, "Zara", "womens", "dress", "regular"))

# =====================================================
# SYNTHETIC SIZE DATASET GENERATOR + TRAINING
# =====================================================

!pip install xgboost -q

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import random

# -----------------------------------
# Brand Size Chart (Example)
# -----------------------------------

brand_size_chart = {
    "Zara": {
        "mens": {
            "shirt": [
                {"size": "S", "chest": 96, "waist": 84},
                {"size": "M", "chest": 100, "waist": 88},
                {"size": "L", "chest": 104, "waist": 92},
                {"size": "XL", "chest": 108, "waist": 96},
            ]
        },
        "womens": {
            "dress": [
                {"size": "S", "bust": 84, "waist": 66, "hip": 90},
                {"size": "M", "bust": 88, "waist": 70, "hip": 94},
                {"size": "L", "bust": 92, "waist": 74, "hip": 98},
                {"size": "XL", "bust": 96, "waist": 78, "hip": 102},
            ]
        }
    }
}

# -----------------------------------
# Generate Synthetic Data
# -----------------------------------

def generate_data(n_samples=2000):
    rows = []

    for _ in range(n_samples):
        brand = "Zara"
        gender = random.choice(["mens", "womens"])

        if gender == "mens":
            category = "shirt"
            size_chart = brand_size_chart[brand][gender][category]

            # Pick a true size randomly
            chosen = random.choice(size_chart)

            # Generate user measurement around that size
            chest = chosen["chest"] + random.randint(-3, 3)
            waist = chosen["waist"] + random.randint(-3, 3)

            rows.append([chest, waist, 0, brand, gender, category, chosen["size"]])

        else:
            category = "dress"
            size_chart = brand_size_chart[brand][gender][category]

            chosen = random.choice(size_chart)

            bust = chosen["bust"] + random.randint(-3, 3)
            waist = chosen["waist"] + random.randint(-3, 3)
            hip = chosen["hip"] + random.randint(-3, 3)

            rows.append([bust, waist, hip, brand, gender, category, chosen["size"]])

    return pd.DataFrame(rows, columns=["chest_or_bust", "waist", "hip", "brand", "gender", "category", "size"])


df = generate_data(3000)

# -----------------------------------
# Encode Categorical Features
# -----------------------------------

brand_enc = LabelEncoder()
gender_enc = LabelEncoder()
cat_enc = LabelEncoder()
size_enc = LabelEncoder()

df["brand"] = brand_enc.fit_transform(df["brand"])
df["gender"] = gender_enc.fit_transform(df["gender"])
df["category"] = cat_enc.fit_transform(df["category"])
df["size"] = size_enc.fit_transform(df["size"])

# -----------------------------------
# Train-Test Split
# -----------------------------------

X = df[["chest_or_bust", "waist", "hip", "brand", "gender", "category"]]
y = df["size"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------
# Train Model
# -----------------------------------

model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

# -----------------------------------
# Evaluate
# -----------------------------------

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# -----------------------------------
# Test New User
# -----------------------------------

new_user = pd.DataFrame([[102, 90, 0,
                          brand_enc.transform(["Zara"])[0],
                          gender_enc.transform(["mens"])[0],
                          cat_enc.transform(["shirt"])[0]]],
                        columns=["chest_or_bust", "waist", "hip", "brand", "gender", "category"])

prediction = model.predict(new_user)
predicted_size = size_enc.inverse_transform(prediction)

print("Recommended Size:", predicted_size[0])

pip install kaggle

!pip install torch torchvision tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

!pip install kaggle

from google.colab import files
files.upload()  # upload kaggle.json

from google.colab import files
files.upload()

from google.colab import drive
drive.mount('/content/drive')

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d paramaggarwal/fashion-product-images-small

!unzip -o fashion-product-images-small.zip

import os
print(os.listdir())

import pandas as pd
import os
import shutil
from tqdm import tqdm

# ✅ Correct paths
df = pd.read_csv("styles.csv", on_bad_lines='skip')

image_dir = "images"
output_dir = "/content/fashion_data"

os.makedirs(output_dir, exist_ok=True)

# 🔥 (optional) take subset for faster training
df = df.sample(5000)

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_id = str(row["id"]) + ".jpg"
    category = str(row["articleType"])

    src = os.path.join(image_dir, img_id)
    dest_folder = os.path.join(output_dir, category)

    if os.path.exists(src):
        os.makedirs(dest_folder, exist_ok=True)
        shutil.copy(src, os.path.join(dest_folder, img_id))

print(os.listdir("/content/fashion_data"))

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder("/content/fashion_data", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

import pickle

classes = dataset.classes

print("Total classes:", len(classes))

with open("/content/drive/MyDrive/classes.pkl", "wb") as f:
    pickle.dump(classes, f)

print("✅ classes.pkl saved")

import torch
import torch.nn as nn
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18(pretrained=True)

num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

from tqdm import tqdm

epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy:", 100 * correct / total)

from google.colab import drive
drive.mount('/content/drive')

import os
print(os.listdir("/content/drive"))

torch.save(model.state_dict(), "/content/drive/MyDrive/model.pth")

import os

print(os.listdir("/content/drive/MyDrive"))

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from google.colab import files
from google.colab import drive
import os

# -----------------------------
# 1. MOUNT DRIVE
# -----------------------------
drive.mount('/content/drive')

# -----------------------------
# 2. MODEL PATH
# -----------------------------
model_path = "/content/drive/MyDrive/model.pth"

print("Model exists:", os.path.exists(model_path))

# -----------------------------
# 3. DEFINE MODEL (FIXED)
# -----------------------------
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 112)  # ✅ FIXED

# -----------------------------
# 4. LOAD MODEL
# -----------------------------
model = torch.load(model_path, map_location="cpu", weights_only=False)
model.eval()

# -----------------------------
# 5. TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# 6. PREDICT FUNCTION
# -----------------------------
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)

    return pred.item()

# -----------------------------
# 7. UPLOAD
# -----------------------------
uploaded = files.upload()

if uploaded:
    image_path = list(uploaded.keys())[0]
    result = predict(image_path)

    print("✅ Prediction Index:", result)

torch.save(model.state_dict(), "fashion_model.pth")

print(predict("shirt.jpg"))

import torch.nn as nn

feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

def get_embedding(image_path):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = feature_extractor(image)

    return features.view(-1).cpu().numpy()

import os
import numpy as np

base_path = "/content/fashion_data"

embeddings = []
image_paths = []

for category in os.listdir(base_path):
    folder = os.path.join(base_path, category)

    for img in os.listdir(folder)[:20]:  # ⚡ limit for speed
        path = os.path.join(folder, img)

        try:
            emb = get_embedding(path)
            embeddings.append(emb)
            image_paths.append(path)
        except:
            continue

embeddings = np.array(embeddings)

print("Embeddings shape:", embeddings.shape)

!pip install faiss-cpu

import faiss

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)

index.add(embeddings)

print("Total items indexed:", index.ntotal)

query = get_embedding("shirt.jpg").reshape(1, -1)

distances, indices = index.search(query, k=5)

print("Similar items:")
for i in indices[0]:
    print(image_paths[i])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    image = image.to(device)   # ✅ FIX

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)

    return pred.item()

user = {
    "chest": 101,
    "waist": 89,
    "height": 175,
    "fit": "regular"   # tight / regular / loose
}

brand_sizes = {
    "Zara": {
        "shirt": [
            {"size": "S", "chest": 96, "waist": 84},
            {"size": "M", "chest": 100, "waist": 88},
            {"size": "L", "chest": 104, "waist": 92},
            {"size": "XL", "chest": 108, "waist": 96},
        ]
    },
    "H&M": {
        "shirt": [
            {"size": "S", "chest": 98, "waist": 86},
            {"size": "M", "chest": 102, "waist": 90},
            {"size": "L", "chest": 106, "waist": 94},
            {"size": "XL", "chest": 110, "waist": 98},
        ]
    }
}

def recommend_size(user, brand, category):
    sizes = brand_sizes[brand][category]

    best = None
    best_score = float("inf")

    for item in sizes:
        score = 0

        # weight chest more (important)
        score += 2 * abs(user["chest"] - item["chest"])
        score += abs(user["waist"] - item["waist"])

        # fit preference adjustment
        if user["fit"] == "tight":
            score += 2
        elif user["fit"] == "loose":
            score -= 2

        if score < best_score:
            best_score = score
            best = item["size"]

    return best

user = {
    "chest": 101,
    "waist": 89,
    "height": 175,
    "fit": "regular"
}

print(recommend_size(user, "Zara", "shirt"))

def predict(image_path):
    from PIL import Image

    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)

    return dataset.classes[pred.item()]

def predict(image_path):
    from PIL import Image

    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)

    return dataset.classes[pred.item()]

def recommend_size(user, brand, category):
    sizes = brand_sizes[brand][category]

    best = None
    best_score = float("inf")

    for item in sizes:
        score = 2 * abs(user["chest"] - item["chest"]) + abs(user["waist"] - item["waist"])

        if user["fit"] == "tight":
            score += 2
        elif user["fit"] == "loose":
            score -= 2

        if score < best_score:
            best_score = score
            best = item["size"]

    return best

def fashion_advisor(image_path, user, brand):
    # Step 1: detect category
    category = predict(image_path)

    # Step 2: find similar items
    similar_items = find_similar(image_path)

    # Step 3: recommend size
    size = recommend_size(user, brand, category.lower())

    return {
        "category": category,
        "recommended_size": size,
        "similar_items": similar_items
    }

def find_similar(image_path, k=5):
    query = get_embedding(image_path).reshape(1, -1)

    distances, indices = index.search(query, k)

    results = []
    for i in indices[0]:
        results.append(image_paths[i])

    return results

def recommend_size(user, brand, category):
    category = category.lower()

    # ✅ Check if category exists
    if brand not in brand_sizes or category not in brand_sizes[brand]:
        return "Size not applicable"

    sizes = brand_sizes[brand][category]

    best = None
    best_score = float("inf")

    for item in sizes:
        score = 2 * abs(user["chest"] - item["chest"]) + abs(user["waist"] - item["waist"])

        if user["fit"] == "tight":
            score += 2
        elif user["fit"] == "loose":
            score -= 2

        if score < best_score:
            best_score = score
            best = item["size"]

    return best

size_categories = ["shirt", "tshirt", "jeans", "pants", "dress"]

def fashion_advisor(image_path, user, brand):

    # Step 1: detect category
    category = predict(image_path)

    # Step 2: find similar items
    similar_items = find_similar(image_path)

    # Step 3: size logic (SMART)
    if category.lower() in size_categories:
        size = recommend_size(user, brand, category)
    else:
        size = "Not needed"

    return {
        "category": category,
        "recommended_size": size,
        "similar_items": similar_items
    }

import os
import pickle

dataset_path = "/content/fashion_data"  # 👈 same path you used in ImageFolder

image_paths = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            image_paths.append(os.path.join(root, file))

print("Total images:", len(image_paths))

# SAVE TO DRIVE
with open("/content/drive/MyDrive/image_paths.pkl", "wb") as f:
    pickle.dump(image_paths, f)

print("✅ image_paths.pkl saved")

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from google.colab import drive
import pickle
import os

# -----------------------------
# 1. SET DEVICE (SAFE)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 2. MOUNT DRIVE
# -----------------------------
drive.mount('/content/drive')

# -----------------------------
# 3. PATHS
# -----------------------------
model_path = "/content/drive/MyDrive/model.pth"
classes_path = "/content/drive/MyDrive/classes.pkl"
image_paths_path = "/content/drive/MyDrive/image_paths.pkl"

# -----------------------------
# 4. LOAD FILES
# -----------------------------
print("Model exists:", os.path.exists(model_path))

with open(classes_path, "rb") as f:
    classes = pickle.load(f)

with open(image_paths_path, "rb") as f:
    image_paths = pickle.load(f)

# -----------------------------
# 5. LOAD MODEL
# -----------------------------
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device)
model.eval()

# -----------------------------
# 6. TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# 7. PREDICT FUNCTION
# -----------------------------
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)

    return pred.item()

# -----------------------------
# 8. FASHION ADVISOR FUNCTION
# -----------------------------
def fashion_advisor(image_path, user=None, brand=None):
    predicted_index = predict(image_path)
    predicted_category = classes[predicted_index]

    print("👕 Predicted Category:", predicted_category)

    # Normalize text for matching
    key = predicted_category.lower().replace("-", "").replace(" ", "")

    filtered_paths = [
        path for path in image_paths
        if key in path.lower().replace("-", "").replace(" ", "")
    ]

    # Optional: filter by brand
    if brand:
        filtered_paths = [
            path for path in filtered_paths
            if brand.lower() in path.lower()
        ]

    return {
        "category": predicted_category,
        "recommendations": filtered_paths[:5]  # top 5 results
    }

# -----------------------------
# 9. RUN
# -----------------------------
user = "Nitish"

result = fashion_advisor("shirt.jpg", user, "Zara")

print("\n🎯 FINAL RESULT:")
print(result)

torch.save(model, "/content/drive/MyDrive/model.pth")

import faiss
faiss.write_index(index, "/content/faiss.index")

import pickle

with open("/content/image_paths.pkl", "wb") as f:
    pickle.dump(image_paths, f)

with open("/content/classes.pkl", "wb") as f:
    pickle.dump(dataset.classes, f)

# SAVE MODEL DIRECTLY TO DRIVE
torch.save(model, "/content/drive/MyDrive/model.pth")

print("✅ model saved to Drive")

import pickle

classes = dataset.classes  # NOT train_dataset

with open("/content/drive/MyDrive/classes.pkl", "wb") as f:
    pickle.dump(classes, f)

print("✅ classes saved")

import os
import pickle

dataset_path = "/content/fashion_data"

image_paths = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            image_paths.append(os.path.join(root, file))

with open("/content/drive/MyDrive/image_paths.pkl", "wb") as f:
    pickle.dump(image_paths, f)

print("✅ image paths saved")

!cp /content/model.pth /content/drive/MyDrive/
!cp /content/faiss.index /content/drive/MyDrive/
!cp /content/image_paths.pkl /content/drive/MyDrive/
!cp /content/classes.pkl /content/drive/MyDrive/

!pip install faiss-cpu

faiss.write_index(index, "fashion_index.faiss")

import torch
import torchvision.models as models
from google.colab import drive

# -----------------------------
# 1. MOUNT DRIVE
# -----------------------------
drive.mount('/content/drive')

# -----------------------------
# 2. PATH
# -----------------------------
model_path = "/content/drive/MyDrive/model.pth"

# -----------------------------
# 3. DEFINE MODEL (IMPORTANT)
# -----------------------------
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 112)  # ✅ your trained classes

# -----------------------------
# 4. LOAD WEIGHTS (CORRECT WAY)
# -----------------------------
model = torch.load(model_path, map_location="cpu", weights_only=False)
model.eval()

# -----------------------------
# 5. SET EVAL MODE
# -----------------------------
model.eval()

print("✅ Model loaded correctly")

print(type(embeddings), embeddings.shape)

import faiss
import pickle
import torch
import os
from google.colab import drive

drive.mount('/content/drive')

base_path = "/content/drive/MyDrive/fashion_ai"
os.makedirs(base_path, exist_ok=True)

# Save FAISS
faiss.write_index(index, f"{base_path}/faiss.index")

# Save image paths
with open(f"{base_path}/image_paths.pkl", "wb") as f:
    pickle.dump(image_paths, f)

# Save model
torch.save(model, f"{base_path}/model.pth")

print("✅ Everything saved to Drive")

base_path = "/content/drive/MyDrive/fashion_ai"

image_dir = f"{base_path}/dataset/images"

classes = ["T-shirt", "Shirt", "Dress", ...]

import pickle

with open("/content/drive/MyDrive/classes.pkl", "wb") as f:
    pickle.dump(classes, f)

print("✅ classes.pkl saved")

print(os.listdir("/content/drive/MyDrive/fashion_ai"))

import os

dataset_path = "/content/fashion_data"

image_paths = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            image_paths.append(os.path.join(root, file))

print("Total images:", len(image_paths))

import pickle

with open("image_paths.pkl", "wb") as f:
    pickle.dump(image_paths, f)

print("✅ image_paths.pkl saved")

torch.save({
    "model_state": model.state_dict(),
    "classes": dataset.classes
}, "/content/drive/MyDrive/fashion_model.pth")