import os
import torch
from torchvision import transforms, models
import pickle
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# Detect dataset layout
if os.path.exists("archive/categorized_images"):
    DATASET_PATH = "archive/categorized_images"
elif os.path.exists("archive/images"):
    DATASET_PATH = "archive/images"
else:
    DATASET_PATH = "dataset"

# Load image paths (for flat dataset)
try:
    with open("image_paths.pkl", "rb") as f:
        image_paths = pickle.load(f)
except Exception:
    image_paths = []

# Load classes
with open("classes.pkl", "rb") as f:
    classes = pickle.load(f)

# Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset to handle both Flat and Subfolder layouts
class FashionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        entries = os.listdir(data_dir)
        subdirs = [e for e in entries if os.path.isdir(os.path.join(data_dir, e))]
        
        if subdirs:
            # Layout: Subfolders
            folder_lookup = {name.lower().replace("-", "").replace(" ", "").replace("_", ""): name for name in subdirs}
            for cls_idx, cls_name in enumerate(classes):
                key = cls_name.lower().replace("-", "").replace(" ", "").replace("_", "")
                if key in folder_lookup:
                    folder = folder_lookup[key]
                    folder_path = os.path.join(data_dir, folder)
                    for img_name in os.listdir(folder_path):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                            self.samples.append((os.path.join(folder_path, img_name), cls_idx))
        else:
            # Layout: Flat
            local_files = set(entries)
            cls_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            for path in image_paths:
                parts = path.replace("\\", "/").split("/")
                if len(parts) >= 2:
                    category = parts[-2]
                    filename = parts[-1]
                    if filename in local_files and category in cls_to_idx:
                        self.samples.append((os.path.join(data_dir, filename), cls_to_idx[category]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, path

# Load Model
device = torch.device("cpu")
state = torch.load("model.pth", map_location=device, weights_only=False)
model_state = state.get("model_state", state) if isinstance(state, dict) else state.state_dict()

# Account for mismatched FC layers
num_classes = model_state["fc.bias"].shape[0]
if num_classes > len(classes):
    classes += [f"UnknownClass_{i}" for i in range(len(classes), num_classes)]

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(model_state)
model.to(device)
model.eval()

# Check dataset
print(f"Loading dataset from {DATASET_PATH}")
dataset = FashionDataset(DATASET_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

print(f"Total images to evaluate: {len(dataset)}")

correct = 0
total = 0
class_correct = {i: 0 for i in range(len(classes))}
class_total = {i: 0 for i in range(len(classes))}

print("Starting evaluation...")
import time
start_time = time.time()

with torch.no_grad():
    for batch_idx, (images, labels, paths) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        for i in range(labels.size(0)):
            label = labels[i].item()
            pred = predicted[i].item()
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1
                
        if (batch_idx + 1) % 50 == 0:
            print(f"Processed {total}/{len(dataset)} images... Accuracy so far: {100 * correct / total:.2f}%")

print(f"\\n--- Evaluation Complete in {time.time() - start_time:.2f}s ---")
accuracy = 100 * correct / total if total > 0 else 0
print(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")

for i in range(len(classes)):
    if class_total[i] > 0:
        cls_acc = 100 * class_correct[i] / class_total[i]
        print(f"Accuracy for {classes[i]}: {cls_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
