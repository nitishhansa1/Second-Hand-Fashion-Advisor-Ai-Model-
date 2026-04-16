import torch
from torchvision import transforms
from PIL import Image
import pickle

device = torch.device("cpu")

with open("classes.pkl", "rb") as f:
    classes = pickle.load(f)

model = torch.load("model.pth", map_location=device, weights_only=False)
model.eval()

# Transform 1: No Normalization
transform_no_norm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Transform 2: ImageNet Normalization
transform_norm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = "archive/images/10000.jpg"  # We know this is a White Skirt
image = Image.open(image_path).convert("RGB")

t1 = transform_no_norm(image).unsqueeze(0).to(device)
t2 = transform_norm(image).unsqueeze(0).to(device)

with torch.no_grad():
    out1 = model(t1)
    out2 = model(t2)

_, p1 = torch.max(out1, 1)
_, p2 = torch.max(out2, 1)

print("Original script prediction:", classes[p1.item()])
print("Normalized script prediction:", classes[p2.item()])
