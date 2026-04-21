import torch
import torchvision.models as models
import pickle

device = torch.device('cpu')

# Load state
state = torch.load('model.pth', map_location=device, weights_only=False)

# Get classes
with open('classes.pkl', 'rb') as f:
    loaded_classes = pickle.load(f)
classes = list(loaded_classes)

model_state = state.get("model_state", state) if isinstance(state, dict) else state.state_dict()
num_classes = model_state["fc.bias"].shape[0]

# Init model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(model_state)

model.to(device)
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224, device=device)
onnx_path = "model.onnx"

torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path, 
    export_params=True, 
    opset_version=11, 
    do_constant_folding=True, 
    input_names=['input'], 
    output_names=['output'], 
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"Exported to {onnx_path}")
