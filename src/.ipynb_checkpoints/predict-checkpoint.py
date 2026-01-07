import torch
import torch.nn.functional as F
from PIL import Image

from src.model import SimpleCNN
from src.preprocess import get_image_transform

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def load_model(model_path="models/best_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleCNN(num_classes=10)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, device

def predict_image(pil_image):
    model, device = load_model()

    transform = get_image_transform()
    image = transform(pil_image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)

    label = CLASS_NAMES[predicted_idx.item()]
    return label, confidence.item()
