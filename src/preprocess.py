import torch
from src.model import MNISTCNN
from src.preprocess import mnist_transform
from PIL import Image
import torch.nn.functional as F

device = "cpu"

model = MNISTCNN()
model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=device))
model.eval()

def predict_image(image):
    image = mnist_transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)
        pred = probs.argmax(dim=1).item()
        confidence = probs[0][pred].item()

    return pred, round(confidence, 3)
