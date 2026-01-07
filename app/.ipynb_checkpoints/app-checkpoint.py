import gradio as gr
from PIL import Image
from src.predict import predict_image

def classify(image):
    label, confidence = predict_image(image)
    return f"Prediction: {label}\nConfidence: {confidence:.2f}"

interface = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(),
    title="CIFAR-10 Image Classifier",
    description="Upload an image and get a prediction"
)

if __name__ == "__main__":
    interface.launch()
