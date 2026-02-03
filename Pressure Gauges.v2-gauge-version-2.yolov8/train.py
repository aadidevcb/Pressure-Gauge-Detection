from ultralytics import YOLO
import torch

def train_model():
    # Verify hardware
    print(f"Using Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Load the model
    model = YOLO("yolov8n.pt")

    # Start training
    model.train(
        data="data.yaml",
        epochs=80,
        imgsz=640,
        batch=8,           # Good for your 4GB VRAM
        pretrained=True,
        device=0,          # GPU index 0
        workers=4          # Now safe to use
    )

if __name__ == '__main__':
    train_model()