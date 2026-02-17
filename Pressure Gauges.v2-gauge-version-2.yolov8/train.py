from ultralytics import YOLO
import torch
import os

def train_on_m3():
    # Verify Apple Silicon GPU availability
    if not torch.backends.mps.is_available():
        print("MPS not found. Check your Python/PyTorch version.")
        return
    else:
        print("M3 GPU (MPS) detected. Starting high-performance training...")

    # Load the model
    # YOLOv8s (small) is a great balance for the M3 16GB
    model = YOLO('yolov8s.pt') 

    # Start training
    model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        device='mps',           # The secret sauce for Mac M3
        batch=16,               # Optimized for 16GB Unified Memory
        workers=8,              # Matches M3 CPU core count
        cache=True,             # Keep data in RAM to save SSD wear/speed up epochs
        optimizer='AdamW',      # Superior convergence for many datasets
        plots=True,             # Generate training curves
        save=True,
        project='pressure_gauge_detection',
        name='needle_v1_m3'
    )

if __name__ == '__main__':
    train_on_m3()
