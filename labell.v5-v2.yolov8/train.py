from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=120,      # more repetitions for small dataset
    imgsz=640,       # IMPORTANT: helps thin needles
    batch=8,
    pretrained=True
)
