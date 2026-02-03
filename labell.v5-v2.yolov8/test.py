from ultralytics import YOLO
import cv2

model = YOLO("./runs/detect/train3/weights/best.pt")

img = cv2.imread("20.png")
results = model(img, conf=0.15)

results[0].show()
