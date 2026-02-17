import cv2
from ultralytics import YOLO
import argparse
import numpy as np
import math

def map_angle_to_value(angle, zero_angle, max_angle, min_value, max_value):
    # Ensure the angles are in a consistent range
    if angle < 0:
        angle += 360
    if zero_angle < 0:
        zero_angle += 360
    if max_angle < 0:
        max_angle += 360

    # Handle angle wrapping
    if max_angle < zero_angle:
        if angle > zero_angle or angle < max_angle:
            # Normal case
            pass
        else:
             # Angle is outside the valid range
             return None
    
    # Clamp the angle to the valid range
    angle = max(zero_angle, min(max_angle, angle))
    
    # Linear interpolation
    percentage = (angle - zero_angle) / (max_angle - zero_angle)
    value = min_value + percentage * (max_value - min_value)
    return value

# Parse command-line arguments
parser = argparse.ArgumentParser(description='YOLOv8 Live Inference with Gauge Reading')
parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for needle detection')
parser.add_argument('--p1', type=int, default=100, help='HoughCircles param1')
parser.add_argument('--p2', type=int, default=30, help='HoughCircles param2')
parser.add_argument('--min_rad', type=int, default=50, help='HoughCircles minRadius')
parser.add_argument('--max_rad', type=int, default=200, help='HoughCircles maxRadius')
parser.add_argument('--zero-angle', type=float, default=0.09, help='Angle of the zero value')
parser.add_argument('--max-angle', type=float, default=40.72, help='Angle of the max value')
parser.add_argument('--min-value', type=float, default=0.0, help='Min value of the gauge')
parser.add_argument('--max-value', type=float, default=80.0, help='Max value of the gauge')
parser.add_argument('--device', type=str, default='cpu', help='Device to run inference on, e.g., "cpu" or "mps"')
parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
args = parser.parse_args()

# Load the YOLOv8 model for needle detection
model = YOLO('./Pressure Gauges.v2-gauge-version-2.yolov8/runs/detect/pressure_gauge_detection/needle_v1_m3/weights/best.pt')
model.to(args.device)

# Open the camera
cap = cv2.VideoCapture(0)

# Create a window
cv2.namedWindow("YOLOv8 Inference")

frame_count = 0
gauge_center = None
gauge_radius = None

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if ret:


        # --- Gauge Center Detection (every 3 frames) ---
        if frame_count % 30 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                       param1=args.p1, param2=args.p2,
                                       minRadius=args.min_rad, maxRadius=args.max_rad)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                c = circles[0, 0]
                gauge_center = (int(c[0]), int(c[1]))
                gauge_radius = int(c[2])

        frame_count += 1
        
        # --- Needle Detection ---
        results = model(frame, conf=args.conf, imgsz=args.imgsz)
        annotated_frame = results[0].plot()
        
        # --- Measurement Extraction ---
        if gauge_center:
            for result in results:
                for box in result.boxes:
                    # Assuming the detected object is the needle
                    x1, y1, x2, y2 = box.xyxy[0]
                    
                    # Find the tip of the needle
                    corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
                    tip = max(corners, key=lambda p: math.hypot(p[0] - gauge_center[0], p[1] - gauge_center[1]))
                    
                    # Calculate needle angle
                    needle_angle = math.degrees(math.atan2(tip[1] - gauge_center[1], tip[0] - gauge_center[0]))
                    
                    # Map angle to value
                    value = map_angle_to_value(needle_angle, args.zero_angle, args.max_angle, args.min_value, args.max_value)
                    
                    if value is not None:
                        cv2.putText(annotated_frame, f"Value: {value:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        if gauge_center:
             cv2.circle(annotated_frame, gauge_center, gauge_radius, (0, 255, 0), 2)
             cv2.circle(annotated_frame, gauge_center, 2, (0, 0, 255), 3)

        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
