import cv2
from ultralytics import YOLO
import argparse
import numpy as np
import math

# --- Globals for Calibration ---
clicks = []
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
        clicks.append((x, y))
        print(f"Clicked at: ({x}, {y})")

def get_angle(p1, p2):
    return math.degrees(math.atan2(int(p1[1]) - int(p2[1]), int(p1[0]) - int(p2[0])))

def map_angle_to_value(angle, zero_angle, max_angle, min_value, max_value):
    # Convert all angles to a 0-360 range for consistent comparison
    angle = (angle + 360) % 360
    zero_angle = (zero_angle + 360) % 360
    max_angle = (max_angle + 360) % 360

    # Calculate the total sweep angle of the gauge
    sweep_angle = (max_angle - zero_angle + 360) % 360
    
    # If sweep_angle is 0, it means zero and max angles are the same point, which is an invalid range
    if sweep_angle == 0:
        return None # Or raise an error
    
    # Calculate the needle's angle relative to the zero_angle
    relative_angle = (angle - zero_angle + 360) % 360

    # Check if the needle's relative angle falls within the sweep
    if relative_angle > sweep_angle:
        return None # Needle is outside the defined sweep (in the "gap")

    # Perform linear interpolation based on the percentage of the sweep angle
    percentage = relative_angle / sweep_angle
    value = min_value + percentage * (max_value - min_value)
    return value

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv8 Live Inference with Fixed Gauge Position')
    # --- General arguments ---
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for needle detection')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run inference on, e.g., "cpu" or "mps"')
    parser.add_argument('--imgsz', type=int, default=320, help='Inference image size')
    
    # --- Fixed Circle arguments ---
    parser.add_argument('--circle-x', type=int, default=320, help='X-coordinate of the fixed circle center')
    parser.add_argument('--circle-y', type=int, default=240, help='Y-coordinate of the fixed circle center')
    parser.add_argument('--circle-r', type=int, default=200, help='Radius of the fixed circle')

    # --- Calibration arguments ---
    parser.add_argument('--calibrate', action='store_true', help='Run in calibration mode to get angles')
    parser.add_argument('--zero-angle', type=float, help='Angle of the zero value (from calibration)')
    parser.add_argument('--max-angle', type=float, help='Angle of the max value (from calibration)')
    parser.add_argument('--min-value', type=float, default=0.0, help='Min value of the gauge')
    parser.add_argument('--max-value', type=float, default=80.0, help='Max value of the gauge')
    args = parser.parse_args()

    # --- Calibration Mode ---
    if args.calibrate:
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse_callback)
        gauge_center = (args.circle_x, args.circle_y)

        while len(clicks) < 2:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (640, 480))
            cv2.circle(frame, gauge_center, args.circle_r, (0, 255, 0), 2)
            
            if len(clicks) == 0:
                cv2.putText(frame, "Click on the ZERO position", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif len(clicks) == 1:
                cv2.putText(frame, "Click on the MAX position", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            for point in clicks:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)

            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

        if len(clicks) == 2:
            zero_angle = get_angle(clicks[0], gauge_center)
            max_angle = get_angle(clicks[1], gauge_center)
            print("\n--- Calibration Complete ---")
            print("Add these arguments to your command:")
            print(f"--zero-angle {zero_angle:.2f} --max-angle {max_angle:.2f}")
        else:
            print("\nCalibration was not completed.")
        
        exit()

    # --- Inference Mode ---
    if args.zero_angle is None or args.max_angle is None:
        print("Error: --zero-angle and --max-angle must be provided for inference.")
        print("Run with the --calibrate flag first to get these values.")
        exit()

    model = YOLO('./Pressure Gauges.v2-gauge-version-2.yolov8/runs/detect/pressure_gauge_detection/needle_v1_m3/weights/best.pt')
    model.to(args.device)
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("YOLOv8 Inference")
    gauge_center = (args.circle_x, args.circle_y)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 480))
        cv2.circle(frame, gauge_center, args.circle_r, (0, 255, 0), 2)

        results = model(frame, conf=args.conf, imgsz=args.imgsz)
        annotated_frame = results[0].plot()
        
        if gauge_center:
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
                    tip = max(corners, key=lambda p: math.hypot(p[0] - gauge_center[0], p[1] - gauge_center[1]))
                    needle_angle = get_angle(tip, gauge_center)
                    value = map_angle_to_value(needle_angle, args.zero_angle, args.max_angle, args.min_value, args.max_value)
                    
                    # --- DEBUG PRINTS ---
                    print(f"Needle Angle: {needle_angle:.2f}, Mapped Value: {value}, Valid Range: [{args.zero_angle:.2f}, {args.max_angle:.2f}]")
                    # --- END DEBUG ---

                    if value is not None:
                        cv2.putText(annotated_frame, f"Value: {value:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
