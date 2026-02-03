from ultralytics import YOLO
import cv2
import math

# ----------------------------
# LOAD MODEL
# ----------------------------
model = YOLO("./runs/detect/train4/weights/best.pt")

# ----------------------------
# GAUGE & CROP SETTINGS
# ----------------------------
CX, CY = 506, 513  # gauge center (FULL FRAME)

CROP_X1, CROP_Y1 = 300, 300
CROP_X2, CROP_Y2 = 800, 800

CONF_THRESH = 0.2

# ----------------------------
# VALUE CALIBRATION
# ----------------------------
THETA_MIN = 225.0   # angle at minimum value (normalized 0–360)
THETA_MAX = 45.0    # angle at maximum value (normalized 0–360)

VALUE_MIN = 0
VALUE_MAX = 100

# ----------------------------
# CAMERA
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not opening")

# ----------------------------
# MAIN LOOP
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop around gauge
    crop = frame[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]

    # YOLO inference
    results = model(crop, conf=CONF_THRESH, verbose=False)

    for r in results:
        for box in r.boxes:
            # Box coords (crop-relative)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Translate back to full-frame
            x1 += CROP_X1
            x2 += CROP_X1
            y1 += CROP_Y1
            y2 += CROP_Y1

            # Find needle tip
            corners = [
                (x1, y1),
                (x2, y1),
                (x1, y2),
                (x2, y2)
            ]

            tip = max(
                corners,
                key=lambda p: math.hypot(p[0] - CX, p[1] - CY)
            )

            # Compute angle
            angle = math.degrees(math.atan2(tip[1] - CY, tip[0] - CX))

            # Normalize angle to 0–360
            if angle < 0:
                angle += 360

            # Angle → value
            value = (angle - THETA_MIN) / (THETA_MAX - THETA_MIN)
            value = 1.0 - value  # invert direction if needed
            value = value * (VALUE_MAX - VALUE_MIN) + VALUE_MIN
            value = max(VALUE_MIN, min(VALUE_MAX, value))

            # Draw visuals
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, tip, 6, (255, 0, 0), -1)

            cv2.putText(
                frame,
                f"Value: {value:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    # Draw gauge center
    cv2.circle(frame, (CX, CY), 5, (0, 0, 255), -1)

    cv2.imshow("Gauge Reader", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ----------------------------
# CLEANUP
# ----------------------------
cap.release()
cv2.destroyAllWindows()
