from ultralytics import YOLO
import cv2
import os

# Define paths
model_path = 'runs/detect/pressure_gauge_detection/needle_v1_m3/weights/best.pt'
image_path = './../60.png'
output_dir = '../../test_results'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'result_60.jpg')

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
else:
    # Load the trained model
    model = YOLO(model_path)

    # Run inference on the image
    results = model(image_path)

    # results is a list of Results objects
    # We can get the annotated image from the first result
    if results:
        annotated_image = results[0].plot()

        # Save the annotated image
        cv2.imwrite(output_path, annotated_image)

        print(f"Test complete. Annotated image saved to {output_path}")
    else:
        print("No results were returned from the model.")
