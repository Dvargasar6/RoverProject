# !/usr/bin/env 
# python3 detect_objects.py <image_path>

from ultralytics import YOLO
import cv2
import sys
import os

def main():
    # Verify args:
    if len(sys.argv) < 2:
        print("Two args are expected:")
        print("Run: python3 detect_objects.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Verify the image
    if not os.path.exists(image_path):
        print(f"Error: The image was not found '{image_path}'")
        sys.exit(1)
    
    # Load and run model:
    print("Loading model...")
    model = YOLO('yolov8n.pt')
    print(f"Processing image: {image_path}...")
    results = model(image_path)
    
    # Save result image:
    output_path = "./OutputImages/output_" + os.path.basename(image_path)
    results[0].save(output_path)
    print(f"Output saved in: {output_path}")
    
    # Show statistics:
    boxes = results[0].boxes
    if boxes is not None:
        print(f"Detected objects: {len(boxes)}")
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"  {i+1}. {model.names[cls]}: {conf:.2f}")

    # Display the output image:
    img_with_boxes = results[0].plot()
    cv2.imshow('YOLOv8 Detection', img_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()