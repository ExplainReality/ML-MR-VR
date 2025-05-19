from ultralytics import YOLO
import cv2

# Load your trained YOLO model
model = YOLO('E:/ML-MR-VR/runs/detect/train4/weights/best.pt')

# Open webcam (0 = default laptop webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model.predict(frame, imgsz=640, conf=0.3)

    # Visualize results on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow('YOLO Detection', annotated_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
