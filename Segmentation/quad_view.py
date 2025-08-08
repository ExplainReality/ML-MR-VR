from flask import Flask, request, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile

app = Flask(__name__)
model = YOLO('E:/ML-MR-VR/runs/segment/train8/weights/best.pt')
print(">>> Starting segmentation Flask app")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model.predict(frame, imgsz=640, conf=0.3)
    annotated = results[0].plot()

    # Save to a temporary file and return
    _, buffer = cv2.imencode('.jpg', annotated)
    return send_file(
        tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", mode="w+b").write(buffer),
        mimetype='image/jpeg'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
