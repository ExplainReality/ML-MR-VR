# Segmentation Project
This script trains a YOLOv8 segmentation model (yolov8s-seg.pt) on a custom dataset defined in data.yaml. It’s set up for segmentation tasks and uses GPU acceleration (device=0) with an image size of 640 and 10 training epochs. The workers=0 setting is included to prevent issues with data loading on Windows. Once training finishes, it reports metrics like precision, recall, and mean Average Precision (mAP) for both bounding boxes and segmentation masks. Results, including the trained model and evaluation metrics are not uploaded, because of size managemenet.

Once we had a stable segmentation model, we switched to a lighter, nano version to better suit our hardware's performance limits.

To validate everything in action, we built a webcam_view tool—allowing us to test and visualize the model's real-time segmentation performance directly on-device. This gave us instant feedback and confirmed that the agents were accurately detecting and segmenting objects in the live feed.
