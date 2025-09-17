# Machine Learning Visual Recognition (MLVR)

This repository contains the complete workflow and source code for our machine learning and computer vision experiments.  
Although the project started with three separate branches for different stages of development, **all content is also available on the `main` branch** for convenience.

---

## 📂 Project Structure

### 1. **Data Collection**
When the project began, no suitable dataset was available.  
To overcome this limitation, we developed and published a **custom web crawler** that:
- Automatically collects and downloads relevant images.
- Stores images locally to avoid API request limitations and rate caps.
- Ensures reproducibility by keeping the dataset stable over time.

This module is essential for building a diverse, high-quality dataset that fuels the later training phases.

---

### 2. **Object Detection**
We implemented and trained multiple YOLO versions for object detection, including:
- **YOLOv8s** (small)
- **YOLOv8n** (nano)
- **YOLOv9**

Key features:
- Trained on **both local hardware and server environments**.
- Capable of detecting **multiple objects simultaneously**.
- Generates **bounding boxes** around detected objects.
- Supports **real-time and batch detection**.

---

### 3. **Segmentation**
The segmentation stage focuses on **precise pixel-level object recognition**.  
We experimented with:
- **YOLOv8-seg**
- **YOLOv9-seg**

Highlights:
- Trained locally for **250+ epochs** to ensure high accuracy.
- Achieved **80%+ accuracy** on validation data.
- Produces **segmentation masks** that highlight the exact shape and boundaries of detected objects.
- Suitable for tasks requiring detailed object outlines rather than simple bounding boxes.

---

## 🚀 Technologies Used
- **YOLOv8 / YOLOv9** (detection and segmentation)
- **Python** (model training, inference scripts, utilities)
- **Custom web scraping** for dataset generation
- **Local & remote training environments**
- **GPU acceleration** (CUDA)

---

## Prerequisites
Before installing, make sure you have the following:
- **Python 3.10+**
- **CUDA 11.8 + cuDNN** (required for GPU acceleration)
- **TensorFlow-GPU 2.10.1** (with `tensorflow-estimator`, `tensorboard`)
- **PyTorch 2.7.0 + CUDA 11.8**, `torchvision`, `torchaudio`
- **ONNX Runtime**
- **Ultralytics (YOLOv8)**

---

## Backend Setup

### Clone the repository
```bash
git clone https://github.com/ExplainReality/ML-MR-VR.git
cd ML-MR-VR
```

### On Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### On Linux/macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

After these steps: you can run the pre-loaded models on your dataset as well. Or you can scrape your own dataset, using the scraper.

## 📌 Notes
- All code from the original three branches (`dataset_collection`, `object_detection`, `segmentation`) is preserved in the `main` branch.
- This repository aims to be both a **learning resource** and a **practical toolset** for future computer vision projects.
Link to ![documentation](https://docs.google.com/document/d/1U6_q_0YK6zWxx516DnNw1ccuuMlfXvjCLV3By1FwMIg/edit?tab=t.8ln26orc533u) created detailed by us. (for access don't hesitate to contact us)
Link to ![menu](https://github.com/nioowsha/Menu3D.git) for the headset, created in Unity.
---
