# Deploying-Quantized-Neural-Networks-for-Real-Time-Driver-Emotion-and-Seatbelt-Monitoring

This project implements a lightweight, real-time system to monitor a driver's state by analyzing their facial emotion and detecting the presence of a seatbelt. The system is built using deep learning and optimized for high performance on CPU-based edge devices through model quantization.

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Features](#-features)
- [Performance Metrics](#-performance-metrics)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Notes & Observations](#-notes--observations)

## 🔭 Project Overview
The primary goal of this project is to build an efficient Driver Monitoring System (DMS) suitable for deployment in real-world automotive environments. The system uses a Convolutional Neural Network (CNN) to classify the driver's emotion into one of seven categories. A key aspect of this project is the optimization pipeline, where the trained model is converted to TensorFlow Lite (TFLite) and quantized to INT8. This process significantly reduces the model's size and inference latency, making it ideal for edge computing.

As a bonus feature, a second, smaller binary CNN is trained to detect the presence of a seatbelt based on a simple heuristic, adding an extra layer of safety monitoring.

## ✨ Features
- **Emotion Recognition:** Classifies driver's facial emotion into 7 categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
- **Seatbelt Detection:** A conceptual model to identify if a seatbelt is being worn.
- **High Performance:** The emotion model is quantized to INT8, ensuring low latency and a small footprint for real-time CPU inference.
- **Lightweight Face Detection:** Uses a classic Haar Cascade classifier from OpenCV for fast and efficient face localization.
- **End-to-End Pipeline:** The notebook covers the entire workflow from data preprocessing and model training to evaluation and deployment-ready optimization.

## 📊 Performance Metrics

### Model Performance
The model was trained for 30 epochs on a subset of the FER2013 dataset.

| Metric | Value |
|--------|--------|
| **Top-1 Test Accuracy** | **18.00%** |

The low accuracy suggests the model is significantly overfitting due to the small dataset used in this run.

### Deployment Metrics

| Metric | Value | Notes |
|--------|--------|-------|
| **Original Keras Model Size** | 8.48 MB | Standard HDF5 format (.h5) |
| **Quantized TFLite Model Size** | 0.72 MB | INT8 quantized (.tflite). A 91.5% reduction in size! |
| **CPU Latency per Frame** | ~4.71 ms | Measured with the TFLite Interpreter |

## 🏗️ System Architecture

### Face & ROI Detection
- Haar Cascade classifier detects the face.
- A predefined Region of Interest (ROI) is extracted for seatbelt analysis.

### State Analysis
- Cropped face → 48x48 grayscale → fed into **Quantized Emotion CNN** (`emotion_classifier_int8.tflite`).
- Chest ROI → fed into **Binary Seatbelt CNN** for seatbelt detection.

## 💾 Installation

```bash
# Clone this repository
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
tensorflow
pandas
scikit-learn
seaborn
matplotlib
opencv-python
onnx
onnxruntime
tf2onnx
```

## 🚀 Usage

1. **Download Dataset:** Obtain the `fer2013.csv` dataset (e.g., from Kaggle - https://www.kaggle.com/datasets/ahmedmoorsy/facial-expression ) and place it in the project root.
2. **Run Notebook:** Open the `.ipynb` notebook and execute all cells sequentially.

## 💿 Dataset

The project uses the **FER2013** dataset for emotion recognition and a synthetic dataset for seatbelt detection.

## 📝 Notes & Observations
- **Dependency Conflict:** TensorFlow 2.19.0 requires `flatbuffers>=24.3.25`, but ONNX installs a different version. May need version pinning.
- **Overfitting:** Final accuracy is low (18%), indicating overfitting due to limited training data.
- **Data Cleaning:** Code handles corrupted rows in `fer2013.csv`.
- **Quantization Success:** Reduced model size from **8.48 MB → 0.72 MB**, achieving fast edge inference.
