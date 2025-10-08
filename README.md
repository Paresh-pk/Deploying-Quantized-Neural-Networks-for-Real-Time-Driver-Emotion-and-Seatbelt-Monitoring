Real-Time Driver State Monitoring using Quantized Neural Networks
This project implements a lightweight, real-time system to monitor a driver's state by analyzing their facial emotion and detecting the presence of a seatbelt. The system is built using deep learning and optimized for high performance on CPU-based edge devices through model quantization.

📋 Table of Contents
Project Overview

Features

Performance Metrics

System Architecture

Installation

Usage

Dataset

Notes & Observations

🔭 Project Overview
The primary goal of this project is to build an efficient Driver Monitoring System (DMS) suitable for deployment in real-world automotive environments. The system uses a Convolutional Neural Network (CNN) to classify the driver's emotion into one of seven categories. A key aspect of this project is the optimization pipeline, where the trained model is converted to TensorFlow Lite (TFLite) and quantized to INT8. This process significantly reduces the model's size and inference latency, making it ideal for edge computing.

As a bonus feature, a second, smaller binary CNN is trained to detect the presence of a seatbelt based on a simple heuristic, adding an extra layer of safety monitoring.

✨ Features
Emotion Recognition: Classifies driver's facial emotion into 7 categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

Seatbelt Detection: A conceptual model to identify if a seatbelt is being worn.

High Performance: The emotion model is quantized to INT8, ensuring low latency and a small footprint for real-time CPU inference.

Lightweight Face Detection: Uses a classic Haar Cascade classifier from OpenCV for fast and efficient face localization.

End-to-End Pipeline: The notebook covers the entire workflow from data preprocessing and model training to evaluation and deployment-ready optimization.

📊 Performance Metrics
The following metrics were reported for the primary Emotion Recognition model.

Model Performance
The model was trained for 30 epochs on a subset of the FER2013 dataset.

Metric

Value

Top-1 Test Accuracy

18.00%

Confusion Matrix
The confusion matrix shows the model's performance across the 7 emotion classes. The low accuracy suggests the model is significantly overfitting, likely due to the very small dataset size used in this run.

Note: You can screenshot the confusion matrix from your notebook and embed it here.

<!-- Example: <img src="images/confusion_matrix.png" width="400"> -->

Deployment Metrics
These metrics demonstrate the effectiveness of INT8 quantization for edge deployment.

Metric

Value

Notes

Original Keras Model Size

8.48 MB

Standard HDF5 format (.h5).

Quantized TFLite Model Size

0.72 MB

INT8 quantized (.tflite). A 91.5% reduction in size!

CPU Latency per Frame

~4.71 ms

Measured with the TFLite Interpreter on a dummy input.

🏗️ System Architecture
The monitoring pipeline consists of two main stages:

Face & ROI Detection:

An input video frame is processed by a Haar Cascade classifier to find the driver's face.

A predefined Region of Interest (ROI) for the chest area is extracted for seatbelt analysis.

State Analysis:

The cropped face is resized to 48x48 grayscale and fed into the Quantized Emotion CNN (emotion_classifier_int8.tflite).

The chest ROI is fed into the Binary Seatbelt CNN to determine if a seatbelt is present.

Note: You can create and embed a simple architecture diagram here.

<!-- Example: <img src="images/architecture.png" width="600"> -->

💾 Installation
To run this project, clone the repository and install the required dependencies.

# Clone this repository
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name

# Install Python packages
pip install -r requirements.txt

Create a requirements.txt file with the following content:

tensorflow
pandas
scikit-learn
seaborn
matplotlib
opencv-python
onnx
onnxruntime
tf2onnx

🚀 Usage
Download Dataset: Obtain the fer2013.csv dataset (e.g., from Kaggle) and place it in the root directory of the project.

Run the Notebook: Open and run the .ipynb notebook in a Jupyter or Google Colab environment. The cells are ordered to execute the complete pipeline from training to model conversion.

💿 Dataset
This project uses the FER2013 (Facial Expression Recognition 2013) dataset for training the emotion classifier. It consists of 48x48 pixel grayscale images of faces.

For the seatbelt detection bonus, a dummy dataset was synthetically generated to demonstrate the concept, as a real-world dataset was not available for this project.

📝 Notes & Observations
Dependency Conflict: During the initial setup (!pip install), a dependency conflict was noted: tensorflow 2.19.0 requires flatbuffers>=24.3.25, but a different version was installed by onnxruntime. The notebook ran successfully, but this could be a point of failure in different environments and may require pinning specific package versions.

Overfitting and Low Accuracy: The final test accuracy of 18.00% is quite low. The training history shows the training accuracy increasing rapidly while validation accuracy stagnates, which is a classic sign of overfitting. This is primarily due to the very small dataset size used in the notebook (Training set shape: (358, 48, 48, 1)). To improve this, the model should be trained on the full FER2013 dataset with more robust data augmentation and potentially a longer training schedule with early stopping.

Data Cleaning: The code includes a necessary check to handle corrupted rows in fer2013.csv, ensuring the data loader is robust.

Quantization Success: The INT8 quantization was highly effective, reducing the model size from 8.48 MB to 0.72 MB while enabling fast CPU inference. This successfully demonstrates the core objective of preparing a model for an edge device.
