# 🧠 Brain Tumor Segmentation using U-Net

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview

This repository implements a **semantic segmentation model** designed to identify and mask brain tumors from MRI scans. Built from scratch using **TensorFlow** and **Keras**, the solution utilizes the **U-Net architecture** to achieve precise pixel-level classification.

The project emphasizes engineering reliability, featuring a scalable data pipeline, custom evaluation callbacks, and interrupt-safe model saving mechanisms.

---

## 🛠️ Tech Stack

* **Core Framework:** TensorFlow, Keras
* **Image Processing:** OpenCV (`cv2`), NumPy
* **Evaluation:** Scikit-learn
* **Visualization:** Matplotlib
* **Data Handling:** Pandas (CSV logging)

---

## 🏗️ Model Architecture

The model utilizes a custom implementation of **U-Net**, a convolutional neural network developed for biomedical image segmentation.

* **Type:** Encoder-Decoder with skip connections.
* **Optimization:** The model is trained using **Dice Loss** to handle class imbalance (tumor vs. background) and optimized for the **Dice Coefficient**.
* **Training:** Achieved **77% Validation Accuracy** in just 20 epochs.

[Image of U-Net architecture for biomedical image segmentation]

---

## 🚀 Key Features

### 1. Robust Data Pipeline
We implemented a scalable pipeline using `tf.data`, OpenCV, and NumPy that handles:
* Real-time image preprocessing and resizing.
* Efficient batching for memory management.
* Complex mask handling for ground truth data.

### 2. Engineering Reliability
* **Interrupt-Safe Saving:** Checkpoints are managed to ensure model progress is saved even if the training process is interrupted unexpectedly.
* **Custom Callbacks:** Automated monitoring of training metrics during execution.

### 3. Comprehensive Evaluation
A custom framework was designed to go beyond simple accuracy, reporting a full suite of segmentation metrics:
* **F1 Score (Dice Coefficient)**
* **Jaccard Index (IoU)**
* **Precision & Recall**
* **Automated Logging:** Results are logged to CSVs for experiment tracking.
* **Visual Validation:** Side-by-side comparison of Original MRI vs. Ground Truth vs. Model Prediction.

---

## 📊 Performance

| Metric | Details |
| :--- | :--- |
| **Epochs** | 20 |
| **Validation Accuracy** | **77%** |
| **Loss Function** | Dice Loss |
| **Optimizer** | Adam |

---

## 💻 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/brain-tumor-segmentation.git](https://github.com/yourusername/brain-tumor-segmentation.git)
