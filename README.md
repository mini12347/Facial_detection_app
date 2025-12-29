# Face Analysis App

## 📌 Project Overview

The **Face Analysis App** is a real-time computer vision application that detects **faces, emotions, age, and gender** from video streams. It combines **deep learning**, **classical machine learning experimentation**, and **pretrained transformer models** to deliver a unified and scalable face analysis system.

This project was developed and presented by **Mastouri Minyar**.

---

## 🎯 Motivation

Face analysis plays a crucial role in many domains such as **security**, **retail**, **human–computer interaction**, and **entertainment**. Most existing systems focus on a single attribute (emotion, age, or gender), while this project aims to:

* Provide **fast and unified face analysis**
* Combine **emotion, age, and gender detection** into a single real-time application
* Explore and compare **classical ML vs deep learning** approaches

---

## 🧠 Project Pipeline

### 1️⃣ Exploratory Data Analysis (EDA)

* Analyzed dataset distribution and balance
* Inspected image quality and class imbalance
* Identified dataset complexity, especially for subtle emotions

---

### 2️⃣ Classical Machine Learning Approach

#### 🔹 Feature Extraction (VGG16)

* Used **VGG16 pretrained on ImageNet** as a deep feature extractor
* Extracted high-level facial features (edges, shapes, structures)

#### 🔹 Dimensionality Reduction (PCA)

* Reduced feature dimensionality
* Kept components with highest variance
* Improved training speed and stability

#### 🔹 Models Tested

**K-Nearest Neighbors (KNN)**

* Initial accuracy: **42%**
* After Grid Search: **44%**
* Best parameters:

  * Metric: Euclidean
  * Neighbors: 12
  * Weights: Distance

**XGBoost**

* Baseline accuracy: **47%**
* Best-performing classes: 0 and 5
* Optuna hyperparameter optimization did not significantly improve results

📉 **Conclusion:** Classical ML struggled with subtle emotional variations despite strong feature extraction.

---

### 3️⃣ Deep Learning Approach (CNN)

Given the limitations of traditional ML, a **custom Convolutional Neural Network (CNN)** was designed.

#### 🔹 Why CNN?

* Learns directly from pixel-level spatial patterns
* Better captures complex emotional expressions

#### 🔹 Model Architecture

* **Input:** 48×48 grayscale images
* **Convolutional Blocks:**

  * Conv2D + Batch Normalization + ReLU
  * MaxPooling after each block
  * Dropout for regularization
* **Fully Connected Layers:**

  * Dense(512) → Dense(256) → Softmax (7 emotion classes)

#### 🔹 Training Strategy

* Data augmentation using `ImageDataGenerator`
* Callbacks:

  * ModelCheckpoint (best validation accuracy)
  * EarlyStopping
  * ReduceLROnPlateau

---

## 📊 Results

### ✅ Overall CNN Performance

* **Test Accuracy:** 81.95%
* **Test Loss:** 0.4771

### 📈 Per-Class Emotion Accuracy

| Emotion  | Accuracy |
| -------- | -------- |
| Ahegao   | 97.93%   |
| Happy    | 92.53%   |
| Angry    | 87.55%   |
| Surprise | 85.48%   |
| Neutral  | 66.39%   |
| Sad      | 61.83%   |

🚀 **Deep learning significantly outperformed classical ML approaches.**

---

## 🧩 Final Application Integration

The final application combines multiple technologies:

* 🧠 **Custom CNN model** for emotion recognition
* 🤗 **Hugging Face pretrained models** for age and gender estimation
* 👁️ **OpenCV Haar Cascade** for real-time face detection

### Features:

* Real-time face detection
* Emotion classification
* Age prediction
* Gender classification

---

## 🏗️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* Hugging Face Transformers
* NumPy, Pandas, Scikit-learn
* XGBoost

---

## 🔮 Future Improvements

* Face identification and recognition
* More fine-grained emotion categories
* Mobile deployment
* Performance optimization for edge devices

---

## 🏁 Conclusion

This project evolved from **classical machine learning pipelines** (EDA, VGG16, PCA, KNN, XGBoost) to a **fully optimized deep learning solution**. By integrating CNNs with transformer-based models, the system achieves **robust, real-time face analysis** and is ready for real-world applications.

---

📌 *If you plan to use or extend this project, feel free to contribute or reach out!*
