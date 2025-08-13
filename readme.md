# 🖋️ MNIST Digit Classifier

A machine learning project to classify handwritten digits (0–9) using the MNIST dataset, with deployment on a custom-built website via a private API.

---

## 📌 Project Overview
This project started as a basic MNIST classifier and evolved into a robust, production-ready system through several iterations and problem-solving steps.  
We faced challenges during deployment and image preprocessing, which we overcame by implementing advanced preprocessing techniques and infrastructure changes.

---

## ⚙️ Workflow
1. **Data Preprocessing**
   - Normalized images (pixel values between 0 and 1).
   - Reshaped images to (28×28) format.
   - Split data into training and test sets.

2. **Model Training**
   - Trained using TensorFlow/Keras on MNIST dataset.
   - Achieved **98% accuracy** on test set.

3. **Problem 1: Deployment Performance Drop**
   - Model accuracy dropped when deployed due to differences between input images and training data.

4. **Solution 1: Image Generator Algorithm**
   - Implemented data augmentation to make the model more robust.
   - Improved results but still weak for digits close to image edges.

5. **Problem 2: Edge Digits**
   - Digits written near the borders caused classification errors.

6. **Solution 2: Cropping & Centering**
   - Cropped digits from boundaries.
   - Centered them within a 28×28 frame.
   - Significantly improved accuracy.

7. **Problem 3: Deployment Crashes**
   - Gradio + Hugging Face Spaces crashed due to resource/time limits.

8. **Solution 3: Private API + Custom Website**
   - Built a private API to handle inference requests.
   - Developed a website from scratch for stable deployment.

---

## 🛠️ Tech Stack
- **Programming Language:** Python
- **Libraries:** TensorFlow/Keras, NumPy, OpenCV, Gradio
- **Deployment:** Hugging Face Spaces, Private API, Custom HTML/CSS/JS Website

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/mnist-digit-classifier.git
cd mnist-digit-classifier




---
title: MNIST Digit Classifier
emoji: 🔢
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "5.42.0"
app_file: main.py
pinned: false
---


## 📂 Project Structure
- `main.py` → Main Gradio application file.
- `preprocessing.ipynb` → Preprocessing steps and data handling.
- `mnist_model_aug.h5` → Trained model file in H5 format.
- `mnist_model_aug.keras` → Trained model in Keras format.
- `model.py` → Model loading and prediction logic.
- `requirements.txt` → Python dependencies.

## 🛠️ Requirements
Install dependencies locally with:
```bash
pip install -r requirements.txt
```


