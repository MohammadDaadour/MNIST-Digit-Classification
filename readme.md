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

# MNIST Digit Classifier

This is a simple Gradio app for classifying handwritten digits using a trained model on the MNIST dataset.

## 🚀 How to Use
1. Draw a digit in the input box or upload an image.
2. Click **"Submit"** to get the prediction.
3. View the predicted digit and confidence score.

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

## 💡 Notes
- Change `sdk_version` to match your installed Gradio version:
```python
import gradio
print(gradio.__version__)
```
Replace `"5.42.0"` above with your version number.
