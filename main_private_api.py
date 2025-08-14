from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
from skimage.filters import threshold_otsu
from fastapi.middleware.cors import CORSMiddleware

import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_model_aug.keras")

model = load_model(MODEL_PATH)

# Load model
# model = load_model('mnist_model_aug.keras')

app = FastAPI(title="MNIST Digit Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_pil_image(pil_img):
    """Core preprocessing function for PIL images"""
    try:
        if pil_img.mode != 'L':
            pil_img = pil_img.convert('L')
        img_array = np.array(pil_img)
        
        # Apply binary thresholding
        thresh = threshold_otsu(img_array)
        img_array = (img_array > thresh).astype(np.uint8) * 255
        
        # Invert if white background
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        
        coords = np.argwhere(img_array > 0)
        if coords.size == 0:
            return None, "No digit detected"
        
        # Add padding to avoid cutting details
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        padding = 3
        y0, x0 = max(0, y0 - padding), max(0, x0 - padding)
        y1, x1 = min(img_array.shape[0], y1 + padding), min(img_array.shape[1], x1 + padding)
        cropped = img_array[y0:y1, x0:x1]
        
        h, w = cropped.shape
        scale = min(24 / h, 24 / w)
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        cropped_img = Image.fromarray(cropped).resize((new_w, new_h), Image.LANCZOS)
        cropped = np.array(cropped_img)
        
        # Center in 28x28 canvas
        new_img = np.zeros((28, 28), dtype=np.uint8)
        top = (28 - new_h) // 2
        left = (28 - new_w) // 2
        new_img[top:top+new_h, left:left+new_w] = cropped
        img_array = new_img
        
        img_array = img_array.astype(np.float32) / 255.0
        return img_array.reshape(1, 28, 28, 1), None
    except Exception as e:
        return None, str(e)

@app.post("/predict", response_model=dict)
async def predict_api(file: UploadFile = File(...)):
    """API endpoint for prediction"""
    contents = await file.read()
    try:
        pil_image = Image.open(BytesIO(contents))
        img_array, error = preprocess_pil_image(pil_image)
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        preds = model.predict(img_array)
        return {
            "prediction": int(np.argmax(preds)),
            "confidence": float(np.max(preds)),
            "probabilities": preds[0].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)