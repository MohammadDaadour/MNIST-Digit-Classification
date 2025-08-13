import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from PIL import Image
from skimage.filters import threshold_otsu
import traceback

print(gr.__version__)

#load
model = load_model('mnist_model_aug.keras')
#

def preprocess_image(img):
    if img is None:
        return None, "No image received"
    try:
        if isinstance(img, dict) and "composite" in img:
            img = img["composite"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img).astype("uint8"))
        if img.mode != 'L':
            img = img.convert('L')
        img_array = np.array(img)
        
        # Apply binary thresholding
        thresh = threshold_otsu(img_array)
        img_array = (img_array > thresh).astype(np.uint8) * 255
        
        # Invert if white background
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        
        coords = np.argwhere(img_array > 0)
        if coords.size == 0:
            return None, "No digit detected"
        
        # Add padding to avoid cutting "9" details
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        padding = 3  # Increased padding for "9"
        y0, x0 = max(0, y0 - padding), max(0, x0 - padding)
        y1, x1 = min(img_array.shape[0], y1 + padding), min(img_array.shape[1], x1 + padding)
        cropped = img_array[y0:y1, x0:x1]
        
        h, w = cropped.shape
        scale = min(24 / h, 24 / w)  # Scale to 24x24 to preserve "9" details
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
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Save preprocessed image for debugging
        Image.fromarray((img_array.reshape(28, 28) * 255).astype(np.uint8)).save("preprocessed.png")
        return img_array, None
    except Exception as e:
        return None, str(e)

def predict(img):
    img_array, err = preprocess_image(img)
    if err:
        return err, None

    preds = model.predict(img_array)
    predicted_class = int(np.argmax(preds))
    confidence = float(np.max(preds))

    # Format confidence as percentage with two decimal places
    return f"{predicted_class} ({confidence*100:.2f}% confidence)"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(type="pil"),
    outputs="text",
    title="MNIST Digit Classifier",
    description="Draw a digit (0-9) and get the prediction.",
    allow_flagging="never"
)
interface.launch(share=True, debug=True)