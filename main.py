import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from PIL import Image
import traceback

print(gr.__version__)

#load
model = load_model('mnist_model_aug.keras')
#



def preprocess_image(img):
    if img is None:
        return None, "No image received"

    try:
        # If sketchpad dict format
        if isinstance(img, dict) and "composite" in img:
            img = img["composite"]

        # If NumPy array, convert to PIL
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img).astype("uint8"))

        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')

        # Convert to numpy array
        img_array = np.array(img)

        # Invert if background is white
        if np.mean(img_array) > 127:
            img_array = 255 - img_array

        # -------------------------
        # Centerize and resize digit
        coords = np.argwhere(img_array > 0)
        if coords.size > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            cropped = img_array[y0:y1, x0:x1]

            # Resize cropped digit to fit 28x28 while keeping aspect ratio
            h, w = cropped.shape
            scale = min(28 / h, 28 / w)
            new_h = max(1, int(h * scale))
            new_w = max(1, int(w * scale))
            cropped_img = Image.fromarray(cropped).resize((new_w, new_h), Image.LANCZOS)
            cropped = np.array(cropped_img)

            # Create blank 28x28 image
            new_img = np.zeros((28, 28), dtype=np.uint8)
            top = (28 - new_h) // 2
            left = (28 - new_w) // 2
            new_img[top:top+new_h, left:left+new_w] = cropped
            img_array = new_img
        # -------------------------

        # Normalize
        img_array = img_array.astype(np.float32) / 255.0

        # Reshape for model input
        img_array = img_array.reshape(1, 28, 28, 1)

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
ف
