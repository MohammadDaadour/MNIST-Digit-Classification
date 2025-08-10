import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from PIL import Image

# Load model once
model = load_model('mnist_model.keras')

def preprocess_image(img):
    """
    input : image path or PIL Image
    
    Load an image, convert to grayscale, resize to 28x28,
    scale using the same scaling as training, and flatten in.
    
    output: Image Array
    """
    img = img.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = (img_array / 255.0) * 9
    img_array = img_array.reshape(1, -1)
    return img_array

def predict(img):
    processed_img = preprocess_image(img)
    pred_probs = model.predict(processed_img)
    predicted_digit = np.argmax(pred_probs)
    return f"Predicted Digit: {predicted_digit}"

# Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="MNIST Digit Classifier",
    description="Upload a handwritten digit image, and the model will predict the digit."
)

if __name__ == "__main__":
    interface.launch()
