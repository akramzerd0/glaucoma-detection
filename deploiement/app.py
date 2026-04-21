import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

MODEL_PATH = "eyepacs_airogs_vgg16_best_threshold.h5"
IMG_SIZE = (224, 224)
THRESHOLD = 0.45

# Expected mapping for folder names sorted alphabetically:
# NRG -> 0
# RG  -> 1
CLASS_NAMES = {0: "NRG", 1: "RG"}

model = tf.keras.models.load_model(MODEL_PATH)

def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image).astype(np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_glaucoma(image: Image.Image):
    if image is None:
        return "No image uploaded.", {"NRG": 0.0, "RG": 0.0}

    x = prepare_image(image)
    prob_rg = float(model.predict(x, verbose=0)[0][0])

    pred_label = 1 if prob_rg >= THRESHOLD else 0
    pred_name = CLASS_NAMES[pred_label]

    result_text = (
        f"Prediction: {pred_name}\n"
        f"Threshold used: {THRESHOLD:.2f}\n"
        f"RG probability: {prob_rg:.4f}\n\n"
        f"Meaning:\n"
        f"- NRG = Non-Referable Glaucoma\n"
        f"- RG = Referable Glaucoma"
    )

    probs = {
        "NRG": round(1 - prob_rg, 4),
        "RG": round(prob_rg, 4),
    }
    return result_text, probs

demo = gr.Interface(
    fn=predict_glaucoma,
    inputs=gr.Image(type="pil", label="Upload retinal image"),
    outputs=[
        gr.Textbox(label="Result"),
        gr.Label(label="Class probabilities")
    ],
    title="Glaucoma Detection with VGG16",
    description=(
        "Upload a retinal fundus image. "
        "The model predicts whether the image is NRG or RG.\n\n"
        "NRG = Non-Referable Glaucoma\n"
        "RG = Referable Glaucoma"
    ),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()