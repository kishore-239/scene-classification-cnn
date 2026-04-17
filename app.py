import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH  = "final_model_efficientnetb0.h5"
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

CLASS_INFO = {
    "buildings": "Urban structures — offices, houses, and city skylines.",
    "forest":    "Dense woodland areas with trees and natural vegetation.",
    "glacier":   "Large ice formations typically found in polar or alpine regions.",
    "mountain":  "Elevated rocky terrain with peaks and slopes.",
    "sea":        "Open ocean or large bodies of water.",
    "street":    "Roads, paths, and urban ground-level scenes.",
}

# ── Load Model ────────────────────────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict(image: Image.Image):
    if image is None:
        return {cls: 0.0 for cls in CLASS_NAMES}, "Please upload an image."

    x     = preprocess_image(image)
    probs = model.predict(x, verbose=0)[0]

    top_idx   = int(np.argmax(probs))
    top_class = CLASS_NAMES[top_idx]
    top_conf  = float(probs[top_idx]) * 100

    label_dict = {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}

    info = (
        f"**Predicted Scene: {top_class.upper()}**\n\n"
        f"Confidence: {top_conf:.1f}%\n\n"
        f"{CLASS_INFO[top_class]}"
    )

    return label_dict, info


# ── Gradio Interface ──────────────────────────────────────────────────────────
with gr.Blocks(title="Scene Classification", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Scene Classification
        ### Built with EfficientNetB0 — Transfer Learning (Intel Image Dataset)
        Upload any outdoor or urban scene photo and the model will classify it
        into one of **6 categories**: buildings, forest, glacier, mountain, sea, street.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Scene Image")
            predict_btn = gr.Button("Classify", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=6, label="Class Probabilities")
            info_output  = gr.Markdown(label="Result")

    predict_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[label_output, info_output]
    )

    gr.Examples(
        examples=[],   # add local example image paths here if available
        inputs=image_input
    )

    gr.Markdown(
        """
        ---
        **Model:** EfficientNetB0 fine-tuned on Intel Image Classification dataset
        **Classes:** buildings · forest · glacier · mountain · sea · street
        **Framework:** TensorFlow / Keras · Gradio
        """
    )


if __name__ == "__main__":
    demo.launch(share=True)
