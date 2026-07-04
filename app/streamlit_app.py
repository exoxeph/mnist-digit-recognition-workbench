import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.common import config, preprocess, predict
from src.v2_cnn import load_trained_model


st.set_page_config(page_title="Digit Recognition Workbench", layout="centered")
st.title("Handwritten Digit Recognition Workbench")


@st.cache_resource
def get_model():
    return load_trained_model()


def canvas_to_28x28(canvas_image_rgba):
    image = Image.fromarray(canvas_image_rgba.astype("uint8"), mode="RGBA").convert("L")
    image = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    return np.array(image, dtype=np.float32)


def main():
    try:
        model = get_model()
    except FileNotFoundError:
        st.error(
            "No trained model found. Run `python main.py` (or "
            "`python -m src.v2_cnn`) to train the CNN before using this app."
        )
        return

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is None:
        st.info("Draw a digit above.")
        return

    if st.button("Predict"):
        image_28x28 = canvas_to_28x28(canvas_result.image_data)

        if image_28x28.max() == 0:
            st.warning("Canvas is empty - draw a digit first.")
            return

        result = predict.predict_with_confidence(
            model, image_28x28, preprocess.preprocess_single_image
        )

        st.subheader(f"Predicted digit: {result['predicted_digit']}")
        st.write(f"Confidence: {result['confidence']:.2%} ({result['confidence_band']})")

        if result["confidence_band"] == "low":
            st.warning("Low confidence - prediction may be unreliable.")

        st.write("Top 3 predictions:")
        for digit, prob in result["top3"]:
            st.write(f"- Digit {digit}: {prob:.2%}")

        st.write("What the model actually sees (28x28):")
        st.image(image_28x28.astype("uint8"), width=140, clamp=True)


if __name__ == "__main__":
    main()
