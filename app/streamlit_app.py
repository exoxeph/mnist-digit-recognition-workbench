import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.common import config, load_data, preprocess, predict
from src.v2_cnn import load_trained_model

GREEN = "#3ddc97"
AMBER = "#f5c344"
RED = "#ff5c5c"
MUTED = "#4a5568"
BAND_COLOR = {"high": GREEN, "medium": AMBER, "low": RED}

st.set_page_config(page_title="MNIST Digit Recognition Workbench", layout="centered")

st.markdown(
    """
    <style>
    .badge-row { display:flex; gap:0.5rem; flex-wrap:wrap; justify-content:flex-end; }
    .badge {
        background:#1c2333; border:1px solid #333c4d; border-radius:6px;
        padding:4px 10px; font-family:monospace; font-size:0.78rem; color:#9aa4b2;
        white-space:nowrap;
    }
    .verdict-box {
        border-left:6px solid #3ddc97; background:rgba(255,255,255,0.03);
        border-radius:8px; padding:1.1rem 1.4rem; margin:1rem 0 1.2rem 0;
    }
    .verdict-title { font-size:1.35rem; font-weight:700; font-family:monospace; }
    .verdict-sub { color:#9aa4b2; margin-top:0.3rem; font-family:monospace; font-size:0.9rem; }
    .prob-track {
        background:#1c2333; border-radius:5px; height:10px; overflow:hidden; width:100%;
    }
    .prob-fill { height:100%; border-radius:5px; }
    .top3-row { display:flex; align-items:center; gap:0.6rem; margin:0.3rem 0; font-family:monospace; }
    .top3-label { width:2.2rem; color:#f5f5f5; }
    .top3-pct { width:3.6rem; text-align:right; color:#9aa4b2; }
    </style>
    """,
    unsafe_allow_html=True,
)

header_left, header_right = st.columns([2, 1])
with header_left:
    st.markdown(
        '<span style="font-size:1.9rem; font-weight:700; font-family:monospace;">'
        "MNIST Digit Recognition Workbench</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span style="color:#9aa4b2;">Draw a digit or load a real test-set sample, '
        "then hit Predict to see the model's confidence-scored verdict.</span>",
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown(
        """
        <div class="badge-row">
            <span class="badge">model: CNN (V2)</span>
            <span class="badge">test acc: 98.45%</span>
            <span class="badge">dataset: MNIST</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")


@st.cache_resource
def get_model():
    return load_trained_model()


@st.cache_resource
def get_sample_digits():
    (_, _), (_, _), (x_test, y_test) = load_data.load_split()
    samples = {}
    for digit in range(config.NUM_CLASSES):
        idx = np.where(y_test == digit)[0][0]
        samples[digit] = x_test[idx]
    return samples


def canvas_to_28x28(canvas_image_rgba):
    image = Image.fromarray(canvas_image_rgba.astype("uint8"), mode="RGBA").convert("L")
    image = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    return np.array(image, dtype=np.float32)


def render_result(result, image_28x28, true_digit=None):
    predicted = result["predicted_digit"]
    confidence = result["confidence"]
    band = result["confidence_band"]

    if true_digit is not None:
        correct = predicted == true_digit
        color = GREEN if correct else RED
        title = f"{'CORRECT' if correct else 'INCORRECT'} - predicted {predicted}, true label {true_digit}"
    else:
        color = BAND_COLOR[band]
        title = f"Predicted digit: {predicted}"

    st.markdown(
        f"""
        <div class="verdict-box" style="border-left-color:{color};">
            <div class="verdict-title" style="color:{color};">{title}</div>
            <div class="verdict-sub">confidence {confidence:.1%} - {band} band</div>
            <div class="prob-track" style="margin-top:0.7rem;">
                <div class="prob-fill" style="width:{confidence * 100:.1f}%; background:{color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    cols[0].metric("Predicted digit", predicted)
    cols[1].metric("Confidence", f"{confidence:.1%}")
    cols[2].metric("Confidence band", band.upper())
    if true_digit is not None:
        cols[3].metric("True label", true_digit)
    else:
        gap = (result["top3"][0][1] - result["top3"][1][1]) * 100
        cols[3].metric("Top-1 vs Top-2 gap", f"{gap:.1f} pp")

    st.markdown("**Top-3 predictions**")
    for digit, prob in result["top3"]:
        bar_color = GREEN if digit == predicted else MUTED
        st.markdown(
            f"""
            <div class="top3-row">
                <div class="top3-label">#{digit}</div>
                <div class="prob-track" style="flex:1;">
                    <div class="prob-fill" style="width:{prob * 100:.1f}%; background:{bar_color};"></div>
                </div>
                <div class="top3-pct">{prob:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("**What the model actually sees (28x28):**")
    st.image(image_28x28.astype("uint8"), width=140, clamp=True)

    if band == "low":
        st.warning("Low confidence - this prediction may be unreliable.")


def draw_tab(model):
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

    if st.button("Predict", key="predict_draw"):
        image_28x28 = canvas_to_28x28(canvas_result.image_data)

        if image_28x28.max() == 0:
            st.warning("Canvas is empty - draw a digit first.")
            return

        result = predict.predict_with_confidence(
            model, image_28x28, preprocess.preprocess_single_image
        )
        render_result(result, image_28x28)


def sample_tab(model):
    samples = get_sample_digits()

    digit = st.radio(
        "Pick a real test-set digit to try:",
        options=list(range(config.NUM_CLASSES)),
        horizontal=True,
        key="sample_digit_choice",
    )

    image_28x28 = samples[digit]
    st.image(image_28x28.astype("uint8"), width=140, clamp=True, caption=f"True label: {digit}")

    if st.button("Predict", key="predict_sample"):
        result = predict.predict_with_confidence(
            model, image_28x28, preprocess.preprocess_single_image
        )
        render_result(result, image_28x28, true_digit=digit)


def main():
    try:
        model = get_model()
    except FileNotFoundError:
        st.error(
            "No trained model found. Run `python main.py` (or "
            "`python -m src.v2_cnn`) to train the CNN before using this app."
        )
        return

    tab_draw, tab_sample = st.tabs(["Draw a Digit", "Load Test Sample"])
    with tab_draw:
        draw_tab(model)
    with tab_sample:
        sample_tab(model)


if __name__ == "__main__":
    main()
