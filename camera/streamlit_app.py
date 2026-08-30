# Streamlit UI: upload or take a photo, pick a model, see the detection result.

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SSD_DIR = os.path.join(SCRIPT_DIR, "..", "ssd")
sys.path.insert(0, SSD_DIR)

from inference import load_model as load_ssd_model, predict_image as predict_ssd, draw_predictions 
from dataset import CATEGORY_ID_TO_NAME  

from yolo_inference import (  
    load_model as load_yolo_model,
    draw_detections,
    DEFAULT_MODEL_PATH as YOLO_DEFAULT_MODEL_PATH,
)
from image_processing import enhance_image, preprocess_image  
from model_metrics import BEST_MODEL, OVERALL_METRICS, PER_CLASS_AP50 

SSD_DEFAULT_CHECKPOINT = os.path.join(SCRIPT_DIR, "..", "ssd", "ssd_fruit_model.pth")


@st.cache_resource(show_spinner=False)
def get_yolo_model(model_path):
    return load_yolo_model(model_path)


@st.cache_resource(show_spinner=False)
def get_ssd_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_ssd_model(checkpoint_path, device)
    return model, device


st.set_page_config(page_title="Fruit Freshness Detector", page_icon="🍎", layout="centered")

st.markdown(
    """
    <style>
    #MainMenu, footer, header { visibility: hidden; }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
    }

    .stApp { background-color: #f4efe4; }

    .block-container { padding-top: 2rem; padding-bottom: 2.5rem; max-width: 640px; }

    /* tighten default block gap */
    div[data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"] {
        gap: 0.85rem !important;
    }

    /* ---- header ---- */
    .app-header { display: flex; align-items: center; gap: 14px; margin-bottom: 0.4rem; }

    .st-key-mode_toggle {
        position: fixed !important; top: 1.4rem; right: 1.8rem; z-index: 999; width: auto !important;
    }
    .st-key-mode_toggle div[data-testid="stButtonGroup"] div[role="radiogroup"] {
        width: auto !important; padding: 3px; background: #ede5d3;
        box-shadow: 0 1px 4px rgba(43, 38, 32, 0.1);
    }
    .st-key-mode_toggle div[data-testid="stButtonGroup"] button[data-variant="segmented_control"] {
        flex: 0 0 auto !important; padding: 0.3rem 0.7rem !important;
    }
    .st-key-mode_toggle div[data-testid="stButtonGroup"] button[data-variant="segmented_control"] p {
        font-size: 0.78rem !important; white-space: nowrap;
    }
    .app-icon {
        width: 46px; height: 46px; border-radius: 14px; flex-shrink: 0;
        background: linear-gradient(135deg, #8ba077, #5f7a52);
        display: flex; align-items: center; justify-content: center; font-size: 1.4rem;
        box-shadow: 0 4px 10px rgba(95, 122, 82, 0.28);
    }
    .app-title { font-size: 1.5rem; font-weight: 700; color: #2b2620; line-height: 1.2; }
    .app-subtitle { color: #9c927e; font-size: 0.88rem; margin-top: 2px; }

    /* ---- card containers ---- */
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] {
        background: #fffdf8; border-radius: 18px !important; border: 1px solid #e7dfcd !important;
        box-shadow: 0 1px 3px rgba(43, 38, 32, 0.05) !important; padding: 1.35rem 1.5rem !important;
        gap: 0.9rem !important;
    }
    div[data-testid="stExpander"] {
        border-radius: 14px !important; border: 1px solid #e7dfcd !important; background: #faf6ec;
        box-shadow: none !important;
    }
    div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] {
        background: transparent !important; border: none !important; box-shadow: none !important;
        padding: 0 !important;
    }

    .field-label { font-weight: 600; color: #443d33; font-size: 0.82rem;
                    letter-spacing: 0.01em; margin: 0; }
    .hint-text { color: #9c927e; font-size: 0.82rem; margin: 0; }

    /* ---- segmented control ---- */
    div[data-testid="stElementContainer"]:has(div[data-testid="stButtonGroup"]):not(.st-key-mode_toggle) { width: 100% !important; }
    div[data-testid="stButtonGroup"] { width: 100%; }
    div[data-testid="stButtonGroup"] div[role="radiogroup"] {
        background: #ede5d3; border-radius: 12px; padding: 4px; gap: 0 !important;
        width: 100% !important; max-width: 100% !important; display: flex !important; flex-wrap: nowrap !important;
    }
    div[data-testid="stButtonGroup"] button[data-variant="segmented_control"] {
        border-radius: 9px !important; border: none !important; background: transparent;
        transition: all 0.15s; font-weight: 500; font-size: 0.9rem;
        flex: 1 1 0% !important; width: auto !important; min-width: 0 !important;
    }
    div[data-testid="stButtonGroup"] button[data-variant="segmented_control"] p { color: #7a7060; }
    div[data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"] {
        background: #ffffff !important; box-shadow: 0 1px 4px rgba(43, 38, 32, 0.14) !important;
    }
    div[data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"] p {
        color: #4f6b42 !important; font-weight: 600;
    }

    /* ---- slider ---- */
    div[data-testid="stSlider"] { padding-top: 0.2rem; }

    /* ---- button ---- */
    .stButton>button {
        background: #5f7a52; color: white; border: none; border-radius: 12px;
        padding: 0.65rem 1.4rem; font-weight: 600; width: 100%; transition: background-color 0.15s;
        box-shadow: 0 2px 8px rgba(95, 122, 82, 0.3);
    }
    .stButton>button:hover { background-color: #4d6542; color: white; }

    /* ---- images ---- */
    div[data-testid="stImage"] img {
        border-radius: 16px; box-shadow: 0 1px 6px rgba(43, 38, 32, 0.1);
    }
    div[data-testid="stImageCaption"] { color: #9c927e; }

    /* ---- file uploader ---- */
    [data-testid="stFileUploaderDropzone"] {
        background: #faf6ec !important; border-radius: 12px; border: 1.5px dashed #ded3ba;
    }

    /* ---- info box ---- */
    [data-testid="stAlert"] { background: #eef2e7 !important; border-radius: 12px; }
    [data-testid="stAlertContainer"] { background: #eef2e7 !important; }
    [data-testid="stAlert"] p { color: #4f6b42 !important; }

    /* ---- badges ---- */
    .badge {
        display: inline-flex; align-items: center; gap: 7px; padding: 6px 13px;
        border-radius: 999px; font-size: 0.86rem; font-weight: 600; margin: 3px 6px 3px 0;
    }
    .badge .dot { width: 7px; height: 7px; border-radius: 50%; }
    .badge.fresh { background: #eef2e7; color: #4f6b42; }
    .badge.fresh .dot { background: #7c9473; }
    .badge.rotten { background: #f7e9e0; color: #a15c34; }
    .badge.rotten .dot { background: #c17a56; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <div class="app-icon">🍎</div>
        <div>
            <div class="app-title">Fruit Freshness Detector</div>
            <div class="app-subtitle">Upload a photo or take one, and see the result.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

mode = st.segmented_control(
    "Mode",
    ["User Mode", "Developer Mode"],
    default="User Mode",
    label_visibility="collapsed",
    key="mode_toggle",
)
mode = mode or "User Mode"

with st.container(border=True):
    if mode == "User Mode":
        algo = BEST_MODEL  # best model only
        st.markdown(
            f'<div class="hint-text">Using <b>{algo}</b>, the best-performing model on our test set.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="field-label">MODEL</div>', unsafe_allow_html=True)
        algo = st.segmented_control(
            "Model", ["YOLOv8n", "SSD300-VGG16"], default="YOLOv8n", label_visibility="collapsed"
        )
        algo = algo or "YOLOv8n"  # keep a model selected

    st.markdown('<div class="field-label">CONFIDENCE THRESHOLD</div>', unsafe_allow_html=True)
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05, label_visibility="collapsed")

    if mode == "Developer Mode":
        with st.expander("📊 Model comparison (test set: 155 fruits)"):
            overall_df = pd.DataFrame(OVERALL_METRICS).T
            overall_df.index.name = "Model"
            display_df = overall_df.map(lambda v: "n/a" if pd.isna(v) else f"{v:.3f}")
            st.markdown("**Overall metrics**")
            st.dataframe(display_df, use_container_width=True)
            st.caption(
                "SSD300-VGG16's Precision/Recall are computed at a 0.25 confidence threshold; "
                "YOLOv8n's come from Ultralytics' own per-class best-F1 threshold, so the two are not "
                "computed the same way."
            )

            st.markdown("**Per-class AP@0.5**")
            per_class_df = pd.DataFrame(PER_CLASS_AP50).T
            st.bar_chart(per_class_df)

with st.container(border=True):
    st.markdown('<div class="field-label">IMAGE SOURCE</div>', unsafe_allow_html=True)
    source = st.segmented_control(
        "Image source", ["Upload a photo", "Take a photo"], default="Upload a photo", label_visibility="collapsed"
    )
    source = source or "Upload a photo"
    if source == "Upload a photo":
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    else:
        uploaded = st.camera_input("Take a photo", label_visibility="collapsed")

if uploaded is not None:
    try:
        image = preprocess_image(Image.open(uploaded))
        image = enhance_image(image)
    except Exception as e:
        st.error(f"Couldn't read this image ({e}). Please try a different file.")
        st.stop()

    st.image(image, caption="Input image", use_container_width=True)

    if st.button("Run detection"):
        detections = []
        try:
            with st.spinner("Detecting..."):
                if algo == "YOLOv8n":
                    model = get_yolo_model(YOLO_DEFAULT_MODEL_PATH)
                    image_bgr = np.array(image)[:, :, ::-1].copy()
                    results = model(image_bgr, conf=threshold, verbose=False)
                    boxes = results[0].boxes
                    annotated_bgr = draw_detections(image_bgr.copy(), boxes, model.names)
                    st.image(annotated_bgr[:, :, ::-1], caption="Detection result (YOLOv8n)", use_container_width=True)

                    detections = [
                        (model.names[int(box.cls[0].item())], float(box.conf[0].item()))
                        for box in boxes
                    ]
                else:
                    model, device = get_ssd_model(SSD_DEFAULT_CHECKPOINT)
                    boxes, labels, scores = predict_ssd(model, image, device, score_threshold=threshold)
                    annotated = draw_predictions(image, boxes, labels, scores)
                    st.image(annotated, caption="Detection result (SSD300-VGG16)", use_container_width=True)

                    detections = [
                        (CATEGORY_ID_TO_NAME.get(int(label.item()), f"id_{int(label.item())}"), float(score.item()))
                        for label, score in zip(labels, scores)
                    ]
        except FileNotFoundError as e:
            st.error(f"Model file not found ({e}). Make sure the trained weights are in place before running detection.")
            st.stop()
        except Exception as e:
            st.error(f"Detection failed: {e}")
            st.stop()

        if not detections:
            st.info("No fruit detected.")
        else:
            st.markdown('<div class="field-label">DETECTED FRUIT</div>', unsafe_allow_html=True)
            badges = "".join(
                f'<span class="badge {"fresh" if name.startswith("Fresh") else "rotten"}">'
                f'<span class="dot"></span>{name} · {conf:.2f}</span>'
                for name, conf in detections
            )
            st.markdown(badges, unsafe_allow_html=True)
