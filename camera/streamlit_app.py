"""
streamlit_app.py

Web UI version of the fruit freshness detector (Streamlit).
Instead of a live camera feed, the user uploads a photo (or takes one with
their device camera), picks which model to run (SSD300-VGG16 or YOLOv8n),
and sees the detection result with bounding boxes and per-detection
confidence scores.

Usage:
    streamlit run streamlit_app.py
"""

import os
import sys

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# camera/ 和 ssd/ 是同级文件夹，需要把 ssd/ 加入模块搜索路径才能 import
SSD_DIR = os.path.join(SCRIPT_DIR, "..", "ssd")
sys.path.insert(0, SSD_DIR)

from inference import load_model as load_ssd_model, predict_image as predict_ssd, draw_predictions  # noqa: E402
from dataset import CATEGORY_ID_TO_NAME  # noqa: E402

from camera_app_yolo import (  # noqa: E402
    load_model as load_yolo_model,
    draw_detections,
    DEFAULT_MODEL_PATH as YOLO_DEFAULT_MODEL_PATH,
)

SSD_DEFAULT_CHECKPOINT = os.path.join(SCRIPT_DIR, "..", "ssd", "ssd_fruit_model.pth")

MAX_IMAGE_DIMENSION = 1280


def preprocess_image(image):
    """
    Shared preprocessing applied before either model runs, so both SSD and
    YOLO always see the same input:
    - fix orientation for photos taken on phones (EXIF rotation tag)
    - downscale very large photos so inference stays fast and the result
      image fits nicely on the page
    """
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) > MAX_IMAGE_DIMENSION:
        image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.LANCZOS)
    return image


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

    .stApp { background-color: #f6fafd; }

    .block-container { padding-top: 2.8rem; max-width: 700px; }

    .app-title { font-size: 1.9rem; font-weight: 700; color: #0f2a3d; margin-bottom: 0.1rem; }
    .app-subtitle { color: #64748b; margin-bottom: 1.8rem; font-size: 0.95rem; }

    .section-label { font-weight: 600; color: #334155; font-size: 0.85rem;
                      text-transform: uppercase; letter-spacing: 0.03em; margin: 1.1rem 0 0.4rem; }

    div[data-testid="stSegmentedControl"] label p { font-weight: 500; }

    .stButton>button {
        background-color: #2f9fe8; color: white; border: none; border-radius: 10px;
        padding: 0.6rem 1.4rem; font-weight: 600; width: 100%; transition: background-color 0.15s;
    }
    .stButton>button:hover { background-color: #1c7fc2; color: white; }

    div[data-testid="stImage"] img {
        border-radius: 14px; box-shadow: 0 1px 6px rgba(15, 42, 61, 0.10);
    }

    .badge {
        display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px;
        border-radius: 999px; font-size: 0.88rem; font-weight: 600; margin: 3px 6px 3px 0;
    }
    .badge.fresh { background: #e5f7ec; color: #187a3d; }
    .badge.rotten { background: #fdeaea; color: #c0392b; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">🍎 Fruit Freshness Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Upload a photo or take one, pick a model, and see the result.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
algo = st.segmented_control("Model", ["YOLOv8n", "SSD300-VGG16"], default="YOLOv8n", label_visibility="collapsed")
algo = algo or "YOLOv8n"  # segmented_control allows deselecting to None; keep a model always picked

threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)

st.markdown('<div class="section-label">Image source</div>', unsafe_allow_html=True)
source = st.segmented_control(
    "Image source", ["Upload a photo", "Take a photo"], default="Upload a photo", label_visibility="collapsed"
)
source = source or "Upload a photo"
if source == "Upload a photo":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
else:
    uploaded = st.camera_input("Take a photo", label_visibility="collapsed")

if uploaded is not None:
    image = preprocess_image(Image.open(uploaded))
    st.image(image, caption="Input image", use_container_width=True)

    if st.button("Run detection"):
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

        if not detections:
            st.info("No fruit detected.")
        else:
            st.markdown('<div class="section-label">Detected fruit</div>', unsafe_allow_html=True)
            badges = "".join(
                f'<span class="badge {"fresh" if name.startswith("Fresh") else "rotten"}">'
                f'{name} · {conf:.2f}</span>'
                for name, conf in detections
            )
            st.markdown(badges, unsafe_allow_html=True)
