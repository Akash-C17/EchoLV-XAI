import streamlit as st
import torch
import numpy as np
import time
from pathlib import Path
from PIL import Image

from src.utils.config import load_config
from src.segmentation.unet import UNet
from src.classification.models import EchoClassifier
from src.segmentation.inference import predict_mask
from src.classification.inference import predict as cls_predict
from src.preprocessing.roi_processing import extract_roi, apply_mask
from src.explainability.gradcam import GradCAM, get_target_layer
from src.explainability.visualization import generate_overlay

st.set_page_config(
    page_title="LV-XAI Echocardiography",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("LV-XAI Echocardiography Analysis")

st.warning(
    "This system is an experimental research prototype and is not intended for "
    "clinical diagnosis or treatment decisions."
)

# Sidebar
with st.sidebar:
    st.header("Configuration")
    cls_mode = st.selectbox("Classification Mode", ["full", "crop", "masked"])
    class_names = st.text_input("Class Names (comma-separated)", "Normal,Abnormal")
    class_list = [c.strip() for c in class_names.split(",")]

@st.cache_resource
def load_models():
    config = load_config("config.yaml")
    device = torch.device("cpu")
    seg_cfg = config.values["segmentation"]
    cls_cfg = config.values["classification"]

    seg_model = UNet(seg_cfg["input_channels"], seg_cfg["output_channels"]).to(device)
    seg_model.eval()

    cls_model = EchoClassifier(
        num_classes=cls_cfg["num_classes"], pretrained=False
    ).to(device)
    cls_model.eval()

    seg_model_path = config.path("models") / "segmentation" / f"best_{seg_cfg['model']}.pth"
    cls_model_path = config.path("models") / "classification" / f"best_{cls_mode}_resnet18.pth"

    if seg_model_path.exists():
        seg_model.load_state_dict(torch.load(seg_model_path, map_location=device))
    else:
        st.sidebar.warning("Segmentation model weights not found. Using untrained model.")

    if cls_model_path.exists():
        cls_model.load_state_dict(torch.load(cls_model_path, map_location=device))
    else:
        st.sidebar.warning("Classification model weights not found. Using untrained model.")

    target_layer = get_target_layer(cls_model, "resnet18")
    gradcam = GradCAM(cls_model, target_layer)

    return seg_model, cls_model, gradcam, config, device

seg_model, cls_model, gradcam, config, device = load_models()

# File Upload
uploaded = st.file_uploader(
    "Upload Echocardiographic Image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded is not None:
    t_start = time.time()
    original = Image.open(uploaded).convert("L")

    seg_size = config.values["segmentation"].get("segmentation_image_size", 256)
    cls_size = config.values["data"]["image_size"]
    threshold = config.values["segmentation"]["threshold"]

    progress = st.progress(0, text="Running segmentation...")

    # Segmentation
    mask_arr = predict_mask(seg_model, original, (seg_size, seg_size), threshold, device)
    mask_pil = Image.fromarray(mask_arr)
    progress.progress(33, text="Extracting ROI...")

    # ROI
    if cls_mode == "crop":
        input_img = extract_roi(original, mask_pil) or original
    elif cls_mode == "masked":
        input_img = apply_mask(original, mask_pil)
    else:
        input_img = original
    progress.progress(55, text="Classifying...")

    # Classification
    result = cls_predict(cls_model, input_img, (cls_size, cls_size), device, class_list)
    progress.progress(75, text="Generating Grad-CAM...")

    # Grad-CAM
    arr = np.array(input_img.resize((cls_size, cls_size))) / 255.0
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    heatmap, _ = gradcam(tensor, target_class=result["predicted_class"])
    img_uint8 = np.array(input_img.resize((cls_size, cls_size))).astype(np.uint8)
    overlay = generate_overlay(img_uint8, heatmap)
    progress.progress(100, text="Done.")

    elapsed = time.time() - t_start

    # Results display
    st.markdown("---")
    st.header("LV Segmentation")
    col1, col2, col3 = st.columns(3)
    col1.image(original, caption="Original Image", use_container_width=True)
    col2.image(mask_pil, caption="Predicted LV Mask", use_container_width=True)
    col3.image(input_img, caption=f"Input to Classifier ({cls_mode})", use_container_width=True)

    st.markdown("---")
    st.header("Classification Result")
    col_a, col_b = st.columns(2)
    col_a.metric("Prediction", result["predicted_label"])
    col_b.metric("Confidence", f"{result['confidence'] * 100:.1f}%")

    st.markdown("---")
    st.header("Explainable AI")
    col_gc, col_ov = st.columns(2)
    col_gc.image(heatmap, caption="Grad-CAM Heatmap", clamp=True, use_container_width=True)
    col_ov.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)

    st.markdown("---")
    st.caption(f"Processing time: {elapsed:.2f}s | Device: {device}")
