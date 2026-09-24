import streamlit as st
import torch
from PIL import Image
import numpy as np
import time

from src.utils.config import load_config
from src.segmentation.unet import UNet
from src.classification.models import EchoClassifier
from src.preprocessing.image_preprocessing import preprocess_image_and_mask
from src.preprocessing.roi_processing import extract_roi, apply_mask
from src.explainability.gradcam import GradCAM
from src.explainability.visualization import generate_overlay

# Page Config
st.set_page_config(page_title="LV-XAI Echocardiography", layout="wide")

st.title("LV-XAI Echocardiography")
st.markdown("""
**This system is an experimental research prototype and is not intended for clinical diagnosis or treatment decisions.**
""")

@st.cache_resource
def load_models():
    config = load_config("config.yaml")
    device = torch.device("cpu")
    
    # Dummy load for demonstration (replace with actual path)
    seg_model = UNet(1, 1).to(device)
    seg_model.eval()
    
    cls_model = EchoClassifier(2, pretrained=False).to(device)
    cls_model.eval()
    
    return seg_model, cls_model, device

seg_model, cls_model, device = load_models()

uploaded_file = st.file_uploader("Upload Echocardiographic Image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    st.write("### Processing...")
    progress = st.progress(0)
    
    # 1. Load Image
    image = Image.open(uploaded_file).convert("L")
    img_tensor = torch.tensor(np.array(image.resize((256, 256)))).unsqueeze(0).unsqueeze(0).float() / 255.0
    progress.progress(25)
    
    # 2. Segmentation
    with torch.no_grad():
        seg_logits = seg_model(img_tensor)
        mask_pred = (torch.sigmoid(seg_logits) > 0.5).squeeze().numpy() * 255
        mask_img = Image.fromarray(mask_pred.astype(np.uint8))
    progress.progress(50)
    
    # 3. ROI
    roi_img = extract_roi(image, mask_img) or image
    progress.progress(75)
    
    # 4. Classification & XAI (Mocked forward pass for UI rendering)
    cls_tensor = torch.tensor(np.array(roi_img.resize((224, 224)))).unsqueeze(0).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        cls_logits = cls_model(cls_tensor)
        probs = torch.softmax(cls_logits, dim=1).squeeze().numpy()
        pred_class = np.argmax(probs)
    progress.progress(100)
    
    # Display
    st.markdown("---")
    st.header("LV SEGMENTATION")
    col1, col2, col3 = st.columns(3)
    col1.image(image, caption="Original Image")
    col2.image(mask_img, caption="Predicted LV Mask")
    col3.image(roi_img, caption="Extracted LV ROI")
    
    st.markdown("---")
    st.header("CLASSIFICATION")
    st.success(f"Prediction: **Class {pred_class}**")
    st.info(f"Confidence: **{probs[pred_class]*100:.1f}%**")
    
    st.markdown("---")
    st.header("EXPLAINABLE AI")
    st.write("Grad-CAM and SHAP attribution maps would appear here.")
