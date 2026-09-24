# EchoLV-XAI

Explainable AI-Based Left Ventricle Segmentation and CNN Classification from Echocardiographic Images using Grad-CAM and SHAP.

This repository contains an experimental research prototype for left-ventricle segmentation, LV-focused CNN classification, and Explainable AI (XAI) analysis from echocardiographic images.

## Medical Disclaimer

This is an experimental research prototype and is not intended for clinical diagnosis or treatment decisions. No clinical performance is claimed by this scaffold.

## Research Objectives

The primary research question addressed by this project is:
**Can explicit left-ventricle segmentation improve the performance and anatomical interpretability of CNN-based classification of echocardiographic images?**

The system evaluates and compares three experimental paradigms:
1. **Full Image Classification:** Raw echocardiogram -> CNN
2. **Cropped LV Classification:** Echocardiogram -> U-Net Segmentation -> Bounding Box ROI -> CNN
3. **Masked LV Classification:** Echocardiogram -> U-Net Segmentation -> Background Zeroed -> CNN

Explainable AI techniques (Grad-CAM and SHAP) are applied to determine whether the model focuses on anatomically relevant cardiac structures.

## System Architecture

1. **Preprocessing & Segmentation:** Images are converted to grayscale and resized. A PyTorch U-Net (or U-Net++) segments the Left Ventricle.
2. **ROI Extraction:** The segmentation mask defines a bounding box to isolate the LV.
3. **Classification:** A ResNet-18 model predicts the pathology (or state) based on the input mode.
4. **Explainability:** Feature embeddings are analyzed via PCA/t-SNE/UMAP. Grad-CAM and SHAP extract attention heatmaps and pixel attributions to validate clinical focus.

## Installation

```bash
git clone https://github.com/Akash-C17/EchoLV-XAI.git
cd phase_1_project
python -m venv .venv

# On Windows:
.venv\Scripts\Activate.ps1
# On Linux/Mac:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Preparation

Datasets should not be tracked by Git. Place your data in the following directories as defined in `config.yaml`:
- Segmentation data (e.g., CAMUS): `data/raw/segmentation/`
- Classification data: `data/raw/classification/`

Create patient-level training splits to prevent data leakage:
```bash
python -m src.data.split_dataset --config config.yaml
```

## Execution Commands

### 1. Segmentation
Train and evaluate the U-Net segmentation model:
```bash
python -m src.segmentation.train --config config.yaml
python -m src.segmentation.evaluate --config config.yaml
```

### 2. Classification
Train and evaluate the ResNet classifier. Use `--mode` to switch between `full`, `crop`, or `masked` paradigms:
```bash
python -m src.classification.train --config config.yaml --mode full
python -m src.classification.evaluate --config config.yaml --mode full
```

### 3. Explainability & Features
Extract CNN features and run dimensionality reduction (PCA, t-SNE, UMAP):
```bash
python -m src.features.extractor --config config.yaml --mode full
python -m src.features.feature_analysis --features_csv outputs/features/full/extracted_features.csv --output_dir outputs/figures/
```

### 4. Streamlit Application
Launch the interactive web application to visualize the end-to-end inference pipeline:
```bash
streamlit run app/streamlit_app.py
```

## Project Structure

- `app/` - Streamlit application (`streamlit_app.py`)
- `data/` - Dataset directories and patient-level CSV splits
- `models/` - Saved PyTorch weights (`.pth`)
- `notebooks/` - Research notebooks for dataset exploration
- `outputs/` - Generated metrics, XAI heatmaps, and dimensionality reduction plots
- `src/` - Core Python modules (preprocessing, segmentation, classification, features, explainability, evaluation)
- `tests/` - Pytest validation suite
- `IMPLEMENTATION_PLAN.md` - Phase-by-phase development roadmap

## Results & Tables

*(To be filled upon completing clinical evaluations)*

| Model Paradigm | Accuracy | Precision | Recall | F1 Score | AUC |
|----------------|----------|-----------|--------|----------|-----|
| Full Image | - | - | - | - | - |
| Cropped LV | - | - | - | - | - |
| Masked LV | - | - | - | - | - |
