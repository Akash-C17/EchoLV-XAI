# Implementation Plan: EchoLV-XAI

This document outlines the step-by-step implementation plan for building the Explainable AI-Based Left Ventricle Segmentation and CNN Classification project.

## Phase 1: Foundation (Completed)
- Project directory scaffold
- Environment configuration and dependency setup
- Reproducible random seed setup
- Structured logging configuration
- Dataset structure validation script

## Phase 2: Data Engineering
- Dataset exploration notebook
- Patient-level data splitting (Train, Validation, Test) to avoid data leakage
- Image preprocessing (Grayscale conversion, Resizing)
- Intensity normalization and contrast enhancement

## Phase 3: Segmentation Module
- U-Net implementation
- Segmentation training loop (Loss functions, Adam optimizer, early stopping)
- Segmentation evaluation (Dice score, IoU, Precision, Recall)

## Phase 4: Region of Interest (ROI) Extraction
- Bounding box extraction based on LV mask
- ROI cropping with configurable padding
- Cropped and masked image visualization

## Phase 5: Baseline Classification
- CNN classifier implementation (ResNet-18)
- Transfer learning pipeline
- Full-image classification (Mode 1) training and evaluation

## Phase 6: LV-Focused Classification
- Mode 2 (Cropped LV) and Mode 3 (Masked LV) classification
- Deep CNN feature extraction
- Dimensionality reduction and visualization (PCA, t-SNE, UMAP)

## Phase 7: Explainable AI - Grad-CAM
- Grad-CAM implementation tailored for ResNet
- Attention heatmap generation
- Visualization of Grad-CAM overlay

## Phase 8: Explainable AI - SHAP
- SHAP explainer implementation
- Feature/Pixel attribution map generation
- Explanation visual overlays

## Phase 9: Analysis and Evaluation
- XAI vs LV-mask quantitative evaluation
- Error analysis (False positives/negatives)
- Ablation study across classification modes

## Phase 10: Deployment and Reporting
- End-to-end inference pipeline
- Streamlit application deployment
- End-to-end testing suite
- Auto-generation of research tables and figures
