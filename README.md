# LV-XAI

Research scaffold for explainable left-ventricle segmentation and classification
from echocardiographic images.

This repository currently contains **Phase 1 only**:

- Configuration loading and validation
- Reproducible random seed setup
- Structured logging
- Dataset structure and file-integrity validation
- Project directory scaffold

Segmentation, classification, Grad-CAM, SHAP, and clinical evaluation are
intentionally not implemented yet.

## Medical disclaimer

This is an experimental research prototype and is not intended for clinical
diagnosis or treatment decisions. No clinical performance is claimed by this
scaffold.

## Environment setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project supports CPU execution. Install a CUDA-compatible PyTorch build
separately when GPU training is required.

## Configuration

Edit [config.yaml](config.yaml) to point to the actual datasets. Dataset files
must not be fabricated or copied into this repository. The validator accepts
image files and, when configured, paired segmentation masks.

## Verify Phase 1

Validate the configuration:

```powershell
python -m src.utils.config --config config.yaml
```

Validate configured dataset locations:

```powershell
python -m src.data.dataset_validator --config config.yaml
```

Run the unit tests:

```powershell
python -m pytest
```

The validator reports missing directories, unsupported files, unreadable image
files, duplicate content, missing masks, and image/mask dimension mismatches.
It returns a non-zero exit status when configured paths are missing or errors
are found. An empty dataset is reported as a warning because the real dataset
has not been supplied yet.

## Planned phases

1. Project structure, configuration, environment, and dataset validation
2. Dataset exploration, patient-level splitting, preprocessing, and visualization
3. U-Net segmentation and evaluation
4. LV ROI extraction
5. Transfer-learning classification
6. Feature extraction and analysis
7. Grad-CAM
8. SHAP
9. XAI validation, error analysis, and ablation
10. Inference, Streamlit, tests, and research outputs

