# EchoLV-XAI

Explainable AI-Based Left Ventricle Segmentation and CNN Classification from Echocardiographic Images using Grad-CAM and SHAP.

This repository contains an experimental research prototype for left-ventricle segmentation, LV-focused CNN classification, and Explainable AI (XAI) analysis from echocardiographic images.

## Medical Disclaimer

This is an experimental research prototype and is not intended for clinical diagnosis or treatment decisions. No clinical performance is claimed by this scaffold.

## Research Objectives

The primary research question addressed by this project is:
**Can explicit left-ventricle segmentation improve the performance and anatomical interpretability of CNN-based classification of echocardiographic images?**

The system compares three paradigms:
1. Full Image Classification
2. Cropped LV Classification
3. Masked LV Classification

The classification predictions are then analyzed using Grad-CAM and SHAP to determine whether the model focuses on anatomically relevant structures.

## Environment Setup

To run this project locally, ensure you have Python installed, then set up the environment:

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\Activate.ps1
# On Linux/Mac:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Note: The project supports CPU execution by default. For GPU training, ensure you install a CUDA-compatible PyTorch build separately.

## Configuration

Dataset locations, hyperparameters, and model configurations are managed via `config.yaml`. Dataset paths must be defined before running the pipelines.

## Development Progress

The project is structured into 10 development phases. Please refer to `IMPLEMENTATION_PLAN.md` for detailed information on each phase.

**Current Status:** Phase 1 Complete.

### Phase 1 Verification

Validate the configuration schema:
```bash
python -m src.utils.config --config config.yaml
```

Validate configured dataset files and locations:
```bash
python -m src.data.dataset_validator --config config.yaml
```

Run the unit tests:
```bash
python -m pytest
```

## Project Architecture

- `app/` - Streamlit application components
- `data/` - Dataset storage (raw and processed splits)
- `models/` - Saved model checkpoints
- `notebooks/` - Research and exploration notebooks
- `outputs/` - Generated figures, logs, and outputs
- `src/` - Core Python modules
- `tests/` - Unit testing suite
