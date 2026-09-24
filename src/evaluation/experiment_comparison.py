import numpy as np

def calculate_xai_lv_overlap(heatmap: np.ndarray, mask: np.ndarray, threshold: float = 0.5):
    """
    Evaluate how much of the XAI heatmap focuses on the LV region.
    Args:
        heatmap: Normalized Grad-CAM or SHAP map [0, 1].
        mask: Binary LV mask.
    Returns:
        overlap metrics dictionary.
    """
    # Binarize heatmap
    binary_heat = (heatmap > threshold).astype(np.float32)
    binary_mask = (mask > 0).astype(np.float32)
    
    intersection = np.sum(binary_heat * binary_mask)
    union = np.sum(np.logical_or(binary_heat, binary_mask))
    
    iou = intersection / (union + 1e-6)
    
    # Energy metrics
    total_energy = np.sum(heatmap)
    energy_in_lv = np.sum(heatmap * binary_mask)
    energy_ratio = energy_in_lv / (total_energy + 1e-6)
    
    return {
        "iou": iou,
        "energy_in_lv": energy_in_lv,
        "energy_ratio": energy_ratio
    }

def analyze_errors(preds, targets, confidences, image_paths):
    """
    Categorize errors into false positives, false negatives, etc.
    """
    errors = []
    for p, t, conf, path in zip(preds, targets, confidences, image_paths):
        if p != t:
            errors.append({
                "path": path,
                "pred": p,
                "target": t,
                "confidence": conf,
                "type": "FP" if p == 1 else "FN"
            })
    return sorted(errors, key=lambda x: x["confidence"], reverse=True)
