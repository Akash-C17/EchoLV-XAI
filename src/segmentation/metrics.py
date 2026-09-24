import torch

def calculate_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """
    Calculate TP, FP, TN, FN for a batch of predictions.
    preds: Probabilities (B, C, H, W)
    targets: Binary mask (B, C, H, W)
    """
    preds = (preds > threshold).float()
    
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    tn = ((1 - preds) * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    
    return tp, fp, tn, fn

def dice_coefficient(tp, fp, tn, fn, smooth=1e-6):
    return (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

def iou_score(tp, fp, tn, fn, smooth=1e-6):
    return (tp + smooth) / (tp + fp + fn + smooth)

def precision_score(tp, fp, tn, fn, smooth=1e-6):
    return (tp + smooth) / (tp + fp + smooth)

def recall_score(tp, fp, tn, fn, smooth=1e-6):
    return (tp + smooth) / (tp + fn + smooth)

def specificity_score(tp, fp, tn, fn, smooth=1e-6):
    return (tn + smooth) / (tn + fp + smooth)

class SegmentationMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()
        
    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.tn = 0.0
        self.fn = 0.0
        
    def update(self, preds, targets):
        tp, fp, tn, fn = calculate_confusion_matrix(preds, targets, self.threshold)
        self.tp += tp.item()
        self.fp += fp.item()
        self.tn += tn.item()
        self.fn += fn.item()
        
    def compute(self):
        return {
            "dice": dice_coefficient(self.tp, self.fp, self.tn, self.fn),
            "iou": iou_score(self.tp, self.fp, self.tn, self.fn),
            "precision": precision_score(self.tp, self.fp, self.tn, self.fn),
            "recall": recall_score(self.tp, self.fp, self.tn, self.fn),
            "specificity": specificity_score(self.tp, self.fp, self.tn, self.fn)
        }
