import torch
import pytest

from src.segmentation.unet import UNet
from src.segmentation.unetplusplus import UNetPlusPlus
from src.segmentation.losses import BCEDiceLoss
from src.segmentation.metrics import SegmentationMetrics, calculate_confusion_matrix

def test_unet_shape():
    model = UNet(n_channels=1, n_classes=1)
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    assert out.shape == (2, 1, 128, 128), "U-Net output shape mismatch"

def test_unetplusplus_shape():
    model = UNetPlusPlus(n_channels=1, n_classes=1)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64), "U-Net++ output shape mismatch"

def test_bce_dice_loss():
    loss_fn = BCEDiceLoss()
    logits = torch.randn(2, 1, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
    
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss), "Loss should not be NaN"

def test_segmentation_metrics():
    metrics = SegmentationMetrics(threshold=0.5)
    
    # Perfect prediction
    preds = torch.ones(1, 1, 10, 10) * 10 # high logits
    targets = torch.ones(1, 1, 10, 10)
    
    probs = torch.sigmoid(preds)
    metrics.update(probs, targets)
    results = metrics.compute()
    
    assert results["dice"] > 0.99, "Perfect prediction should have Dice ~1"
    assert results["iou"] > 0.99, "Perfect prediction should have IoU ~1"
    
def test_confusion_matrix():
    preds = torch.tensor([[[[0.9, 0.1], [0.8, 0.2]]]])
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    
    tp, fp, tn, fn = calculate_confusion_matrix(preds, targets, threshold=0.5)
    
    assert tp.item() == 1
    assert fp.item() == 1
    assert tn.item() == 1
    assert fn.item() == 1
