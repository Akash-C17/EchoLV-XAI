import pytest
import torch
import numpy as np

from src.explainability.gradcam import GradCAM
from src.classification.models import EchoClassifier

def test_gradcam():
    model = EchoClassifier(num_classes=2, in_channels=1, pretrained=False)
    
    # Target the last conv layer before the GAP
    # In resnet18, layer4 is the final sequential block
    target_layer = model.backbone.layer4[-1].conv2
    
    cam = GradCAM(model, target_layer)
    x = torch.randn(1, 1, 224, 224)
    
    heatmap, target_class = cam(x)
    
    assert heatmap.shape == (224, 224), "Grad-CAM heatmap should match input spatial dimensions"
    assert heatmap.min() >= 0 and heatmap.max() <= 1.0, "Heatmap should be normalized to [0, 1]"
    assert target_class in [0, 1], "Target class should be valid"
