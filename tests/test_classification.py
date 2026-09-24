import torch
import pytest

from src.classification.models import EchoClassifier
from src.classification.losses import ClassificationLoss

def test_echoclassifier_shape():
    model = EchoClassifier(num_classes=2, in_channels=1, pretrained=False)
    x = torch.randn(2, 1, 224, 224)
    out = model(x)
    assert out.shape == (2, 2), "Classification output shape mismatch"
    
def test_echoclassifier_features():
    model = EchoClassifier(num_classes=2, in_channels=1, pretrained=False)
    x = torch.randn(2, 1, 224, 224)
    logits, features = model(x, return_features=True)
    assert logits.shape == (2, 2)
    assert features.shape == (2, 512), "ResNet18 feature vector should be 512-dim"

def test_classification_loss():
    loss_fn = ClassificationLoss()
    logits = torch.randn(2, 2)
    targets = torch.tensor([0, 1])
    
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0, "Loss should be non-negative"
