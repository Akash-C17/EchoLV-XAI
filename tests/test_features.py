import pytest
import torch
import numpy as np
from PIL import Image

from src.features.extractor import extract_features
from src.classification.models import EchoClassifier

def test_feature_extraction():
    model = EchoClassifier(num_classes=2, pretrained=False)
    
    # Mock dataloader
    class MockLoader:
        def __iter__(self):
            yield torch.randn(2, 1, 224, 224), torch.tensor([0, 1])
            
    features, labels = extract_features(model, MockLoader(), device=torch.device("cpu"))
    
    assert features.shape == (2, 512), "Should extract 512-dim features from 2 images"
    assert labels == [0, 1]
