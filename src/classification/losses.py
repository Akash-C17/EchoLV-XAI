import torch.nn as nn

class ClassificationLoss(nn.Module):
    """
    Standard CrossEntropyLoss for classification.
    Allows easy extensibility for FocalLoss or ClassWeights if needed.
    """
    def __init__(self, class_weights=None):
        super(ClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, targets):
        return self.criterion(logits, targets)
