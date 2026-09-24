import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: Unnormalized logits (B, C, H, W)
            targets: Binary mask (B, C, H, W)
        """
        probs = torch.sigmoid(logits)
        
        # Flatten label and prediction tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()                            
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)  
        
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
