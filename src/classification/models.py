import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class EchoClassifier(nn.Module):
    """
    ResNet-18 based classifier for Echocardiography.
    Modified to accept 1-channel (grayscale) input if necessary,
    and exposes the final feature vector before classification.
    """
    def __init__(self, num_classes=2, pretrained=True, in_channels=1):
        super(EchoClassifier, self).__init__()
        
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.backbone = resnet18(weights=weights)
        else:
            self.backbone = resnet18(weights=None)
            
        # Modify first conv layer to accept 1 channel instead of 3
        if in_channels == 1:
            original_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                1, original_conv.out_channels, 
                kernel_size=original_conv.kernel_size, 
                stride=original_conv.stride, 
                padding=original_conv.padding, 
                bias=original_conv.bias
            )
            # If pretrained, sum the weights across the RGB channels for the new 1-channel layer
            if pretrained:
                with torch.no_grad():
                    self.backbone.conv1.weight.copy_(original_conv.weight.sum(dim=1, keepdim=True))
                    
        # Remove the final fully connected layer
        self.num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Classification head
        self.classifier = nn.Linear(self.num_features, num_classes)
        
    def forward(self, x, return_features=False):
        """
        Forward pass.
        If return_features is True, returns both the logits and the global pooled feature vector.
        """
        features = self.backbone(x) # (B, 512)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
        
    def freeze_backbone(self):
        """Freeze all layers except the classifier head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
            
    def unfreeze_backbone(self, unfreeze_from_layer=3):
        """
        Unfreeze the backbone partially or fully.
        ResNet has layer1, layer2, layer3, layer4.
        """
        # First, unfreeze everything
        for param in self.parameters():
            param.requires_grad = True
            
        # Freeze up to a specific layer if desired
        if unfreeze_from_layer > 1:
            for param in self.backbone.conv1.parameters(): param.requires_grad = False
            for param in self.backbone.bn1.parameters(): param.requires_grad = False
            for param in self.backbone.layer1.parameters(): param.requires_grad = False
        if unfreeze_from_layer > 2:
            for param in self.backbone.layer2.parameters(): param.requires_grad = False
        if unfreeze_from_layer > 3:
            for param in self.backbone.layer3.parameters(): param.requires_grad = False
