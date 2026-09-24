import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Grad-CAM implementation tailored for PyTorch CNNs.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple
        self.gradients = grad_output[0]
        
    def __call__(self, x, target_class=None):
        """
        Generate Grad-CAM heatmap.
        Args:
            x: Input tensor (1, C, H, W)
            target_class: Class index for which Grad-CAM is computed.
                          If None, it uses the class with the highest score.
        """
        self.model.eval()
        
        # Forward pass
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
            
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
            
        self.model.zero_grad()
        
        # Target for backprop
        target = logits[0, target_class]
        target.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global Average Pooling on gradients (weights)
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weight the activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam = cam / cam_max
            
        # Resize to match input image
        _, _, H, W = x.size()
        cam = cv2.resize(cam, (W, H))
        
        return cam, target_class
