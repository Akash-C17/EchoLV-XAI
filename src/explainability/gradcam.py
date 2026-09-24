"""Grad-CAM implementation for PyTorch CNNs."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import cv2


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Mathematical procedure:
        1. Forward pass through CNN up to target convolutional layer.
        2. Compute gradient of target class score w.r.t. feature maps.
        3. Global-average-pool gradients to obtain channel weights alpha_k.
        4. Weight feature maps by alpha_k, apply ReLU, normalize, resize.

    Reference: Selvaraju et al. (2017) - Grad-CAM: Visual Explanations from Deep Networks.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        self._hooks.append(
            self.target_layer.register_forward_hook(self._save_activation)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, input, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Clean up registered hooks to prevent memory leaks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __call__(
        self,
        x: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate Grad-CAM heatmap for an input image tensor.

        Args:
            x: Input tensor of shape (1, C, H, W).
            target_class: Class index to explain. Defaults to predicted class.

        Returns:
            Tuple of (heatmap, target_class) where heatmap is float32 in [0, 1],
            shape (H, W) matching the input spatial dimensions.
        """
        self.model.eval()
        x = x.clone().detach().requires_grad_(True)

        # Forward pass
        output = self.model(x)
        if isinstance(output, tuple):
            output = output[0]

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        # Backward for target class
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=False)

        # alpha_k: global average pooling on gradients
        gradients = self._gradients.cpu().numpy()[0]    # (C, Hf, Wf)
        activations = self._activations.cpu().numpy()[0]  # (C, Hf, Wf)
        weights = np.mean(gradients, axis=(1, 2))         # (C,)

        # Weighted sum of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for k, w in enumerate(weights):
            cam += w * activations[k]

        # ReLU
        cam = np.maximum(cam, 0)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize to match input spatial dimensions
        _, _, H, W = x.shape
        cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)

        return cam.astype(np.float32), target_class


def get_target_layer(model: nn.Module, model_name: str = "resnet18") -> nn.Module:
    """
    Automatically retrieve the recommended target convolutional layer for Grad-CAM.

    For ResNet architectures, the final convolutional layer in layer4 is used,
    as it captures the highest-level spatial features before global average pooling.
    """
    if model_name in ("resnet18", "resnet50"):
        return model.backbone.layer4[-1].conv2
    raise ValueError(f"Unsupported model_name for automatic layer detection: {model_name}")
