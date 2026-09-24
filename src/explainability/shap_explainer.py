import torch
import numpy as np
import matplotlib.pyplot as plt

class SHAPExplainer:
    """
    SHAP explanation for PyTorch models using shap.GradientExplainer.
    """
    def __init__(self, model, background_data):
        """
        Args:
            model: PyTorch model (must be in eval mode).
            background_data: A batch of representative background images (Tensor of shape B, C, H, W).
        """
        try:
            import shap
        except ImportError as e:
            raise ImportError("shap library is required for SHAPExplainer.") from e
            
        self.model = model
        self.model.eval()
        # Using GradientExplainer as DeepExplainer can sometimes have issues with certain PyTorch ops
        self.explainer = shap.GradientExplainer(self.model, background_data)
        
    def explain(self, image_tensor, target_class=None):
        """
        Compute SHAP values for a given image.
        Args:
            image_tensor: Input tensor (1, C, H, W).
            target_class: Class to explain (default is predicted class).
        Returns:
            shap_values: Numpy array of SHAP attributions.
        """
        if target_class is None:
            with torch.no_grad():
                logits = self.model(image_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                target_class = torch.argmax(logits, dim=1).item()
                
        # GradientExplainer requires requires_grad=True on inputs if not already set, 
        # but shap library handles this internally for standard inputs.
        shap_values, _ = self.explainer.shap_values(image_tensor)
        
        # shap_values is a list of arrays (one per class). We want the target class.
        if isinstance(shap_values, list):
            attributions = shap_values[target_class]
        else:
            attributions = shap_values
            
        # Squeeze batch dimension
        attributions = np.squeeze(attributions, axis=0)
        
        return attributions, target_class

def plot_shap(original_img, attributions, target_class, output_path):
    """
    Plot SHAP attribution map alongside the original image.
    Positive SHAP values (red) push the prediction towards the class.
    Negative SHAP values (blue) push the prediction away from the class.
    """
    import shap
    
    # We use SHAP's built-in image_plot which handles the red/blue diverging colormap
    # SHAP expects shapes (H, W, C) or (H, W)
    if original_img.ndim == 2:
        original_img = np.expand_dims(original_img, axis=-1)
    if attributions.ndim == 2:
        attributions = np.expand_dims(attributions, axis=-1)
        
    # SHAP expects list of attributions for multi-class, but we can just pass [attributions]
    # and [original_img] to use their plotting tool.
    fig = plt.figure()
    shap.image_plot([attributions], np.array([original_img]), show=False)
    plt.title(f"SHAP Attributions (Class {target_class})")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
