"""Segmentation evaluation on the held-out test set."""

import argparse
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.data.dataset_loader import EchoSegmentationDataset
from src.data.transforms import ToTensorDual, ComposeTransforms
from src.segmentation.unet import UNet
from src.segmentation.unetplusplus import UNetPlusPlus
from src.segmentation.metrics import SegmentationMetrics

logger = get_logger("lv_xai.evaluate_segmentation")

def save_qualitative_results(images, masks, preds, output_dir, batch_idx):
    for i in range(images.size(0)):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        img = to_pil_image(images[i])
        mask = to_pil_image(masks[i].float())
        pred = to_pil_image(preds[i].float())
        
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        axes[2].imshow(pred, cmap='gray')
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')
        
        axes[3].imshow(img, cmap='gray')
        axes[3].imshow(pred, cmap='jet', alpha=0.3)
        axes[3].set_title('Overlay')
        axes[3].axis('off')
        
        plt.savefig(os.path.join(output_dir, f"result_batch{batch_idx}_img{i}.png"))
        plt.close()

def _main():
    parser = argparse.ArgumentParser(description="Evaluate LV Segmentation Model")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    splits_dir = config.root_dir / "data" / "splits"
    seg_dir = config.path("segmentation_data")
    out_dir = config.path("outputs") / "segmentation" / "evaluation"
    model_dir = config.path("models") / "segmentation"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    test_dataset = EchoSegmentationDataset(splits_dir / "test.csv", seg_dir, transform=ComposeTransforms([ToTensorDual()]))
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    model_name = config.values["segmentation"]["model"].lower()
    in_channels = config.values["segmentation"]["input_channels"]
    out_channels = config.values["segmentation"]["output_channels"]
    
    if model_name == "unet++":
        model = UNetPlusPlus(in_channels, out_channels).to(device)
    else:
        model = UNet(in_channels, out_channels).to(device)
        
    model_path = model_dir / f"best_{model_name}.pth"
    if not model_path.exists():
        logger.error(f"Model weights not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    metrics = SegmentationMetrics(threshold=config.values["segmentation"]["threshold"])
    
    logger.info("Starting evaluation on test set...")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > config.values["segmentation"]["threshold"]).float()
            
            metrics.update(probs, masks)
            
            # Save qualitative results for the first few batches (up to ~20 examples)
            if batch_idx < 5:
                save_qualitative_results(images.cpu(), masks.cpu(), preds.cpu(), out_dir, batch_idx)

    final_metrics = metrics.compute()
    logger.info("Evaluation Results:")
    for k, v in final_metrics.items():
        logger.info(f"{k.capitalize()}: {v:.4f}")
        
    with open(out_dir / "test_metrics.txt", "w") as f:
        for k, v in final_metrics.items():
            f.write(f"{k.capitalize()}: {v:.4f}\n")

if __name__ == "__main__":
    _main()
