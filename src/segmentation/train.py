"""Segmentation training loop."""

import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.data.dataset_loader import EchoSegmentationDataset
from src.data.transforms import ToTensorDual, ComposeTransforms
from src.segmentation.unet import UNet
from src.segmentation.unetplusplus import UNetPlusPlus
from src.segmentation.losses import BCEDiceLoss
from src.segmentation.metrics import SegmentationMetrics

logger = get_logger("lv_xai.train_segmentation")

def train_one_epoch(model, dataloader, optimizer, criterion, device, metrics):
    model.train()
    metrics.reset()
    total_loss = 0.0
    
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        metrics.update(torch.sigmoid(logits), masks)
        
    return total_loss / len(dataloader.dataset), metrics.compute()

def evaluate(model, dataloader, criterion, device, metrics):
    model.eval()
    metrics.reset()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            
            total_loss += loss.item() * images.size(0)
            metrics.update(torch.sigmoid(logits), masks)
            
    return total_loss / len(dataloader.dataset), metrics.compute()

def plot_history(history, output_dir):
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    
    # Loss plot
    plt.figure()
    plt.plot(df['epoch'], df['train_loss'], label='Train')
    plt.plot(df['epoch'], df['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    plt.close()
    
    # Dice plot
    plt.figure()
    plt.plot(df['epoch'], df['train_dice'], label='Train')
    plt.plot(df['epoch'], df['val_dice'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "dice_curve.png"))
    plt.close()

def _main():
    parser = argparse.ArgumentParser(description="Train LV Segmentation Model")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config.values["project"].get("seed", 42))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Setup paths
    splits_dir = config.root_dir / "data" / "splits"
    seg_dir = config.path("segmentation_data")
    out_dir = config.path("outputs") / "segmentation"
    model_dir = config.path("models") / "segmentation"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Data loaders
    # Note: Replace ComposeTransforms([ToTensorDual()]) with actual preprocessing (resize, augmentations) as needed
    train_dataset = EchoSegmentationDataset(splits_dir / "train.csv", seg_dir, transform=ComposeTransforms([ToTensorDual()]))
    val_dataset = EchoSegmentationDataset(splits_dir / "val.csv", seg_dir, transform=ComposeTransforms([ToTensorDual()]))
    
    batch_size = config.values["segmentation"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    model_name = config.values["segmentation"]["model"].lower()
    in_channels = config.values["segmentation"]["input_channels"]
    out_channels = config.values["segmentation"]["output_channels"]
    
    if model_name == "unet++":
        model = UNetPlusPlus(in_channels, out_channels).to(device)
    else:
        model = UNet(in_channels, out_channels).to(device)
        
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.values["segmentation"]["learning_rate"],
        weight_decay=config.values["training"]["weight_decay"]
    )
    criterion = BCEDiceLoss()
    metrics = SegmentationMetrics(threshold=config.values["segmentation"]["threshold"])
    
    epochs = config.values["segmentation"]["epochs"]
    patience = config.values["training"]["early_stopping_patience"]
    
    best_val_dice = 0.0
    patience_counter = 0
    history = []
    
    for epoch in range(1, epochs + 1):
        train_loss, train_mets = train_one_epoch(model, train_loader, optimizer, criterion, device, metrics)
        val_loss, val_mets = evaluate(model, val_loader, criterion, device, metrics)
        
        logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f}, Dice: {train_mets['dice']:.4f} | Val Loss: {val_loss:.4f}, Dice: {val_mets['dice']:.4f}")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_dice": train_mets["dice"], "val_dice": val_mets["dice"],
            "train_iou": train_mets["iou"], "val_iou": val_mets["iou"]
        })
        
        if val_mets["dice"] > best_val_dice:
            best_val_dice = val_mets["dice"]
            patience_counter = 0
            torch.save(model.state_dict(), model_dir / f"best_{model_name}.pth")
            logger.info("Saved new best model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break
                
    torch.save(model.state_dict(), model_dir / f"last_{model_name}.pth")
    plot_history(history, out_dir)
    
if __name__ == "__main__":
    _main()
