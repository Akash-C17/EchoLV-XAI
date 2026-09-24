"""Classification training loop with Transfer Learning."""

import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.classification.dataset import EchoClassificationDataset
from src.classification.models import EchoClassifier
from src.classification.losses import ClassificationLoss
from src.classification.evaluate import evaluate_model

logger = get_logger("lv_xai.train_classification")

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        
    return total_loss / len(dataloader.dataset)

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", default="full", choices=["full", "crop", "masked"])
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config.values["project"].get("seed", 42))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    splits_dir = config.root_dir / "data" / "splits"
    cls_dir = config.path("classification_data")
    mask_dir = config.path("segmentation_data").parent / "masks" if args.mode != "full" else None
    
    model_dir = config.path("models") / "classification"
    out_dir = config.path("outputs") / "classification" / args.mode
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    from torchvision.transforms import ToTensor, Resize, Compose
    transform = Compose([Resize((224, 224)), ToTensor()])
    
    train_dataset = EchoClassificationDataset(splits_dir / "train.csv", cls_dir, transform=transform, mode=args.mode, mask_dir=mask_dir)
    val_dataset = EchoClassificationDataset(splits_dir / "val.csv", cls_dir, transform=transform, mode=args.mode, mask_dir=mask_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=config.values["classification"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.values["classification"]["batch_size"], shuffle=False)
    
    num_classes = config.values["classification"]["num_classes"]
    model = EchoClassifier(num_classes=num_classes, pretrained=config.values["classification"]["pretrained"]).to(device)
    
    # Stage 1: Train classifier head only
    logger.info("Stage 1: Training classifier head")
    model.freeze_backbone()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.values["classification"]["learning_rate"] * 10)
    criterion = ClassificationLoss()
    
    for epoch in range(1, 6): # Brief warmup
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        logger.info(f"Warmup Epoch {epoch} - Loss: {train_loss:.4f}")
        
    # Stage 2: Fine-tune backbone
    logger.info("Stage 2: Fine-tuning backbone")
    model.unfreeze_backbone(unfreeze_from_layer=3) # Unfreeze top layers
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.values["classification"]["learning_rate"])
    
    epochs = config.values["classification"]["epochs"]
    patience = config.values["training"]["early_stopping_patience"]
    best_val_auc = 0.0
    patience_counter = 0
    history = []
    
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics, _, _, _ = evaluate_model(model, val_loader, device, num_classes)
        
        logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Acc: {metrics['accuracy']:.4f} | Val AUC: {metrics.get('auc', 0):.4f}")
        
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_acc": metrics["accuracy"], "val_auc": metrics.get("auc", 0)
        })
        
        val_metric = metrics.get("auc", metrics["accuracy"])
        if val_metric > best_val_auc:
            best_val_auc = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), model_dir / f"best_{args.mode}_resnet18.pth")
            logger.info("Saved new best model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break
                
    df = pd.DataFrame(history)
    df.to_csv(out_dir / "training_history.csv", index=False)
    
if __name__ == "__main__":
    _main()
