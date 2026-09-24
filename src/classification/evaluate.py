import argparse
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.classification.dataset import EchoClassificationDataset
from src.classification.models import EchoClassifier

logger = get_logger("lv_xai.evaluate_classification")

def evaluate_model(model, dataloader, device, num_classes=2):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            if num_classes == 2:
                all_probs.extend(probs[:, 1].cpu().numpy())
            else:
                all_probs.extend(probs.cpu().numpy())
                
    metrics = {
        "accuracy": accuracy_score(all_targets, all_preds),
        "precision": precision_score(all_targets, all_preds, average='weighted', zero_division=0),
        "recall": recall_score(all_targets, all_preds, average='weighted', zero_division=0),
        "f1": f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    }
    
    if num_classes == 2:
        try:
            metrics["auc"] = roc_auc_score(all_targets, all_probs)
        except ValueError:
            metrics["auc"] = 0.5 # if only one class present in targets
            
    return metrics, all_targets, all_preds, all_probs

def plot_confusion_matrix(targets, preds, output_path):
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", default="full", choices=["full", "crop", "masked"])
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    splits_dir = config.root_dir / "data" / "splits"
    cls_dir = config.path("classification_data")
    mask_dir = config.path("segmentation_data").parent / "masks" if args.mode != "full" else None
    
    out_dir = config.path("outputs") / "classification" / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Needs torchvision transforms
    from torchvision.transforms import ToTensor, Resize, Compose
    transform = Compose([Resize((224, 224)), ToTensor()])
    
    test_dataset = EchoClassificationDataset(splits_dir / "test.csv", cls_dir, transform=transform, mode=args.mode, mask_dir=mask_dir)
    test_loader = DataLoader(test_dataset, batch_size=config.values["classification"]["batch_size"], shuffle=False)
    
    model = EchoClassifier(
        num_classes=config.values["classification"]["num_classes"],
        pretrained=False
    ).to(device)
    
    model_path = config.path("models") / "classification" / f"best_{args.mode}_resnet18.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        logger.warning(f"Model not found at {model_path}. Using random weights.")
        
    metrics, targets, preds, probs = evaluate_model(model, test_loader, device, num_classes=config.values["classification"]["num_classes"])
    
    logger.info("--- Evaluation Results ---")
    for k, v in metrics.items():
        logger.info(f"{k.capitalize()}: {v:.4f}")
        
    plot_confusion_matrix(targets, preds, out_dir / "confusion_matrix.png")
    
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(classification_report(targets, preds, zero_division=0))

if __name__ == "__main__":
    _main()
