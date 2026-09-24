import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.classification.dataset import EchoClassificationDataset
from src.classification.models import EchoClassifier

logger = get_logger("lv_xai.feature_extractor")

def extract_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            _, features = model(images, return_features=True)
            
            all_features.append(features.cpu())
            all_labels.extend(labels.tolist())
            
    return torch.cat(all_features, dim=0).numpy(), all_labels

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
    
    out_dir = config.path("outputs") / "features" / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    
    from torchvision.transforms import ToTensor, Resize, Compose
    transform = Compose([Resize((224, 224)), ToTensor()])
    
    dataset = EchoClassificationDataset(splits_dir / "test.csv", cls_dir, transform=transform, mode=args.mode, mask_dir=mask_dir)
    loader = DataLoader(dataset, batch_size=config.values["classification"]["batch_size"], shuffle=False)
    
    model = EchoClassifier(num_classes=config.values["classification"]["num_classes"], pretrained=False).to(device)
    model_path = config.path("models") / "classification" / f"best_{args.mode}_resnet18.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        logger.warning(f"Model not found at {model_path}. Using random weights for extraction.")
        
    logger.info("Extracting features...")
    features, labels = extract_features(model, loader, device)
    
    # Save as CSV
    feature_cols = [f"feature_{i}" for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=feature_cols)
    df["label"] = labels
    
    output_file = out_dir / "extracted_features.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Features saved to {output_file}")

if __name__ == "__main__":
    _main()
