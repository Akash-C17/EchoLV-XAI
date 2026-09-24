"""Patient-level dataset splitting for LV-XAI to prevent data leakage."""

import argparse
import random
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger("lv_xai.split_dataset")


def _extract_patient_id(filename: str) -> str:
    """Extract patient ID from filename.
    
    Assumes standard format like 'patient0001_2CH_ED.png'
    where the first segment before the underscore is the patient ID.
    Modify this if your dataset uses a different convention.
    """
    return filename.split("_")[0]


def create_splits(
    image_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> None:
    """Create train/val/test splits based on patient IDs."""
    if not image_dir.exists():
        logger.error("Image directory does not exist: %s", image_dir)
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all image files
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    all_files = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensions]
    
    if not all_files:
        logger.warning("No valid images found in %s. Cannot create splits.", image_dir)
        return

    # Group by patient ID
    patient_to_files = {}
    for path in all_files:
        # Ignore masks when listing raw images if they are mixed in, 
        # assuming mask names contain 'mask' or are in a 'masks' dir.
        if "mask" in path.name.lower() or "masks" in path.parts:
            continue
            
        patient_id = _extract_patient_id(path.name)
        if patient_id not in patient_to_files:
            patient_to_files[patient_id] = []
        patient_to_files[patient_id].append(str(path.relative_to(image_dir)))
        
    patients = sorted(list(patient_to_files.keys()))
    if not patients:
        logger.warning("No non-mask images found to split.")
        return
        
    random.shuffle(patients)
    
    num_patients = len(patients)
    train_end = int(train_ratio * num_patients)
    val_end = train_end + int(val_ratio * num_patients)
    
    train_patients = set(patients[:train_end])
    val_patients = set(patients[train_end:val_end])
    test_patients = set(patients[val_end:])
    
    # Ensure no leakage
    assert train_patients.isdisjoint(val_patients), "Leakage detected between Train and Val!"
    assert train_patients.isdisjoint(test_patients), "Leakage detected between Train and Test!"
    
    logger.info("Total patients: %d", num_patients)
    logger.info("Train patients: %d", len(train_patients))
    logger.info("Val patients: %d", len(val_patients))
    logger.info("Test patients: %d", len(test_patients))

    def write_split(split_patients: set[str], split_name: str) -> None:
        records = []
        for pid in sorted(split_patients):
            for rel_path in patient_to_files[pid]:
                records.append({"patient_id": pid, "image_path": rel_path})
        
        df = pd.DataFrame(records)
        out_path = output_dir / f"{split_name}.csv"
        df.to_csv(out_path, index=False)
        logger.info("Wrote %d records to %s", len(df), out_path)
        
    write_split(train_patients, "train")
    write_split(val_patients, "val")
    write_split(test_patients, "test")


def _main() -> int:
    parser = argparse.ArgumentParser(description="Create patient-level dataset splits")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config.values["project"].get("seed", 42))
    
    segmentation_dir = config.path("segmentation_data")
    splits_dir = config.root_dir / "data" / "splits"
    
    logger.info("Creating splits for dataset: %s", segmentation_dir)
    create_splits(segmentation_dir, splits_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
