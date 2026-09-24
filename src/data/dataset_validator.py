"""Validate dataset files without assuming a dataset-specific naming scheme."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.utils.config import ProjectConfig, load_config
from src.utils.logger import get_logger


@dataclass
class ValidationReport:
    """Structured dataset validation results."""

    root: Path
    image_count: int = 0
    mask_count: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    unreadable_files: list[str] = field(default_factory=list)
    dimension_mismatches: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_dataset(
    root: str | Path,
    supported_extensions: list[str],
    mask_extensions: list[str] | None = None,
    require_masks: bool = False,
) -> ValidationReport:
    """Validate images and optional image/mask pairs below ``root``.

    Pairing is conservative: a mask is paired with an image only when it has
    the same relative path stem under a sibling ``masks`` directory or the
    same filename stem anywhere below the root.
    """
    dataset_root = Path(root)
    report = ValidationReport(root=dataset_root)
    if not dataset_root.exists():
        report.errors.append(f"Dataset directory does not exist: {dataset_root}")
        return report
    if not dataset_root.is_dir():
        report.errors.append(f"Dataset path is not a directory: {dataset_root}")
        return report

    extensions = {item.lower() for item in supported_extensions}
    mask_suffixes = {item.lower() for item in (mask_extensions or supported_extensions)}
    files = sorted(path for path in dataset_root.rglob("*") if path.is_file())
    image_files = [path for path in files if path.suffix.lower() in extensions]
    mask_files = [path for path in files if path.suffix.lower() in mask_suffixes and "mask" in path.stem.lower()]
    report.image_count = len(image_files)
    report.mask_count = len(mask_files)
    if not image_files:
        report.warnings.append(f"No supported image files found under {dataset_root}")

    hashes: dict[str, Path] = {}
    for image_path in image_files:
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                digest = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
                if digest in hashes:
                    report.duplicates.append(f"{image_path} duplicates {hashes[digest]}")
                else:
                    hashes[digest] = image_path
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            message = f"{image_path}: {exc}"
            report.unreadable_files.append(message)
            report.errors.append(f"Unreadable image: {message}")

    if require_masks and not mask_files:
        report.errors.append(f"No mask files found under {dataset_root}")
    if mask_files:
        report.dimension_mismatches.extend(_find_dimension_mismatches(image_files, mask_files))
        report.errors.extend(f"Image/mask dimension mismatch: {item}" for item in report.dimension_mismatches)
    return report


def _find_dimension_mismatches(image_files: list[Path], mask_files: list[Path]) -> list[str]:
    masks_by_stem = {path.stem: path for path in mask_files}
    mismatches: list[str] = []
    for image_path in image_files:
        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is None:
            continue
        try:
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if image.size != mask.size:
                    mismatches.append(f"{image_path} ({image.size}) vs {mask_path} ({mask.size})")
        except (OSError, UnidentifiedImageError):
            continue
    return mismatches


def validate_configured_datasets(config: ProjectConfig) -> list[ValidationReport]:
    """Validate both configured dataset roots."""
    extensions = config.values["data"]["supported_image_extensions"]
    mask_extensions = config.values["data"].get("segmentation_mask_extensions", extensions)
    return [
        validate_dataset(config.path("segmentation_data"), extensions, mask_extensions, require_masks=True),
        validate_dataset(config.path("classification_data"), extensions),
    ]


def _main() -> int:
    parser = argparse.ArgumentParser(description="Validate configured LV-XAI datasets")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    logger = get_logger("lv_xai.dataset_validator")
    reports = validate_configured_datasets(config)
    for report in reports:
        logger.info(
            "%s: %d images, %d masks, %d warning(s), %d error(s)",
            report.root,
            report.image_count,
            report.mask_count,
            len(report.warnings),
            len(report.errors),
        )
        for warning in report.warnings:
            logger.warning(warning)
        for error in report.errors:
            logger.error(error)
    return 0 if all(report.ok for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(_main())

