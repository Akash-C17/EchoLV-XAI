from pathlib import Path

import pytest
from PIL import Image

from src.data.dataset_validator import validate_dataset
from src.utils.config import ConfigurationError, load_config
from src.utils.seed import set_seed


def test_config_loads():
    config = load_config(Path(__file__).parents[1] / "config.yaml")
    assert config.values["project"]["seed"] == 42
    assert config.path("segmentation_data").name == "segmentation"


def test_invalid_config_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("project: []", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(path)


def test_dataset_validator_accepts_readable_image(tmp_path):
    image_path = tmp_path / "patient_001.png"
    Image.new("L", (16, 16), color=1).save(image_path)
    report = validate_dataset(tmp_path, [".png"])
    assert report.ok
    assert report.image_count == 1
    assert not report.unreadable_files


def test_dataset_validator_reports_missing_directory(tmp_path):
    report = validate_dataset(tmp_path / "missing", [".png"])
    assert not report.ok
    assert "does not exist" in report.errors[0]


def test_seed_rejects_negative_values():
    with pytest.raises(ValueError):
        set_seed(-1)

