"""Configuration loading and validation for the LV-XAI project."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when the project configuration is invalid."""


@dataclass(frozen=True)
class ProjectConfig:
    """Validated configuration with paths resolved relative to the config file."""

    values: dict[str, Any]
    root_dir: Path

    def path(self, key: str) -> Path:
        """Return a configured path resolved relative to the project root."""
        try:
            configured = self.values["paths"][key]
        except KeyError as exc:
            raise ConfigurationError(f"Missing paths.{key} in configuration") from exc
        path = Path(configured)
        return path if path.is_absolute() else self.root_dir / path


def load_config(path: str | Path = "config.yaml") -> ProjectConfig:
    """Load and validate a YAML configuration file."""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(values, dict):
        raise ConfigurationError("Configuration root must be a mapping")
    _validate_config(values)
    return ProjectConfig(values=values, root_dir=config_path.parent)


def _validate_config(values: dict[str, Any]) -> None:
    required_sections = {"project", "data", "segmentation", "classification", "training", "xai", "paths"}
    missing = required_sections.difference(values)
    if missing:
        raise ConfigurationError(f"Missing configuration sections: {', '.join(sorted(missing))}")
    if not isinstance(values["project"].get("seed"), int):
        raise ConfigurationError("project.seed must be an integer")
    for key in ("image_size", "segmentation_image_size", "num_classes"):
        value = values["data"].get(key)
        if not isinstance(value, int) or value <= 0:
            raise ConfigurationError(f"data.{key} must be a positive integer")
    for key in ("segmentation_data", "classification_data"):
        if not isinstance(values["paths"].get(key), str) or not values["paths"][key]:
            raise ConfigurationError(f"paths.{key} must be a non-empty string")
    extensions = values["data"].get("supported_image_extensions")
    if not isinstance(extensions, list) or not extensions or not all(
        isinstance(item, str) and item.startswith(".") for item in extensions
    ):
        raise ConfigurationError("data.supported_image_extensions must contain file extensions")


def _main() -> int:
    parser = argparse.ArgumentParser(description="Validate an LV-XAI configuration")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"Valid configuration: {Path(args.config).resolve()}")
    print(f"Project root: {config.root_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

