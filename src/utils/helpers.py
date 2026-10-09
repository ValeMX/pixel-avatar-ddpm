# src/utils/helpers.py
import torch
import yaml
import shutil

from pathlib import Path
from typing import Any


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any] | None:
    """Retrieve project configs from YAML file and returns as a dictionary."""
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Config file in {config_path} not found.")
        return None


def load_mappings(
    mapping_path: str = "configs/attribute_mappings.yaml",
) -> dict[str, Any] | None:
    """Retrieve mapping configs from YAML file and returns as a dictionary."""
    try:
        with open(mapping_path, "r", encoding="utf-8") as file:
            mapping = yaml.safe_load(file)
        return mapping
    except FileNotFoundError:
        print(f"Error: Mapping file in {mapping_path} not found.")
        return None


def get_generator(
    seed: int = 42,
    device: torch.device | None = None,
) -> torch.Generator:
    """Create a torch.Generator with a specific seed for reproducibility.

    Args:
        seed (int): The seed value for the generator. Default is 42.
        device (torch.device | None): The device on which to create the generator.
    """
    return torch.Generator(device=device).manual_seed(seed)


def clean_directory(directory_path: str) -> None:
    """Clean the specified directory by removing all files and subdirectories."""
    directory = Path(directory_path)
    if directory.exists():
        for file in directory.iterdir():
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                for subfile in file.iterdir():
                    subfile.unlink()
                file.rmdir()
        directory.rmdir()
    directory.mkdir(parents=True, exist_ok=True)
