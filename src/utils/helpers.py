# src/utils/helpers.py
import random
import numpy as np
import torch
import yaml

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


def set_seed(seed=42):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
