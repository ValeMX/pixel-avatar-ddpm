import torch
import torch.nn as nn
import yaml

from pathlib import Path
from torch.optim import AdamW
from typing import Any


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any] | None:
    """Retrieve project configs from YAML file and returns as a dictionary.

    Args:
        config_path (str): Path to the configuration YAML file. Default is "configs/config.yaml".

    Returns:
        dict[str, Any] | None: A dictionary containing the configuration parameters if the file is found and successfully loaded, otherwise None.
    """
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
    """Retrieve mapping configs from YAML file and returns as a dictionary.

    Args:
        mapping_path (str): Path to the mapping YAML file. Default is "configs/attribute_mappings.yaml".

    Returns:
        dict[str, Any] | None: A dictionary containing the mapping parameters if the file is found and successfully loaded, otherwise None.
    """
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
        device (torch.device, optional): The device on which to create the generator. Default is None.

    Returns:
        torch.Generator: A torch.Generator object initialized with the specified seed.
    """
    return torch.Generator(device=device).manual_seed(seed)


def get_splits(
    split_path: str = "data/splits",
    train_path: str = "/train",
    val_path: str = "/validation",
    test_path: str = "/test",
) -> list[str]:
    """Retrieve the available splits from the specified directory.

    A split is considered available if it contains the required subdirectories for training, validation, and testing.

    Args:
        split_path (str): Path to the directory containing splits.
        train_path (str): Relative path to the training split directory.
        val_path (str): Relative path to the validation split directory.
        test_path (str): Relative path to the test split directory.

    Returns:
        list[str]: A list of available split names.
    """

    split_dir = Path(split_path)
    if not split_dir.exists():
        print(f"Error: Split directory in {split_path} not found.")
        return []

    required_subdirectories = (
        Path(train_path).name,
        Path(val_path).name,
        Path(test_path).name,
    )

    splits = [
        split.name
        for split in split_dir.iterdir()
        if split.is_dir()
        and all(
            subdir in [sub.name for sub in split.iterdir()]
            for subdir in required_subdirectories
        )
    ]

    return sorted(splits)


def get_models(
    model_path: str = "checkpoints",
    vocabulary_file: str = "tokenizer_vocabulary.csv",
    hyperparameters_file: str = "model_hyperparameters.yaml",
) -> list[str]:
    """Retrieve the available models from the specified directory.

    Args:
        model_path (str): Path to the directory containing models.
        vocabulary_file (str): Name of the vocabulary file to check for existence.
        hyperparameters_file (str): Name of the hyperparameters file to check for existence.

    Returns:
        list[str]: A list of available model names.
    """

    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"Error: Model directory in {model_path} not found.")
        return []

    models = [
        model.name
        for model in model_dir.iterdir()
        if model.is_dir()
        and (model / vocabulary_file).exists()
        and (model / hyperparameters_file).exists()
    ]

    return sorted(models)


def get_checkpoints(
    checkpoint_path: str = "checkpoints",
    checkpoint_extension: str = ".pt",
) -> list[str]:
    """Retrieve the available checkpoints from the specified directory.

    Args:
        checkpoint_path (str): Path to the directory containing checkpoints.
        checkpoint_extension (str): The file extension of the checkpoint files.

    Returns:
        list[str]: A list of available checkpoint file names.
    """
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory in {checkpoint_path} not found.")
        return []

    checkpoints = [
        checkpoint.name
        for checkpoint in checkpoint_dir.iterdir()
        if checkpoint.is_file() and checkpoint.suffix == checkpoint_extension
    ]

    return sorted(checkpoints)


def load_hyperparameters(hyperparameters_path: str) -> dict[str, Any] | None:
    """Load hyperparameters from a YAML file and return as a dictionary.

    Args:
        hyperparameters_path (str): Path to the YAML file containing hyperparameters.

    Returns:
        dict[str, Any] | None: A dictionary of hyperparameters, or None if the file is not found.
    """
    try:
        with open(hyperparameters_path, "r", encoding="utf-8") as file:
            hyperparameters = yaml.safe_load(file)
        return hyperparameters
    except FileNotFoundError:
        print(f"Error: Hyperparameters file in {hyperparameters_path} not found.")
        return None


def save_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: AdamW,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    loss_val: float,
) -> None:
    """Save the model, optimizer, and learning rate scheduler state to a checkpoint file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (nn.Module): The model to save.
        optimizer (AdamW): The optimizer to save.
        lr_scheduler (torch.optim.lr_scheduler.LambdaLR): The learning rate scheduler to save.
        epoch (int): The current epoch number.
        loss_val (float): The validation loss at the current epoch.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "loss_val": loss_val,
    }

    # Check if folder exists, if not create it
    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint from a file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): The device to map the checkpoint to.

    Returns:
        dict[str, Any]: A dictionary containing the checkpoint data.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def scale(
    tensor: torch.Tensor, source_range: tuple, target_range: tuple, clamp: bool = False
) -> torch.Tensor:
    """Rescale a tensor from a source range to a target range.

    Args:
        tensor (torch.Tensor): The input tensor to be rescaled.
        source_range (tuple): A tuple (min, max) representing the source range of the tensor values.
        target_range (tuple): A tuple (min, max) representing the target range for the rescaled values.
        clamp (bool): If True, clamp the output values to the target range. Default is False.

    Returns:
        torch.Tensor: The rescaled tensor.
    """
    tensor = (tensor - source_range[0]) / (source_range[1] - source_range[0])
    tensor = tensor * (target_range[1] - target_range[0]) + target_range[0]

    if clamp:
        tensor = tensor.clamp(target_range[0], target_range[1])

    return tensor
