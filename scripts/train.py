import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import yaml

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import Scheduler
from src.dataset import AvatarDataset
from src.models import Diffusion
from src.models.text_encoder import Tokenizer

from src.utils.helpers import *


def build_ordinary_split(
    data_path: str,
    split_path: str,
    generator: torch.Generator,
    data_ratio: float = 1.0,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    image_extension: str = "png",
    caption_extension: str = "txt",
):
    """Build an ordinary split of the dataset.

    Args:
        data_path (str): Path to the dataset directory containing images and captions.
        split_path (str): Path to the directory where the split data will be saved.
        generator (torch.Generator): A PyTorch random number generator for reproducibility.
        data_ratio (float, optional): Ratio of the dataset to use. Default is 1.0 (use the entire dataset).
        train_ratio (float, optional): Ratio of the training set. Default is 0.8 (80% for training).
        validation_ratio (float, optional): Ratio of the validation set. Default is 0.1 (10% for validation).
        image_extension (str, optional): Extension of image files. Default is "png".
        caption_extension (str, optional): Extension of caption files. Default is "txt".
    """
    # Get all image and caption files in the dataset directory
    image_files = list(Path(data_path).glob(f"*.{image_extension}"))
    caption_files = list(Path(data_path).glob(f"*.{caption_extension}"))

    # Shuffle the files to ensure randomness with a fixed seed for reproducibility
    image_files.sort()
    caption_files.sort()

    permutation = torch.randperm(len(image_files), generator=generator).tolist()

    image_files = [image_files[i] for i in permutation]
    caption_files = [caption_files[i] for i in permutation]

    # Split the data according to the specified ratios
    split_image_files = image_files[: int(len(image_files) * data_ratio)]
    split_caption_files = caption_files[: int(len(caption_files) * data_ratio)]

    # Further split the data into train, validation, and test sets
    train_size = int(len(split_image_files) * train_ratio)
    val_size = int(len(split_image_files) * validation_ratio)

    train_image_files = split_image_files[:train_size]
    val_image_files = split_image_files[train_size : train_size + val_size]
    test_image_files = split_image_files[train_size + val_size :]

    train_caption_files = split_caption_files[:train_size]
    val_caption_files = split_caption_files[train_size : train_size + val_size]
    test_caption_files = split_caption_files[train_size + val_size :]

    # Clean the split directory before saving new splits
    clean_directory(split_path)

    # Copy the split image and caption files to the split directory
    split_dir = Path(split_path)
    split_dir.mkdir(parents=True, exist_ok=True)

    for img in train_image_files:
        (split_dir / "train").mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / img.name).write_bytes(img.read_bytes())

    for img in val_image_files:
        (split_dir / "validation").mkdir(parents=True, exist_ok=True)
        (split_dir / "validation" / img.name).write_bytes(img.read_bytes())

    for img in test_image_files:
        (split_dir / "test").mkdir(parents=True, exist_ok=True)
        (split_dir / "test" / img.name).write_bytes(img.read_bytes())

    for cap in train_caption_files:
        (split_dir / "train").mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / cap.name).write_text(cap.read_text())

    for cap in val_caption_files:
        (split_dir / "validation").mkdir(parents=True, exist_ok=True)
        (split_dir / "validation" / cap.name).write_text(cap.read_text())

    for cap in test_caption_files:
        (split_dir / "test").mkdir(parents=True, exist_ok=True)
        (split_dir / "test" / cap.name).write_text(cap.read_text())

    print(
        f"Data split completed. ",
        f"Train: {len(train_image_files)}, ",
        f"Validation: {len(val_image_files)}, ",
        f"Test: {len(test_image_files)}\n",
        f"Split data saved in: {split_dir.resolve()}",
        sep="",
    )


def get_captions_from_split(split_path: str, split_type: str = "train") -> list[str]:
    """Retrieve captions from the specified split of the dataset.

    Args:
        split_path (str): Path to the directory containing the split data.
        split_type (str, optional): Type of split to retrieve captions from.
            Can be "train", "validation", or "test". Default is "train".

    Returns:
        list[str]: A list of captions from the specified split.
    """
    captions = []
    caption_files = list(Path(split_path).glob(f"{split_type}/*.txt"))

    for cap_file in caption_files:
        with open(cap_file, "r", encoding="utf-8") as f:
            captions.append(f.read().strip())

    return captions


def clear_checkpoints(checkpoint_path: str, keep_last: bool = True):
    """Clear all checkpoint files in the specified directory. TODO non per forza serve il penultimo

    Args:
        checkpoint_path (str): Path to the directory containing checkpoint files.
        keep_last (bool, optional): Whether to keep the last checkpoint file. Default is True.
    """
    checkpoint_dir = Path(checkpoint_path)
    if checkpoint_dir.exists() and checkpoint_dir.is_dir():
        for file in checkpoint_dir.glob("*.pt"):
            if keep_last:
                # Keep the last checkpoint
                if file == max(checkpoint_dir.glob("*.pt"), key=os.path.getctime):
                    continue
            file.unlink()


def save_model_hyperparameters(
    vocabulary_path: str,
    hyperparameters_path: str,
    tokenizer: Tokenizer,
    input_channels: int,
    output_channels: int,
    base_channels: int,
    spatial_resolutions: int,
    text_embedding_dimension: int,
    heads: int,
    text_encoder_layers: int,
):
    """Save the model hyperparameters to a file.

    Args:
        hyperparameters_path (str): Path to the file where the parameters will be saved.
        vocabulary_path (str): Path to the file where the vocabulary will be saved.
        tokenizer (Tokenizer): The tokenizer used for the model.
        input_channels (int): The number of input channels.
        output_channels (int): The number of output channels.
        base_channels (int): The base number of channels.
        spatial_resolutions (int): The spatial resolutions.
        heads (int): The number of attention heads.
        text_embedding_dimension (int): The dimension of the text embedding.
        text_encoder_layers (int): The number of layers in the text encoder.
    """
    # Check if folder exists, if not create it
    save_dir = Path(hyperparameters_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    hyperparameters = {
        "resnet_in_channels": input_channels,
        "resnet_out_channels": output_channels,
        "resnet_base_channels": base_channels,
        "resnet_spatial_resolutions": spatial_resolutions,
        "resnet_attention_heads": heads,
        "text_hidden_size": text_embedding_dimension,
        "text_encoder_layers": text_encoder_layers,
    }

    tokenizer.save_vocabulary(vocabulary_path)
    with open(hyperparameters_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(hyperparameters, file, sort_keys=False)

    print(f"Tokenizer vocabulary saved to {vocabulary_path}")
    print(f"Model hyperparameters saved to {hyperparameters_path}")


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 200,
    ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Get a learning rate scheduler that implements a warmup followed by a cosine decay.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        total_steps (int): Total number of training steps.
        warmup_steps (int, optional): Number of steps to linearly increase the learning rate. Default is 200.
        ratio (float, optional): Minimum learning rate as a fraction of the initial learning rate. Default is 0.05.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: A learning rate scheduler that can be used with the optimizer.
    """

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return ratio + (1 - ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def train(
    epochs: int,
    training_steps: int,
    device: torch.device,
    generator: torch.Generator,
    diffusion: Diffusion,
    scheduler: Scheduler,
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    optimizer: AdamW,
    lr_scheduler: LambdaLR,
    criterion: nn.Module,
    start_epoch: int = 0,
    start_loss: float = float("inf"),
    validation_interval: int = 10,
    patience: int = 10,
    patience_delta: float = 0.0001,
    dropout: float = 0.1,
    checkpoint_path: str = "./checkpoints",
) -> dict[int, float]:
    """Train the diffusion model.

    Args:
        epochs (int): Number of training epochs.
        training_steps (int): Number of training steps for the diffusion process.
        device (torch.device): Device to run the training on (CPU or GPU).
        generator (torch.Generator): A PyTorch random number generator for reproducibility.
        diffusion (Diffusion): The diffusion model to train.
        scheduler (Scheduler): The scheduler for the diffusion process.
        dataloader_train (DataLoader): DataLoader for the training dataset.
        dataloader_val (DataLoader): DataLoader for the validation dataset.
        optimizer (AdamW): Optimizer for updating model parameters.
        lr_scheduler (LambdaLR): The learning rate scheduler.
        criterion (nn.Module): Loss function to compute the loss.
        start_epoch (int): The epoch to start training from. Default is 0.
        validation_interval (int): The interval (in epochs) at which to perform validation. Default is 10.
        patience (int): Number of validation checks with no improvement before stopping training. Default is 3.
        patience_delta (float): Minimum change in validation loss to qualify as an improvement. Default is 0.01.
        checkpoint_path (str): The path to the directory where checkpoints will be saved. Default is "./checkpoints".
    Returns:
        dict[int, float]: A dictionary containing the validation loss for each validation step.
    """
    best_loss_val = start_loss
    loss_val = float("inf")
    losses = {}

    patience_counter = 0

    diffusion.train()
    epoch_bar = tqdm.tqdm(
        range(start_epoch, epochs),
        desc="Training Progress",
        unit="epoch",
        dynamic_ncols=True,
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
    )
    for epoch in epoch_bar:
        batch_bar = tqdm.tqdm(
            dataloader_train,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

        for batch_idx, (images, captions) in enumerate(batch_bar):
            optimizer.zero_grad()

            current_batch_size = images.size(0)
            t = torch.randint(
                0,
                training_steps,
                (current_batch_size,),
                device=device,
            ).long()

            x_noise, noise = scheduler.add_noise(
                images.to(device),
                t,
            )

            captions_train = captions
            if dropout > 0.0:
                captions_train = [
                    captions[i] if torch.rand(1).item() > dropout else ""
                    for i in range(current_batch_size)
                ]

            encoded_captions = torch.tensor(
                [tokenizer.encode(caption) for caption in captions_train],
                dtype=torch.long,
                device=device,
            )

            predicted_noise = diffusion(x_noise, t, encoded_captions.to(device))

            loss = criterion(predicted_noise, noise)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            if batch_idx % 10 == 0:
                epoch_bar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "loss_val": f"{loss_val:.4f}"}
                )

        # Validation step
        if (epoch % validation_interval == 0 and epoch > 0) or epoch == epochs - 1:
            loss_val = evaluate(
                diffusion=diffusion,
                scheduler=scheduler,
                dataloader=dataloader_val,
                tokenizer=tokenizer,
                criterion=criterion,
                device=device,
                training_steps=training_steps,
                generator=generator,
            )

            losses[epoch] = loss_val

            save_checkpoint(
                checkpoint_path + "/last.pt",
                diffusion,
                optimizer,
                lr_scheduler,
                epoch,
                loss_val,
            )

            if loss_val < best_loss_val * (1 - patience_delta):
                best_loss_val = loss_val
                patience_counter = 0
                save_checkpoint(
                    checkpoint_path + "/best.pt",
                    diffusion,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    loss_val,
                )
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

    return losses


@torch.no_grad()
def evaluate(
    diffusion: Diffusion,
    scheduler: Scheduler,
    dataloader: DataLoader,
    tokenizer: Tokenizer,
    criterion: nn.Module,
    device: torch.device,
    training_steps: int,
    generator: torch.Generator,
    n_t: int = 10,
):
    """Evaluate the diffusion model on the validation dataset.

    Args:
        diffusion (Diffusion): The diffusion model to evaluate.
        scheduler (Scheduler): The scheduler for the diffusion process.
        dataloader (DataLoader): DataLoader for the validation dataset.
        tokenizer (Tokenizer): The tokenizer used for encoding captions.
        criterion (nn.Module): Loss function to compute the loss.
        device (torch.device): Device to run the evaluation on (CPU or GPU).
        training_steps (int): Number of training steps for the diffusion process.
        generator (torch.Generator): A PyTorch random number generator for reproducibility.
        n_t (int, optional): Number of timesteps to evaluate. Default is 10.

    Returns:
        float: The average loss over the validation dataset.
    """
    diffusion.eval()

    # Create a tensor of timesteps to evaluate
    ts = torch.linspace(0, training_steps - 1, n_t).long()

    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    total, count = 0.0, 0

    for images, captions in dataloader:
        images = images.to(device)

        tokens = torch.tensor(
            [tokenizer.encode(c) for c in captions],
            dtype=torch.long,
            device=device,
        )

        for t_value in ts:
            t = torch.full(
                (images.size(0),), int(t_value), device=device, dtype=torch.long
            )

            noise = torch.randn(images.shape, generator=generator, device=device)
            ab = alphas_cumprod[t].view(-1, 1, 1, 1)
            x_t = ab.sqrt() * images + (1 - ab).sqrt() * noise
            total += criterion(diffusion(x_t, t, tokens), noise).item() * images.size(0)
            count += images.size(0)

    diffusion.train()
    return total / count


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config()

    if config is None:
        print("Error: Configuration file not found or invalid.")
        sys.exit(1)

    seed = config["training"]["seed"]

    # 1. Choose an available split of the dataset or create a new one
    data_path = config["dataset"]["processed_dataset_path"]
    split_path = config["dataset"]["split_dataset_path"]
    checkpoint_path = config["dataset"]["checkpoint_path"]

    available_splits = get_splits(split_path)

    print(f"Available splits: {available_splits}")
    print(
        "Choose a split to use for training (or type 'new' to create a new split)> ",
        end="",
    )
    chosen_split = input()

    if chosen_split == "new":
        print(f"Enter a name for the new split: ")
        split_id = input().strip()
        split_path = split_path + f"/{split_id}"

        # Create a new split from the dataset
        print(
            f"Building ordinary split of the dataset at {data_path} with seed {seed}..."
        )
        build_ordinary_split(
            data_path=data_path,
            split_path=split_path,
            generator=get_generator(seed),
        )
    else:
        if chosen_split in available_splits:
            print(f"Using existing split: {chosen_split}")
            split_id = chosen_split
            split_path = split_path + f"/{split_id}"
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)

    # 2. Choose an available tokenizer vocabulary and model or create new ones
    available_checkpoints = get_models(checkpoint_path)
    print(f"Available checkpoints: {available_checkpoints}")
    print(
        "Choose a checkpoint to use for tokenizer vocabulary and model (or type 'new' to create a new ones)> ",
        end="",
    )
    chosen_checkpoint = input()

    tokenizer = Tokenizer()
    hyperparameters = None

    if chosen_checkpoint == "new":
        print("Enter a name for the new checkpoint: ")
        checkpoint_id = input().strip()
        checkpoint_path = checkpoint_path + f"/{checkpoint_id}"

        print(
            "Creating a new tokenizer vocabulary and model using hyperparameters from the config file..."
        )
        train_captions = get_captions_from_split(split_path, "train")
        tokenizer.build_vocabulary(train_captions)
        hyperparameters = config["model"]
    else:
        if chosen_checkpoint in available_checkpoints:
            print(f"Using existing checkpoint: {chosen_checkpoint}")
            checkpoint_path = checkpoint_path + f"/{chosen_checkpoint}"

            # Load the tokenizer vocabulary from the checkpoint
            tokenizer.load_vocabulary(checkpoint_path + "/tokenizer_vocabulary.csv")

            # Load the model hyperparameters from the checkpoint
            hyperparameters = load_hyperparameters(
                checkpoint_path + "/model_hyperparameters.yaml"
            )

        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)

    if hyperparameters is None:
        print("Error: Model hyperparameters not found. Exiting.")
        sys.exit(1)

    # 3. Initialize the diffusion model
    vocabulary_size = tokenizer.get_vocabulary_size()

    resnet_in_channels = hyperparameters["resnet_in_channels"]
    resnet_out_channels = hyperparameters["resnet_out_channels"]
    resnet_base_channels = hyperparameters["resnet_base_channels"]
    resnet_spatial_resolutions = hyperparameters["resnet_spatial_resolutions"]
    resnet_attention_heads = hyperparameters["resnet_attention_heads"]

    text_hidden_size = hyperparameters["text_hidden_size"]
    text_encoder_layers = hyperparameters["text_encoder_layers"]

    diffusion = Diffusion(
        input_channels=resnet_in_channels,
        output_channels=resnet_out_channels,
        base_channels=resnet_base_channels,
        spatial_resolutions=resnet_spatial_resolutions,
        time_embedding_dimension=resnet_base_channels,
        text_embedding_dimension=text_hidden_size,
        heads=resnet_attention_heads,
        vocabulary_size=vocabulary_size,
        text_encoder_layers=text_encoder_layers,
        dropout=0,
    ).to(device)

    # 4. Initialize the scheduler for the diffusion process
    training_steps = config["scheduler"]["timesteps"]
    beta_start = config["scheduler"]["beta_start"]
    beta_end = config["scheduler"]["beta_end"]
    beta_max = config["scheduler"]["beta_max"]
    s = config["scheduler"]["s"]
    type = config["scheduler"]["type"]

    scheduler = Scheduler(
        generator=get_generator(seed, device),
        training_steps=training_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_max=beta_max,
        type=type,
        s=s,
    )

    # 5. Set up the training parameters
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    dropout = config["training"]["dropout"]
    validation_interval = config["training"]["validation_interval"]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataset_train = AvatarDataset(
        data_path=split_path + "/train",
        image_extension="png",
        caption_extension="txt",
        transform=transform,
    )

    dataset_val = AvatarDataset(
        data_path=split_path + "/validation",
        image_extension="png",
        caption_extension="txt",
        transform=transform,
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        generator=get_generator(seed),
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        generator=get_generator(seed),
    )

    optimizer = AdamW(diffusion.parameters(), lr=learning_rate)
    steps = epochs * len(dataloader_train)
    lr_scheduler = get_lr_scheduler(optimizer, total_steps=steps, warmup_steps=10)
    criterion = nn.MSELoss()

    # 6. Save the model hyperparameters and tokenizer vocabulary for reproducibility
    save_model_hyperparameters(
        vocabulary_path=checkpoint_path + f"/tokenizer_vocabulary.csv",
        hyperparameters_path=checkpoint_path + f"/model_hyperparameters.yaml",
        tokenizer=tokenizer,
        input_channels=resnet_in_channels,
        output_channels=resnet_out_channels,
        base_channels=resnet_base_channels,
        spatial_resolutions=resnet_spatial_resolutions,
        text_embedding_dimension=text_hidden_size,
        heads=resnet_attention_heads,
        text_encoder_layers=text_encoder_layers,
    )

    # 7. Choose an available checkpoint to resume training or start from scratch
    available_checkpoints = get_checkpoints(checkpoint_path)
    print(f"Available checkpoints: {available_checkpoints}")
    print(
        "Choose a checkpoint to resume training (or type 'new' to start from scratch)> ",
        end="",
    )
    chosen_checkpoint = input()

    start_epoch = 0
    loss_val = float("inf")

    if chosen_checkpoint == "new":
        print("Starting training from scratch...")
    else:
        if chosen_checkpoint in available_checkpoints:
            print(f"Resuming training from checkpoint: {chosen_checkpoint}")
            checkpoint_file = checkpoint_path + f"/{chosen_checkpoint}"
            checkpoint = load_checkpoint(checkpoint_file, device)
            diffusion.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            loss_val = checkpoint.get("loss_val", float("inf"))
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)

    losses = train(
        start_epoch=start_epoch,
        epochs=epochs,
        training_steps=training_steps,
        device=device,
        generator=get_generator(seed, device),
        diffusion=diffusion,
        scheduler=scheduler,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        start_loss=loss_val,
        validation_interval=validation_interval,
        dropout=dropout,
        checkpoint_path=checkpoint_path,
    )

    print("Training completed. Validation losses per epoch:")
    for epoch, loss in losses.items():
        print(f"Epoch {epoch}: Validation Loss = {loss:.4f}")
