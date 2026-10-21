import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import yaml

from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import Scheduler
from src.dataset import AvatarDataset
from src.models import Diffusion
from src.models.text_encoder import Tokenizer
from src.utils.helpers import *
from src.utils.splits import *


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


def save_model_hyperparameters(
    vocabulary_path: str,
    hyperparameters_path: str,
    tokenizer: Tokenizer,
    image_size: int,
    input_channels: int,
    output_channels: int,
    base_channels: int,
    spatial_resolutions: int,
    heads: int,
    time_embedding_multiplier: int,
    text_embedding_dimension: int,
    text_encoder_layers: int,
    dropout: float,
    timesteps: int,
    beta_start: float,
    beta_end: float,
    beta_max: float,
    s: float,
    scheduler_type: str,
):
    """Save the model hyperparameters to a file.

    Args:
        hyperparameters_path (str): Path to the file where the parameters will be saved.
        vocabulary_path (str): Path to the file where the vocabulary will be saved.
        tokenizer (Tokenizer): The tokenizer used for the model.
        image_size (int): The size of the input images.
        input_channels (int): The number of input channels.
        output_channels (int): The number of output channels.
        base_channels (int): The base number of channels.
        spatial_resolutions (int): The spatial resolutions.
        heads (int): The number of attention heads.
        time_embedding_multiplier (int): The multiplier for the time embedding dimension.
        text_embedding_dimension (int): The dimension of the text embedding.
        text_encoder_layers (int): The number of layers in the text encoder.
        dropout (float): The dropout rate.
        timesteps (int): The number of timesteps for the diffusion process.
        beta_start (float): The starting value of beta for the diffusion process.
        beta_end (float): The ending value of beta for the diffusion process.
        beta_max (float): The maximum value of beta for the diffusion process.
        s (float): The s parameter for the diffusion process.
        scheduler_type (str): The type of scheduler used for the diffusion process.
    """
    # Check if folder exists, if not create it
    save_dir = Path(hyperparameters_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    hyperparameters = {
        "model": {
            "image_size": image_size,
            "resnet_in_channels": input_channels,
            "resnet_out_channels": output_channels,
            "resnet_base_channels": base_channels,
            "resnet_spatial_resolutions": spatial_resolutions,
            "resnet_attention_heads": heads,
            "time_embedding_multiplier": time_embedding_multiplier,
            "text_encoder_layers": text_encoder_layers,
            "text_hidden_size": text_embedding_dimension,
            "dropout": dropout,
        },
        "scheduler": {
            "timesteps": timesteps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "beta_max": beta_max,
            "s": s,
            "type": f"{scheduler_type}",
        },
    }

    tokenizer.save_vocabulary(vocabulary_path)
    with open(hyperparameters_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(hyperparameters, file, sort_keys=False)

    print(f"Tokenizer vocabulary saved to {vocabulary_path}")
    print(f"Model hyperparameters saved to {hyperparameters_path}")


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 100,
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
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
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
        dropout (float): Dropout rate for the captions. Default is 0.1.
        checkpoint_path (str): The path to the directory where checkpoints will be saved. Default is "./checkpoints".
    Returns:
        dict[int, float]: A dictionary containing the validation loss for each validation step.
    """
    diffusion.train()

    best_loss_val = loss_val = start_loss
    losses = {}
    patience_counter = 0

    epoch_bar = tqdm.tqdm(
        range(start_epoch, epochs),
        initial=start_epoch,
        total=epochs,
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
        if epoch % validation_interval == 0 or epoch == epochs - 1:
            save_checkpoint(
                checkpoint_path + "/last.pt",
                diffusion,
                optimizer,
                lr_scheduler,
                epoch,
                loss_val,
            )

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

    val_bar = tqdm.tqdm(
        dataloader,
        desc="Validation Progress",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )
    for images, captions in val_bar:
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

    mappings = load_mappings()

    if mappings is None:
        print("Error: Mapping file not found or invalid.")
        sys.exit(1)

    seed = config["training"]["seed"]

    # 1. Choose an available split of the dataset or create a new one
    data_path = config["dataset"]["processed_dataset_path"]
    split_path = config["dataset"]["split_dataset_path"]
    checkpoint_path = config["dataset"]["checkpoint_path"]

    available_splits = get_splits(split_path)

    print(f"Available splits:")
    for i, split in enumerate(available_splits, start=1):
        print(f"  {i}: {split}")
    chosen_split = input(
        "Select a split by number (or type 'new' to create a new split): "
    ).strip()

    if chosen_split == "new":
        split_type = input(
            "Choose the type of the new split (Compositional (C) or Ordinary (O)): "
        ).strip()

        if split_type.lower() not in ["c", "o"]:
            print("Invalid choice. Exiting.")
            sys.exit(1)

        image_size = input(
            f"Enter the image size for the new split (32 or 64): "
        ).strip()

        if image_size not in ["32", "64"]:
            print("Invalid image size. Exiting.")
            sys.exit(1)

        image_size = int(image_size)

        split_id = input("Enter a name for the new split: ").strip()
        split_path = split_path + f"/{split_id}"

        # Create a new split from the dataset
        if split_type.lower() == "o":
            print(
                f"Building ordinary split of the dataset at {data_path} with seed {seed}..."
            )
            build_ordinary_split(
                data_path=data_path + str(image_size),
                split_path=split_path,
                generator=get_generator(seed),
            )
        else:
            print(
                f"Building compositional split of the dataset at {data_path} with seed {seed}..."
            )
            build_compositional_split(
                data_path=data_path + str(image_size),
                split_path=split_path,
                mappings=mappings,
                combos=config["dataset"]["held_out_combinations"],
                seed=seed,
            )
    else:
        try:
            chosen_split_index = int(chosen_split) - 1
            if 0 <= chosen_split_index < len(available_splits):
                split_path = split_path + f"/{available_splits[chosen_split_index]}"
                print(f"Using existing split: {available_splits[chosen_split_index]}")
            else:
                raise ValueError("Invalid choice. Exiting.")
        except ValueError:
            print("Invalid choice. Exiting.")
            sys.exit(1)

    # 2. Choose an available tokenizer vocabulary and model or create new ones
    available_checkpoints = get_models(checkpoint_path)
    print(f"Available checkpoints:")
    for i, checkpoint in enumerate(available_checkpoints, start=1):
        print(f"  {i}: {checkpoint}")

    chosen_checkpoint = input(
        "Choose a checkpoint to use for tokenizer vocabulary and model (or type 'new' to create a new ones): "
    ).strip()

    tokenizer = Tokenizer()
    hyperparameters = None

    if chosen_checkpoint == "new":
        checkpoint_id = input("Enter a name for the new checkpoint: ").strip()
        checkpoint_path = checkpoint_path + f"/{checkpoint_id}"

        print(
            "Creating a new tokenizer vocabulary and model using hyperparameters from the config file..."
        )
        train_captions = get_captions_from_split(split_path, "train")
        tokenizer.build_vocabulary(train_captions)
        hyperparameters = config
    else:
        try:
            chosen_checkpoint_index = int(chosen_checkpoint) - 1
            if 0 <= chosen_checkpoint_index < len(available_checkpoints):
                checkpoint_path = (
                    checkpoint_path
                    + f"/{available_checkpoints[chosen_checkpoint_index]}"
                )
                print(
                    f"Using existing checkpoint: {available_checkpoints[chosen_checkpoint_index]}"
                )
                tokenizer.load_vocabulary(checkpoint_path + "/tokenizer_vocabulary.csv")
                hyperparameters = load_hyperparameters(
                    checkpoint_path + "/model_hyperparameters.yaml"
                )
            else:
                print("Invalid choice. Exiting.")
                sys.exit(1)
        except ValueError:
            print("Invalid choice. Exiting.")
            sys.exit(1)

    if hyperparameters is None:
        print("Error: Model hyperparameters not found. Exiting.")
        sys.exit(1)

    # 3. Initialize the diffusion model
    vocabulary_size = tokenizer.get_vocabulary_size()

    model_hyperparameters = hyperparameters["model"]
    scheduler_hyperparameters = hyperparameters["scheduler"]

    image_size = model_hyperparameters["image_size"]
    resnet_in_channels = model_hyperparameters["resnet_in_channels"]
    resnet_out_channels = model_hyperparameters["resnet_out_channels"]
    resnet_base_channels = model_hyperparameters["resnet_base_channels"]
    resnet_spatial_resolutions = model_hyperparameters["resnet_spatial_resolutions"]
    resnet_attention_heads = model_hyperparameters["resnet_attention_heads"]

    time_embedding_multiplier = model_hyperparameters["time_embedding_multiplier"]
    time_embedding_dimension = resnet_base_channels * time_embedding_multiplier

    text_hidden_size = model_hyperparameters["text_hidden_size"]
    text_encoder_layers = model_hyperparameters["text_encoder_layers"]
    model_dropout = model_hyperparameters["dropout"]

    diffusion = Diffusion(
        input_channels=resnet_in_channels,
        output_channels=resnet_out_channels,
        base_channels=resnet_base_channels,
        spatial_resolutions=resnet_spatial_resolutions,
        time_embedding_dimension=time_embedding_dimension,
        text_embedding_dimension=text_hidden_size,
        heads=resnet_attention_heads,
        vocabulary_size=vocabulary_size,
        text_encoder_layers=text_encoder_layers,
        dropout=model_dropout,
    ).to(device)

    # 4. Initialize the scheduler for the diffusion process
    training_steps = scheduler_hyperparameters["timesteps"]
    beta_start = scheduler_hyperparameters["beta_start"]
    beta_end = scheduler_hyperparameters["beta_end"]
    beta_max = scheduler_hyperparameters["beta_max"]
    s = scheduler_hyperparameters["s"]
    type = scheduler_hyperparameters["type"]

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
    weight_decay = config["training"]["weight_decay"]
    training_dropout = config["training"]["dropout"]
    validation_interval = config["training"]["validation_interval"]
    patience = config["training"]["patience"]

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
        shuffle=True,
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        generator=get_generator(seed),
        shuffle=False,
    )

    optimizer = AdamW(
        diffusion.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    steps = epochs * len(dataloader_train)
    lr_scheduler = get_lr_scheduler(optimizer, steps)
    criterion = nn.MSELoss()

    # 6. Save the model hyperparameters and tokenizer vocabulary for reproducibility
    save_model_hyperparameters(
        vocabulary_path=checkpoint_path + f"/tokenizer_vocabulary.csv",
        hyperparameters_path=checkpoint_path + f"/model_hyperparameters.yaml",
        tokenizer=tokenizer,
        image_size=image_size,
        input_channels=resnet_in_channels,
        output_channels=resnet_out_channels,
        base_channels=resnet_base_channels,
        spatial_resolutions=resnet_spatial_resolutions,
        time_embedding_multiplier=time_embedding_multiplier,
        text_embedding_dimension=text_hidden_size,
        heads=resnet_attention_heads,
        text_encoder_layers=text_encoder_layers,
        dropout=model_dropout,
        timesteps=training_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_max=beta_max,
        s=s,
        scheduler_type=type,
    )

    # 7. Choose an available checkpoint to resume training or start from scratch
    available_checkpoints = get_checkpoints(checkpoint_path)
    print(f"Available checkpoints:")
    for i, checkpoint in enumerate(available_checkpoints, start=1):
        print(f"  {i}: {checkpoint}")
    chosen_checkpoint = input(
        "Choose a checkpoint to resume training (or type 'new' to start from scratch): "
    ).strip()

    start_epoch = 0
    loss_val = float("inf")

    if chosen_checkpoint == "new":
        print("Starting training from scratch...")
    else:
        try:
            chosen_checkpoint_index = int(chosen_checkpoint) - 1
            if 0 <= chosen_checkpoint_index < len(available_checkpoints):
                print(
                    f"Resuming training from checkpoint: {available_checkpoints[chosen_checkpoint_index]}"
                )
                checkpoint_file = (
                    checkpoint_path
                    + f"/{available_checkpoints[chosen_checkpoint_index]}"
                )
                checkpoint = load_checkpoint(checkpoint_file, device)
                diffusion.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                loss_val = checkpoint["loss_val"]
            else:
                print("Invalid choice. Exiting.")
                sys.exit(1)
        except ValueError:
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
        patience=patience,
        dropout=training_dropout,
        checkpoint_path=checkpoint_path,
    )

    print("Training completed. Validation losses per epoch:")
    for epoch, loss in losses.items():
        print(f"Epoch {epoch}: Validation Loss = {loss:.4f}")
