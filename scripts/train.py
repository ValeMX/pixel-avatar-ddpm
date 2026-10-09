import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
import tqdm

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import Scheduler
from src.dataset import AvatarDataset
from src.models import Diffusion
from src.models.text_encoder import Tokenizer
from src.utils.helpers import load_config, get_generator, clean_directory


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


def save_checkpoint(
    model: nn.Module, optimizer: Adam, epoch: int, checkpoint_path: str
):
    """Save the model and optimizer state to a checkpoint file.

    Args:
        model (nn.Module): The model to save.
        optimizer (Adam): The optimizer to save.
        epoch (int): The current epoch number.
        checkpoint_path (str): Path to the checkpoint file.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # Check if folder exists, if not create it
    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config()

    if config is None:
        print("Error: Configuration file not found or invalid.")
        sys.exit(1)

    seed = config["training"]["seed"]

    # 1. Build the ordinary split of the dataset
    split_id = 1
    data_path = config["dataset"]["processed_dataset_path"]
    split_path = config["dataset"]["split_dataset_path"] + f"/{split_id}"
    checkpoint_path = config["dataset"]["checkpoint_path"] + f"/{split_id}"

    print(f"Building ordinary split of the dataset at {data_path} with seed {seed}...")
    # build_ordinary_split(
    #     data_path=data_path,
    #     split_path=split_path,
    #     generator=get_generator(seed),
    # )

    # 2. Initialize the tokenizer and build the vocabulary
    tokenizer = Tokenizer()

    train_captions = get_captions_from_split(split_path, "train")
    print(f"Training captions: {len(train_captions)}")

    tokenizer.build_vocabulary(train_captions)

    vocabulary_size = tokenizer.get_vocabulary_size()
    print(f"Vocabulary size: {vocabulary_size}")
    # print(f"Tokenizer vocabulary: {tokenizer.vocabulary}")

    # 3. Initialize the diffusion model
    resnet_in_channels = config["model"]["resnet_in_channels"]
    resnet_out_channels = config["model"]["resnet_out_channels"]
    resnet_base_channels = config["model"]["resnet_base_channels"]
    resnet_spatial_resolutions = config["model"]["resnet_spatial_resolutions"]
    resnet_attention_heads = config["model"]["resnet_attention_heads"]

    text_hidden_size = config["model"]["text_hidden_size"]
    text_encoder_layers = config["model"]["text_encoder_layers"]

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
    ).to(device)

    # 4. Initialize the scheduler for the diffusion process
    beta_start = config["scheduler"]["beta_start"]
    beta_end = config["scheduler"]["beta_end"]
    training_steps = config["scheduler"]["timesteps"]

    scheduler = Scheduler(
        generator=get_generator(seed, device),
        training_steps=training_steps,
        beta_start=beta_start,
        beta_end=beta_end,
    )

    # 5. Set up the training parameters
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataset = AvatarDataset(
        data_path=split_path + "/train",
        image_extension="png",
        caption_extension="txt",
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        generator=get_generator(seed),
    )

    optimizer = Adam(diffusion.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # 6. Start the training loop
    diffusion.train()
    for epoch in tqdm.tqdm(
        range(epochs),
        desc="Training Progress",
        unit="epoch",
        dynamic_ncols=True,
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ):
        if epoch % 5 == 0 and epoch > 0:
            checkpoint_file = checkpoint_path + f"/epoch_{epoch}.pt"
            save_checkpoint(diffusion, optimizer, epoch, checkpoint_file)

        batch_bar = tqdm.tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

        for batch_idx, (images, captions) in enumerate(batch_bar):
            optimizer.zero_grad()

            t = torch.randint(0, training_steps, (batch_size,), device=device).long()
            x_noise, noise = scheduler.add_noise(images.to(device), t)
            encoded_captions = torch.tensor(
                [tokenizer.encode(caption) for caption in captions],
                dtype=torch.long,
                device=device,
            )

            predicted_noise = diffusion(x_noise, t, encoded_captions.to(device))

            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                batch_bar.set_postfix({"loss": f"{loss.item():.4f}"})
