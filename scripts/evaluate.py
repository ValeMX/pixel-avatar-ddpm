import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import argparse
import time
import csv
import itertools
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from src.dataset import Scheduler, AvatarDataset
from src.models import Diffusion
from src.models.text_encoder import Tokenizer
from src.utils.helpers import *


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count the number and the size of trainable parameters in the model.

    Args:
        model (nn.Module): The PyTorch model to count parameters for.

    Returns:
        int: The total number of trainable parameters in the model.
        int: The total size of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    )


@torch.no_grad()
def generate_batch(
    diffusion: Diffusion,
    scheduler: Scheduler,
    tokenizer: Tokenizer,
    prompts: list[str],
    generator: torch.Generator,
    channels: int = 3,
    output_size: int = 32,
    inference_steps: int = 250,
    cfg_scale: float = 3.5,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> torch.Tensor:
    """Generates a batch of images from the given prompts using the diffusion model.

    Args:
        diffusion (Diffusion): The diffusion model to use for generation.
        scheduler (Scheduler): The scheduler for the diffusion process.
        tokenizer (Tokenizer): The tokenizer for encoding prompts.
        prompts (list[str]): A list of text prompts to generate images from.
        generator (torch.Generator): A random number generator for reproducibility.
        channels (int, optional): Number of channels in the output images. Defaults to 3.
        output_size (int, optional): Size of the output images. Defaults to 32.
        inference_steps (int, optional): Number of denoising steps. Defaults to 250.
        cfg_scale (float, optional): Classifier-Free Guidance scale. Defaults to 3.5.
        device (torch.device, optional): Device to run the generation on. Defaults to CUDA if available.

    Returns:
        torch.Tensor: A tensor containing the generated images in uint8 format, scaled to [0, 255].
    """
    diffusion.eval()
    batch_size = len(prompts)
    cfg = cfg_scale > 0

    if cfg:
        tokens_cond = torch.tensor(
            [tokenizer.encode(p) for p in prompts], dtype=torch.long, device=device
        )
        tokens_uncond = torch.tensor(
            [tokenizer.encode("") for _ in prompts], dtype=torch.long, device=device
        )
        encoded_captions = torch.cat([tokens_cond, tokens_uncond], dim=0)
    else:
        encoded_captions = torch.tensor(
            [tokenizer.encode(p) for p in prompts], dtype=torch.long, device=device
        )

    scheduler.set_inference_timesteps(inference_steps)
    output_shape = (batch_size, channels, output_size, output_size)
    model_output = torch.randn(output_shape, generator=generator, device=device)

    timesteps_bar = tqdm(
        scheduler.timesteps,
        desc="Generating images",
        unit="step",
        leave=False,
    )
    for timestep in timesteps_bar:
        model_input = model_output
        if cfg:
            model_input = model_input.repeat(2, 1, 1, 1)
            model_timestep = torch.tensor([timestep] * (batch_size * 2), device=device)
        else:
            model_timestep = torch.tensor([timestep] * batch_size, device=device)

        output = diffusion(model_input, model_timestep, encoded_captions)

        if cfg:
            output_cond, output_uncond = output.chunk(2)
            output = cfg_scale * (output_cond - output_uncond) + output_uncond

        model_output = scheduler.remove_noise(int(timestep), model_output, output)

    # Scale the output images to [0, 255] and convert to uint8
    images = scale(model_output, (-1, 1), (0, 255), clamp=True)
    return images.to(torch.uint8)


# Salvare 4 immagini generate per visualizzare la diversità
def evaluate_diversity(
    diffusion: Diffusion,
    scheduler: Scheduler,
    tokenizer: Tokenizer,
    prompt: str,
    seed=42,
    num_seeds: int = 5,
    channels: int = 3,
    output_size: int = 32,
    inference_steps: int = 250,
    cfg_scale: float = 3.5,
    device: torch.device = torch.device("cuda"),
    output_dir: str = "diversity_samples",
) -> float:
    """Evaluate the diversity of generated images for a given prompt using different random seeds.

    Args:
        diffusion: The diffusion model.
        scheduler: The scheduler.
        tokenizer: The tokenizer.
        prompt: The prompt to generate images for.
        seed: The initial random seed. Default is 42.
        num_seeds: The number of different random seeds to use. Default is 5.
        channels: The number of channels in the generated images. Default is 3.
        output_size: The size of the generated images. Default is 32.
        inference_steps: The number of inference steps. Default is 250.
        cfg_scale: The scale for classifier-free guidance. Default is 3.5.
        device: The device to run the evaluation on. Default is "cuda".
        output_dir: The directory to save the generated images. Default is "diversity_samples".

    Returns:
        float: The average diversity (MSE) between the generated images.
    """
    np.random.seed(seed)

    generated_imgs = []
    bar = tqdm(
        range(num_seeds),
        desc="Generating images for diversity evaluation",
        unit="seed",
        leave=False,
    )
    for s in bar:
        gen = get_generator(np.random.randint(0, 2**32), device)
        img = generate_batch(
            diffusion=diffusion,
            scheduler=scheduler,
            tokenizer=tokenizer,
            prompts=[prompt],
            generator=gen,
            channels=channels,
            output_size=output_size,
            inference_steps=inference_steps,
            cfg_scale=cfg_scale,
            device=device,
        )
        generated_imgs.append(img.float() / 255.0)

    # Compute pairwise MSE distances between generated images
    distances = []
    for i, j in itertools.combinations(range(num_seeds), 2):
        mse_dist = torch.mean((generated_imgs[i] - generated_imgs[j]) ** 2).item()
        distances.append(mse_dist)

    return float(np.mean(distances))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config()

    if config is None:
        print("Error: Configuration file not found or invalid.")
        sys.exit(1)

    checkpoint_path = config["dataset"]["checkpoint_path"]
    split_path = config["dataset"]["split_dataset_path"]

    available_splits = get_splits(split_path)
    print(f"Available splits:")
    for i, split in enumerate(available_splits):
        print(f"  {i}: {split}")

    selected_split = int(input("Select a split by number: "))
    if selected_split < 0 or selected_split >= len(available_splits):
        print("Invalid split selection.")
        sys.exit(1)
    split_path = split_path + "/" + available_splits[selected_split]

    # 1. Load Tokenizer and Hyperparameters
    available_models = get_models()
    print("Available checkpoints:")
    for i, checkpoint in enumerate(available_models):
        print(f"  {i}: {checkpoint}")

    selected_checkpoint = int(input("Select a checkpoint by number: "))
    if selected_checkpoint < 0 or selected_checkpoint >= len(available_models):
        print("Invalid checkpoint selection.")
        sys.exit(1)
    checkpoint_path = available_models[selected_checkpoint]

    vocabulary_path = Path(checkpoint_path) / "tokenizer_vocabulary.csv"
    hyperparameters_path = Path(checkpoint_path) / "model_hyperparameters.yaml"

    print(f"Using checkpoint folder: {checkpoint_path}")
    print(f"Using vocabulary: {vocabulary_path}")
    print(f"Using hyperparameters: {hyperparameters_path}")

    tokenizer = Tokenizer()
    tokenizer.load_vocabulary(str(vocabulary_path))
    vocabulary_size = tokenizer.get_vocabulary_size()

    hyperparameters = load_hyperparameters(str(hyperparameters_path))

    if hyperparameters is None:
        print(
            f"Error: unable to load hyperparameters from {hyperparameters_path} file."
        )
        sys.exit(1)

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
    dropout = model_hyperparameters["dropout"]

    # 2. Initialize Diffusion Model and Load Checkpoint
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
        dropout=dropout,
    ).to(device)

    available_checkpoints = get_checkpoints(checkpoint_path)
    print(f"Available checkpoints:")
    for i, checkpoint in enumerate(available_checkpoints):
        print(f"  {i}: {checkpoint}")

    selected_checkpoint = int(input("Select a checkpoint by number: "))
    if selected_checkpoint < 0 or selected_checkpoint >= len(available_checkpoints):
        print("Invalid checkpoint selection.")
        sys.exit(1)

    checkpoint = load_checkpoint(
        checkpoint_path + "/" + available_checkpoints[selected_checkpoint],
        device,
    )
    diffusion.load_state_dict(checkpoint["model_state_dict"])

    # 3. Count Parameters and Size
    params, params_size = count_parameters(diffusion)
    print(f"Number of parameters: {params:,}")
    print(f"Total size of parameters: {params_size:,} bytes")

    # 4. Initialize Scheduler
    training_steps = scheduler_hyperparameters["timesteps"]
    beta_start = scheduler_hyperparameters["beta_start"]
    beta_end = scheduler_hyperparameters["beta_end"]
    beta_max = scheduler_hyperparameters["beta_max"]
    s = scheduler_hyperparameters["s"]
    scheduler_type = scheduler_hyperparameters["scheduler_type", "linear"]

    inference_steps = config["inference"]["timesteps"]
    guidance_scale = config["inference"]["guidance_scale"]
    batch_size = config["inference"]["batch_size"]
    metric_samples = config["inference"]["metric_samples"]
    diversity_samples = config["inference"]["diversity_samples"]

    seed = int(input("Enter a random seed for evaluation: "))

    scheduler = Scheduler(
        generator=get_generator(seed, device),
        training_steps=training_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_max=beta_max,
        type=scheduler_type,
        s=s,
    )

    scheduler.set_inference_timesteps(inference_steps)

    # 5. Measure Sampling Time and VRAM Usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    start_time = time.time()

    # Generate a batch of images to measure time and VRAM usage
    _ = generate_batch(
        diffusion=diffusion,
        scheduler=scheduler,
        tokenizer=tokenizer,
        prompts=[""] * batch_size,
        generator=get_generator(seed, device),
        channels=resnet_in_channels,
        output_size=image_size,
        inference_steps=inference_steps,
        cfg_scale=guidance_scale,
        device=device,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    sampling_time_single_batch = time.time() - start_time
    sampling_time_per_image = sampling_time_single_batch / batch_size
    vram_peak_bytes = (
        torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
    )
    vram_peak_mb = vram_peak_bytes / (1024**2)

    print(f"Average sampling time per image: {sampling_time_per_image:.4f} s")
    print(f"Peak VRAM usage: {vram_peak_mb:.2f} MB")

    # 6. Prepare Dataset and DataLoader for FID/KID Evaluation
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset_test = AvatarDataset(
        data_path=split_path,
        image_extension="png",
        caption_extension="txt",
        transform=transform,
    )

    dataloader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        generator=get_generator(seed, device),
    )

    # 7. FID and KID Evaluation
    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = KernelInceptionDistance(subset_size=min(50, metric_samples)).to(device)

    # Load real images and update FID/KID with them
    real_count = 0
    print("Loading real images for FID/KID evaluation...")
    bar = tqdm(dataloader, desc="Processing real images", unit="batch", leave=False)
    for real_imgs, captions in bar:
        real_imgs_uint8 = (real_imgs * 255).to(torch.uint8).to(device)
        fid.update(real_imgs_uint8, real=True)
        kid.update(real_imgs_uint8, real=True)
        real_count += real_imgs.size(0)
        if real_count >= metric_samples:
            break

    # Generate synthetic images and update FID/KID with them
    print("Generating synthetic images for FID/KID evaluation...")
    gen_count = 0

    bar = tqdm(
        dataloader, desc="Generating synthetic images", unit="batch", leave=False
    )
    for _, captions in bar:
        current_batch_prompts = list(captions)
        fake_imgs_uint8 = generate_batch(
            diffusion=diffusion,
            scheduler=scheduler,
            tokenizer=tokenizer,
            prompts=current_batch_prompts,
            generator=get_generator(seed, device),
            channels=resnet_in_channels,
            output_size=image_size,
            inference_steps=inference_steps,
            cfg_scale=guidance_scale,
            device=device,
        )

        fid.update(fake_imgs_uint8, real=False)
        kid.update(fake_imgs_uint8, real=False)
        gen_count += len(current_batch_prompts)
        if gen_count >= metric_samples:
            break

    fid_score = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    kid_score = kid_mean.item()

    print(f"FID: {fid_score:.4f}")
    print(f"KID: {kid_score:.4f}")

    # 8. Diversity Evaluation on the Same Prompt
    # TODO: estrai un prompt di test casuale da un'immagine di test
    test_prompt = ""
    diversity_score = evaluate_diversity(
        diffusion=diffusion,
        scheduler=scheduler,
        tokenizer=tokenizer,
        prompt=test_prompt,
        num_seeds=5,
        channels=resnet_in_channels,
        output_size=image_size,
        inference_steps=inference_steps,
        cfg_scale=guidance_scale,
        device=device,
    )
    print(f"Diversity Score: {diversity_score:.6f}")

    # 9. Save Metrics to CSV
    output_path = checkpoint_path + "/metrics.csv"
    file_exists = Path(output_path).exists()
    with open(output_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "checkpoint_path",
                    "hyperparameters_path",
                    "split_path",
                    "scheduler_type",
                    "parameters_count",
                    "parameters_size_bytes",
                    "image_size",
                    "sampling_time_per_img_s",
                    "gpu_vram_peak_mb",
                    "fid",
                    "kid",
                    "diversity_score",
                ]
            )
        writer.writerow(
            [
                checkpoint_path,
                hyperparameters_path,
                split_path,
                scheduler_type,
                f"{params:,}",
                f"{params_size:,}",
                f"{image_size}",
                f"{sampling_time_per_image:.4f}",
                f"{vram_peak_mb:.2f}",
                f"{fid_score:.4f}",
                f"{kid_score:.4f}",
                f"{diversity_score:.6f}",
            ]
        )

    print(f"Metrics saved successfully to {output_path}")
