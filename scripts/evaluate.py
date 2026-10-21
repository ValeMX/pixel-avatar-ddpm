import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import csv
import itertools
import numpy as np
import time
import torch
import torch.nn as nn
import shutil

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
from src.utils.splits import get_compositional_test_samples


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


def evaluate_diversity(
    diffusion: Diffusion,
    scheduler: Scheduler,
    tokenizer: Tokenizer,
    prompts: list[str],
    images: list[Image.Image],
    num_seeds: int = 5,
    channels: int = 3,
    output_size: int = 32,
    inference_steps: int = 250,
    cfg_scale: float = 3.5,
    output_dir: str = "diversity_samples",
    device: torch.device = torch.device("cuda"),
) -> dict[int, float]:
    """Evaluate the diversity of generated images for a given prompt using different random seeds.

    Args:
        diffusion (Diffusion): The diffusion model to use for generation.
        scheduler (Scheduler): The scheduler for the diffusion process.
        tokenizer (Tokenizer): The tokenizer for encoding prompts.
        prompts (list[str]): A list of text prompts to generate images from.
        images (list[Image.Image]): A list of original images corresponding to the prompts.
        num_seeds (int, optional): Number of different random seeds to use for generation. Defaults to 5.
        channels (int, optional): Number of channels in the output images. Defaults to 3.
        output_size (int, optional): Size of the output images. Defaults to 32.
        inference_steps (int, optional): Number of denoising steps. Defaults to 250.
        cfg_scale (float, optional): Classifier-Free Guidance scale. Defaults to 3.5.
        output_dir (str, optional): Directory to save generated images for visual inspection. Defaults to "diversity_samples".
        device (torch.device, optional): Device to run the generation on. Defaults to CUDA if available.

    Returns:
        dict[int, float]: A dictionary mapping prompt indices to their average diversity (MSE).
    """
    generated_imgs = []
    bar = tqdm(
        range(num_seeds),
        desc="Generating images for diversity evaluation",
        unit="seed",
        leave=False,
    )

    # Check if the output directory exists, if not create it
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save the original images for visual inspection
    for i, img in enumerate(images):
        img.save(f"{output_dir}/original_{i}.png")

    if len(images) > 0:
        generated_imgs.append(
            torch.stack([transforms.ToTensor()(img).to(device) for img in images])
        )

    for s in bar:
        gen = get_generator(np.random.randint(0, np.iinfo(np.int32).max), device)
        scheduler.set_generator(gen)
        img = generate_batch(
            diffusion=diffusion,
            scheduler=scheduler,
            tokenizer=tokenizer,
            prompts=prompts,
            generator=gen,
            channels=channels,
            output_size=output_size,
            inference_steps=inference_steps,
            cfg_scale=cfg_scale,
            device=device,
        )

        # Save the generated image for visual inspection
        for i in range(img.size(0)):
            img_pil = Image.fromarray(img[i].permute(1, 2, 0).cpu().numpy())
            img_pil.save(f"{output_dir}/seed_{s}_sample_{i}.png")

        # Normalize to [0, 1] for MSE calculation
        generated_imgs.append(img.float() / 255.0)

    # Compute pairwise MSE distances between generated images for the same prompt
    distances = {}
    for img1, img2 in itertools.combinations(generated_imgs, 2):
        for prompt_idx in range(img1.size(0)):
            mse = torch.mean((img1[prompt_idx] - img2[prompt_idx]) ** 2).item()
            (
                distances[prompt_idx].append(mse)
                if prompt_idx in distances
                else distances.setdefault(prompt_idx, [mse])
            )

    return {prompt_idx: np.mean(mse_list) for prompt_idx, mse_list in distances.items()}


def evaluate_computational_metrics(
    diffusion: Diffusion,
    scheduler: Scheduler,
    tokenizer: Tokenizer,
    batch_size: int,
    seed: int = 42,
    channels: int = 3,
    output_size: int = 32,
    inference_steps: int = 250,
    cfg_scale: float = 3.5,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[dict[int, float], dict[int, float]]:
    """Evaluate the computational metrics (sampling time and VRAM usage) of the diffusion model.

    Args:
        diffusion: The diffusion model.
        scheduler: The scheduler.
        tokenizer: The tokenizer.
        batch_size: The batch size to use for evaluation.
        seed: The random seed for reproducibility. Default is 42.
        channels: The number of channels in the generated images. Default is 3.
        output_size: The size of the generated images. Default is 32.
        inference_steps: The number of inference steps. Default is 250.
        cfg_scale: The scale for classifier-free guidance. Default is 3.5.
        device: The device to run the evaluation on. Default is "cuda" if available.

    Returns:
        tuple[dict[int, float], dict[int, float]]: A tuple containing the average sampling time per image (in seconds) and the peak VRAM usage (in MB) for each prompt.
    """
    sampling_times = {}
    vram_peaks = {}
    batch_sizes = set([batch_size] + [1, 2, 4, 8, 16, 32])

    for bs in batch_sizes:
        if bs > batch_size:
            continue

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()

        start_time = time.time()

        # Generate a batch of images to measure time and VRAM usage
        _ = generate_batch(
            diffusion=diffusion,
            scheduler=scheduler,
            tokenizer=tokenizer,
            prompts=[""] * bs,
            generator=get_generator(seed, device),
            channels=channels,
            output_size=output_size,
            inference_steps=inference_steps,
            cfg_scale=cfg_scale,
            device=device,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        sampling_time_per_batch = time.time() - start_time
        sampling_time_per_image = sampling_time_per_batch / bs

        vram_peak_bytes = (
            torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
        )
        vram_peak_mb = vram_peak_bytes / (1024**2)

        sampling_times[bs] = sampling_time_per_image
        vram_peaks[bs] = vram_peak_mb

    return sampling_times, vram_peaks


def evaluate_fid_kid(
    diffusion: Diffusion,
    scheduler: Scheduler,
    tokenizer: Tokenizer,
    dataloader: DataLoader,
    batch_size: int,
    metric_samples: int,
    conditioned: bool = True,
    seed: int = 42,
    channels: int = 3,
    output_size: int = 32,
    inference_steps: int = 250,
    cfg_scale: float = 3.5,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[float, float, float]:
    """Evaluate the FID and KID metrics of the diffusion model.

    Args:
        diffusion: The diffusion model.
        scheduler: The scheduler.
        tokenizer: The tokenizer.
        dataloader: The DataLoader for the dataset to evaluate on.
        batch_size: The batch size to use for evaluation.
        metric_samples: The number of samples to use for FID/KID evaluation.
        seed: The random seed for reproducibility. Default is 42.
        channels: The number of channels in the generated images. Default is 3.
        output_size: The size of the generated images. Default is 32.
        inference_steps: The number of inference steps. Default is 250.
        cfg_scale: The scale for classifier-free guidance. Default is 3.5.
        device: The device to run the evaluation on. Default is "cuda" if available.

    Returns:
        tuple[float, float, float]: A tuple containing the FID, KID mean, and KID standard deviation scores.
    """
    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = KernelInceptionDistance(feature=2048, subset_size=min(50, metric_samples)).to(
        device
    )

    dataloader_iterator = itertools.cycle(dataloader)

    # Load real images and update FID/KID with them
    print("Loading real images for FID/KID evaluation...")
    bar = tqdm(
        range(metric_samples // batch_size),
        desc="Processing real images",
        unit="batch",
        leave=True,
    )
    for _ in bar:
        real_imgs, captions = next(dataloader_iterator)
        real_imgs_uint8 = (real_imgs * 255).to(torch.uint8).to(device)
        fid.update(real_imgs_uint8, real=True)
        kid.update(real_imgs_uint8, real=True)

    # Generate synthetic images and update FID/KID with them
    print("Generating synthetic images for FID/KID evaluation...")
    bar = tqdm(
        range(metric_samples // batch_size),
        desc="Generating synthetic images",
        unit="batch",
        leave=True,
    )
    for _ in bar:
        _, captions = next(dataloader_iterator)
        current_batch_prompts = list(captions)

        if not conditioned:
            current_batch_prompts = [""] * len(current_batch_prompts)

        fake_imgs_uint8 = generate_batch(
            diffusion=diffusion,
            scheduler=scheduler,
            tokenizer=tokenizer,
            prompts=current_batch_prompts,
            generator=get_generator(seed, device),
            channels=channels,
            output_size=output_size,
            inference_steps=inference_steps,
            cfg_scale=cfg_scale,
            device=device,
        )

        fid.update(fake_imgs_uint8, real=False)
        kid.update(fake_imgs_uint8, real=False)

    fid_score = fid.compute().item()
    kid_mean, kid_std = kid.compute()

    return fid_score, kid_mean.item(), kid_std.item()


def evaluate_compositional_metrics(
    split_path: str,
    checkpoint_path: str,
    diffusion: Diffusion,
    scheduler: Scheduler,
    tokenizer: Tokenizer,
    seed: int = 42,
    channels: int = 3,
    output_size: int = 32,
    inference_steps: int = 250,
    guidance_scale: float = 3.5,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> list[float]:
    """Evaluate the compositional metrics of the diffusion model.

    For each compositional sample in the dataset split, this function generates an image
    using the diffusion model and computes the Mean Squared Error (MSE) between the generated image
    and the corresponding real image.
    This function saves both the real and generated images for visual inspection and saves
    the MSE scores for each compositional sample in the output folder.

    Args:
        split_path: The path to the dataset split for compositional evaluation.
        checkpoint_path: The path to the model checkpoint.
        diffusion: The diffusion model.
        scheduler: The scheduler.
        tokenizer: The tokenizer.
        seed: The random seed for reproducibility. Default is 42.
        channels: The number of channels in the generated images. Default is 3.
        output_size: The size of the generated images. Default is 32.
        inference_steps: The number of inference steps. Default is 250.
        guidance_scale: The scale for classifier-free guidance. Default is 3.5.
        device: The device to run the evaluation on. Default is "cuda" if available.

    Returns:
        list[float]: A list containing the MSE scores for each compositional sample.
    """
    # Initialize the list to store the MSE scores
    mse_scores = []

    compositional_samples = get_compositional_test_samples(
        split_path=split_path,
        seed=seed,
    )

    samples_path = Path(checkpoint_path) / "compositional_samples"
    samples_path.mkdir(parents=True, exist_ok=True)

    for combination in compositional_samples:
        real_image_path = combination["image"]
        prompt_path = combination["prompt"]
        combination_str = f"{combination['attribute_1']}-{combination['value_1']}---{combination['attribute_2']}-{combination['value_2']}"

        shutil.copy(
            real_image_path,
            samples_path / f"real_{combination_str}.png",
        )

        prompt = ""
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        generated_image = generate_batch(
            diffusion=diffusion,
            scheduler=scheduler,
            tokenizer=tokenizer,
            prompts=[prompt],
            generator=get_generator(seed, device),
            channels=channels,
            output_size=output_size,
            inference_steps=inference_steps,
            cfg_scale=guidance_scale,
            device=device,
        )

        generated_image = Image.fromarray(
            generated_image[0].permute(1, 2, 0).cpu().numpy()
        ).convert("RGB")
        generated_image.save(samples_path / f"generated_{combination_str}.png")

        # Compute MSE between the real and generated images
        real_image_np = (
            np.array(Image.open(real_image_path).convert("RGB"), dtype=np.float32)
            / 255.0
        )
        generated_image_np = np.array(generated_image, dtype=np.float32) / 255.0
        mse = np.mean((real_image_np - generated_image_np) ** 2)
        mse_scores.append(float(mse))

    metrics_path = samples_path / "compositional_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(range(len(mse_scores)))
        writer.writerow(mse_scores)

    return mse_scores


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config()

    if config is None:
        print("Error: Configuration file not found or invalid.")
        sys.exit(1)

    checkpoint_path = config["dataset"]["checkpoint_path"]
    split_path = config["dataset"]["split_dataset_path"]
    metrics_path = config["dataset"]["metrics_path"]
    metrics_computational_path = config["dataset"]["metrics_computational_path"]

    available_splits = get_splits(split_path)

    print(f"Available splits:")
    for i, split in enumerate(available_splits, start=1):
        print(f"  {i}: {split}")

    chosen_split = input("Select a split by number: ").strip()
    try:
        chosen_split_index = int(chosen_split) - 1
        if 0 <= chosen_split_index < len(available_splits):
            split_path = split_path + f"/{available_splits[chosen_split_index]}"
            print(f"Using split: {available_splits[chosen_split_index]}")
        else:
            raise ValueError("Invalid choice. Exiting.")
    except ValueError:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    # 1. Load Tokenizer and Hyperparameters
    available_models = get_models(checkpoint_path)
    print("Available models:")
    for i, model in enumerate(available_models, start=1):
        print(f"  {i}: {model}")

    chosen_model = input("Select a model by number: ").strip()
    try:
        chosen_model_index = int(chosen_model) - 1
        if 0 <= chosen_model_index < len(available_models):
            checkpoint_path += "/" + available_models[chosen_model_index]
            print(f"Using model: {available_models[chosen_model_index]}")
        else:
            raise ValueError("Invalid choice. Exiting.")
    except ValueError:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    vocabulary_path = checkpoint_path + "/tokenizer_vocabulary.csv"
    hyperparameters_path = checkpoint_path + "/model_hyperparameters.yaml"

    tokenizer = Tokenizer()
    tokenizer.load_vocabulary(vocabulary_path)
    vocabulary_size = tokenizer.get_vocabulary_size()

    hyperparameters = load_hyperparameters(hyperparameters_path)

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
    for i, checkpoint in enumerate(available_checkpoints, start=1):
        print(f"  {i}: {checkpoint}")

    chosen_checkpoint = input("Select a checkpoint by number: ").strip()
    try:
        chosen_checkpoint_index = int(chosen_checkpoint) - 1
        if 0 <= chosen_checkpoint_index < len(available_checkpoints):
            chosen_checkpoint = available_checkpoints[chosen_checkpoint_index]
            print(f"Using checkpoint: {available_checkpoints[chosen_checkpoint_index]}")
        else:
            raise ValueError("Invalid choice. Exiting.")
    except ValueError:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    checkpoint = load_checkpoint(
        checkpoint_path + "/" + chosen_checkpoint,
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
    scheduler_type = scheduler_hyperparameters["type"]

    # Load inference parameters from the configuration file
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
    sampling_times, vram_peaks = evaluate_computational_metrics(
        diffusion=diffusion,
        scheduler=scheduler,
        tokenizer=tokenizer,
        batch_size=batch_size,
        seed=seed,
        channels=resnet_in_channels,
        output_size=image_size,
        inference_steps=inference_steps,
        cfg_scale=guidance_scale,
        device=device,
    )

    sampling_time_per_image = np.mean(list(sampling_times.values()))
    vram_peak_mb = np.max(list(vram_peaks.values()))

    print(f"Average sampling time per image: {sampling_time_per_image:.4f} s")
    print(f"Peak VRAM usage: {vram_peak_mb:.2f} MB")

    # 6. Prepare Dataset and DataLoader for FID/KID Evaluation
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset_test = AvatarDataset(
        data_path=split_path + "/test",
        image_extension="png",
        caption_extension="txt",
        transform=transform,
    )

    dataloader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        generator=get_generator(seed),
    )

    # 7. FID and KID Evaluation
    if "unconditioned" in checkpoint_path.lower():
        print("Evaluating FID and KID for unconditioned generation...")
        fid_score, kid_mean, kid_std = evaluate_fid_kid(
            diffusion=diffusion,
            scheduler=scheduler,
            tokenizer=tokenizer,
            dataloader=dataloader,
            batch_size=batch_size,
            metric_samples=metric_samples,
            conditioned=False,
            seed=seed,
            channels=resnet_in_channels,
            output_size=image_size,
            inference_steps=inference_steps,
            cfg_scale=0.0,
            device=device,
        )

    else:
        fid_score, kid_mean, kid_std = evaluate_fid_kid(
            diffusion=diffusion,
            scheduler=scheduler,
            tokenizer=tokenizer,
            dataloader=dataloader,
            batch_size=batch_size,
            metric_samples=metric_samples,
            seed=seed,
            channels=resnet_in_channels,
            output_size=image_size,
            inference_steps=inference_steps,
            cfg_scale=guidance_scale,
            device=device,
        )

    print(f"FID: {fid_score:.4f}")
    print(f"KID: {kid_mean:.4f} ± {kid_std:.4f}")

    # 8. Compositional Evaluation
    compositional_scores = []
    if "compositional" in split_path.lower():
        compositional_scores = evaluate_compositional_metrics(
            split_path=split_path,
            checkpoint_path=checkpoint_path,
            diffusion=diffusion,
            scheduler=scheduler,
            tokenizer=tokenizer,
            seed=seed,
            channels=resnet_in_channels,
            output_size=image_size,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            device=device,
        )

    # 9. Diversity Evaluation on the Same Prompt
    np.random.seed(seed)

    test_prompts = []
    original_images = []

    # Extract a prompt from the test dataset for diversity evaluation
    if len(dataset_test) > 0 and "unconditioned" not in checkpoint_path.lower():
        choice = np.random.randint(0, len(dataset_test))
        test_caption_file = dataset_test.caption_paths[choice]
        with open(test_caption_file, "r", encoding="utf-8") as f:
            test_prompt = f.read().strip()
        test_prompts.append(test_prompt)

        test_image_file = dataset_test.image_paths[choice]
        original_image = Image.open(test_image_file).convert("RGB")
        original_images.append(original_image)
    elif "unconditioned" in checkpoint_path.lower():
        test_prompts.append("")

    diversity_scores = evaluate_diversity(
        diffusion=diffusion,
        scheduler=scheduler,
        tokenizer=tokenizer,
        images=original_images,
        prompts=test_prompts,
        num_seeds=diversity_samples,
        channels=resnet_in_channels,
        output_size=image_size,
        inference_steps=inference_steps,
        cfg_scale=guidance_scale,
        device=device,
        output_dir=checkpoint_path + "/diversity_samples",
    )

    print(f"Diversity (MSE) score: {diversity_scores[0]:.6f}")
    if len(test_prompts) > 0:
        print(f"Prompt used for diversity evaluation: {test_prompts}")

    # 10. Save Metrics to CSV
    output_path = checkpoint_path + "/" + metrics_computational_path
    file_exists = Path(output_path).exists()
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(sampling_times.keys())
        writer.writerow(sampling_times.values())
        writer.writerow(vram_peaks.values())

    output_path = checkpoint_path + "/" + metrics_path
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
                    "fid",
                    "kid_mean",
                    "kid_std",
                    "diversity_score",
                    "test_prompt",
                ]
            )
        writer.writerow(
            [
                checkpoint_path,
                hyperparameters_path,
                split_path,
                scheduler_type,
                f"{params}",
                f"{params_size}",
                f"{image_size}",
                f"{fid_score:.6f}",
                f"{kid_mean:.6f}",
                f"{kid_std:.6f}",
                f"{diversity_scores[0]:.6f}",
                f"{test_prompts[0] if len(test_prompts) > 0 else 'N/A'}",
            ]
        )

    print(f"Metrics saved successfully to {output_path}")
