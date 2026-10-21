import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import os
import torch

from tqdm import tqdm
from PIL import Image

from src.dataset import Scheduler
from src.models import Diffusion
from src.models.text_encoder import Tokenizer
from src.utils.helpers import *


@torch.no_grad()
def generate(
    channels: int,
    prompt: str,
    generator: torch.Generator,
    scheduler: Scheduler,
    tokenizer: Tokenizer,
    diffusion: Diffusion,
    prompt_unconditional: str = "",
    cfg: bool = True,
    cfg_scale=7.5,
    output_size: int = 32,
    inference_steps: int = 50,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> np.ndarray:
    """Generate an image based on a given prompt using the diffusion model.

    Args:
        channels (int): The number of channels in the generated image (e.g., 3 for RGB).
        prompt (str): The text prompt for image generation.
        generator (torch.Generator): A torch.Generator for random number generation.
        scheduler (Scheduler): The Scheduler object for the diffusion process.
        tokenizer (Tokenizer): The Tokenizer object for encoding text prompts.
        diffusion (Diffusion): The Diffusion model for image generation.
        prompt_unconditional (str): The unconditional prompt for Classifier-Free Guidance (CFG).
        cfg (bool): Whether to use Classifier-Free Guidance (CFG) or not. Default is True.
        cfg_scale (float): The scale for CFG, controlling the influence of the conditional prompt. Default is 7.5.
        output_size (int): The size of the output image (output_size x output_size). Default is 32.
        inference_steps (int): The number of steps for the diffusion process during inference. Default is 50.
        device (torch.device): The device to run the model on (CPU or GPU). Default is CUDA if available, otherwise CPU.

    Returns:
        ndarray: A numpy array representing the generated image.
    """
    diffusion.eval()

    # Encode the prompt
    encoded_captions = None
    if cfg:
        # Encode the conditional prompt
        tokens_conditional = tokenizer.encode(prompt)
        tokens_conditional = torch.tensor(
            tokens_conditional,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        # Encode the unconditional prompt
        tokens_unconditional = tokenizer.encode(prompt_unconditional)
        tokens_unconditional = torch.tensor(
            tokens_unconditional,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        encoded_captions = torch.cat([tokens_conditional, tokens_unconditional], dim=0)
    else:
        # Encode the conditional prompt
        tokens_conditional = tokenizer.encode(prompt)
        tokens_conditional = torch.tensor(
            tokens_conditional,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        encoded_captions = tokens_conditional

    scheduler.set_inference_timesteps(inference_steps)

    output_shape = (1, channels, output_size, output_size)
    model_output = torch.randn(output_shape, generator=generator, device=device)

    timesteps = tqdm(scheduler.timesteps)
    for i, timestep in enumerate(timesteps):
        model_input = model_output
        model_timestep = timestep
        model_timestep = model_timestep.unsqueeze(0)
        if cfg:
            model_input = model_input.repeat(2, 1, 1, 1)
            model_timestep = torch.tensor([timestep] * 2, device=device)
        else:
            model_timestep = torch.tensor([timestep], device=device)

        output = diffusion(model_input, model_timestep, encoded_captions)

        if cfg:
            output_conditional, output_unconditional = output.chunk(2)
            output = (
                cfg_scale * (output_conditional - output_unconditional)
                + output_unconditional
            )

        model_output = scheduler.remove_noise(int(timestep), model_output, output)

    images = scale(model_output, (-1, 1), (0, 255), clamp=True)
    images = images.permute(0, 2, 3, 1)
    images = images.to("cpu", torch.uint8).numpy()

    return images[0]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config()

    if config is None:
        print("Error: Configuration file not found or invalid.")
        sys.exit(1)

    seed = config["training"]["seed"]
    checkpoint_path = config["dataset"]["checkpoint_path"]

    # 1. Choose an available tokenizer vocabulary and model
    available_models = get_models(checkpoint_path)
    print(f"Available models: ")
    for i, model in enumerate(available_models, start=1):
        print(f"  {i}. {model}")

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

    # 2. Load the tokenizer and model hyperparameters
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

    # 3. Initialize the diffusion model
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

    # 4. Initialize the scheduler for the diffusion process
    training_steps = scheduler_hyperparameters["timesteps"]
    beta_start = scheduler_hyperparameters["beta_start"]
    beta_end = scheduler_hyperparameters["beta_end"]
    beta_max = scheduler_hyperparameters["beta_max"]
    s = scheduler_hyperparameters["s"]
    scheduler_type = scheduler_hyperparameters["type"]

    scheduler = Scheduler(
        generator=get_generator(seed, device),
        training_steps=training_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_max=beta_max,
        type=scheduler_type,
        s=s,
    )

    # 5. Choose an available checkpoint for the model and load it
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

    # 6. Get inference parameters from the configuration file
    inference_steps = config["inference"]["timesteps"]
    cfg = config["inference"]["guidance_scale"] > 0
    cfg_scale = config["inference"]["guidance_scale"]

    # 7. Check if the output directory exists, and create it if it doesn't
    output_dir = Path(checkpoint_path) / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 8. Generate images based on user-provided prompts
    while True:
        print("Enter a prompt for image generation (or type 'exit' to quit): ")
        user_prompt = input()

        if user_prompt.lower() == "exit":
            break

        generated_image = generate(
            channels=resnet_in_channels,
            prompt=user_prompt,
            generator=scheduler.generator,
            scheduler=scheduler,
            tokenizer=tokenizer,
            diffusion=diffusion,
            cfg=cfg,
            cfg_scale=cfg_scale,
            output_size=image_size,
            inference_steps=inference_steps,
            device=device,
        )

        # Convert the generated image to a PIL Image and display it
        pil_image = Image.fromarray(generated_image)
        pil_image.save(
            f"{output_dir}/generated_image_{len([f for f in os.listdir(output_dir) if f.startswith('generated_image_')])}.png"
        )
        pil_image.show()
