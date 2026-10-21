# Pixel Avatar DDPM

A text-conditioned Denoising Diffusion Probabilistic Model (DDPM) that generates small pixel-art cartoon avatars from natural-language descriptions. The model is trained from scratch on the [Cartoon Set](https://google.github.io/cartoonset/) dataset, with captions automatically generated from the dataset's attribute annotations.

This project was developed for the **Deep Learning** course at **Politecnico di Bari**.

Everything in this repository is implemented in PyTorch, including the noise scheduler, the U-Net-style denoiser (with residual, self-attention and cross-attention blocks), the tokenizer and the text encoder.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Reproducibility Guide](#reproducibility-guide)
6. [Configuration](#configuration)
7. [Model Architecture](#model-architecture)
8. [Utility Scripts](#utility-scripts)

---

## Overview

The pipeline consists of the following stages:

1. **Preprocessing**: avatars from Cartoon Set (10k) are cropped to remove the white border, resized to a small square resolution (default `32x32`), and paired with a caption generated from the CSV attribute files (e.g. *"A cartoon avatar with a round face and light tan skin, ..."*).
2. **Splitting**: the processed dataset is split into train / validation / test sets with a fixed seed.
3. **Training**: a diffusion model predicts the noise added to the images (epsilon-prediction, MSE loss), conditioned on the caption through cross-attention.
4. **Generation**: images are sampled with ancestral DDPM sampling, optionally using Classifier-Free Guidance (CFG).
4. **Evaluation**: trained models are evaluated using computational metrics, FID and KID indices, test sampling and diversity across seeds generations.

---

## Project Structure

```
.
├── configs/
│   ├── config.yaml                # Main configuration (dataset, model, scheduler, training, inference)
│   └── attribute_mappings.yaml    # Attribute index -> textual description (used for captions)
├── data/                          # (git-ignored) raw, processed and split data
│   ├── raw/cartoonset10k/         # Original Cartoon Set images + CSV attributes
│   ├── processed32/               # Cropped/resized 32x32 images (.png) + generated captions (.txt)
│   ├── processed64/               # Cropped/resized 64x64 images (.png) + generated captions (.txt)
│   ├── splits/<split_id>/         # train / validation / test splits
│   └── samples_by_attribute/      # Example images per attribute value (extract_samples.py)
├── checkpoints/                   # One folder per model: 
│   └── <model_name>/              # weights, vocabulary, hyperparameters, metrics and generated samples
├── results/                       # Generated outputs and experiment results (display_results.py)
├── scripts/
│   ├── preprocess.py              # Crop, resize and caption the raw dataset
│   ├── train.py                   # Train the diffusion model
│   ├── generate.py                # Interactive image generation from text prompts
│   ├── evaluate.py                # Evaluate the diffusion model
│   └── extract_samples.py         # Extract example images for each attribute value
├── src/
│   ├── dataset/
│   │   ├── avatar_dataset.py      # PyTorch Dataset (image + caption)
│   │   ├── preprocessor.py        # Cropping, resizing, caption generation
│   │   └── scheduler.py           # Noise scheduler (linear / cosine), add/remove noise
│   ├── models/
│   │   ├── diffusion.py           # Full model: text encoder + denoising network
│   │   ├── resnet/                # Denoising network (U-Net)
│   │   │   ├── resnet.py
│   │   │   ├── resnet_layer.py
│   │   │   ├── residual_block.py
│   │   │   ├── attention_block.py # Self-attention + cross-attention with text
│   │   │   ├── downsample.py
│   │   │   ├── upsample.py
│   │   │   └── time_encoder.py    # Sinusoidal timestep embedding + MLP
│   │   └── text_encoder/
│   │       ├── tokenizer.py       # Word-level tokenizer and vocabulary builder
│   │       ├── text_encoder.py    # Transformer encoder for captions
│   │       └── text_encoder_block.py
│   └── utils/
│       ├── helpers.py             # Config loading, checkpoints, generators
│       └── splits.py              # Splits creations
├── requirements.txt
└── README.md
```

---

## Installation

### Requirements

- Python 3.14.7 or newer
- A CUDA-capable GPU is strongly recommended (the pinned PyTorch build targets CUDA 13.4). CPU execution works but is very slow.

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/ValeMX/pixel-avatar-ddpm.git
cd pixel-avatar-ddpm

# 2. Create and activate a virtual environment
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3. Install the dependencies
pip install -r requirements.txt
```

### Dependencies

| Package        | Version        |
| -------------- | -------------- |
| matplotlib     | 3.11.2         |
| numpy          | 2.5.4          |
| pandas         | 3.0.6          |
| Pillow         | 13.0.0         |
| PyYAML         | 6.0.3          |
| torch          | 2.14.1+cu134   |
| torchvision    | 0.30.0+cu130   |
| tqdm           | 4.67.1         |

---

## Dataset

The project uses the **Cartoon Set 10k** dataset released by Google.

1. Download the 10k version from <https://huggingface.co/datasets/cgarciae/cartoonset/blob/main/data/cartoonset10k.tgz>.
2. Extract it so that images (`.png`) and attribute files (`.csv`) are located in:

```
data/raw/cartoonset10k/
```

Each avatar `cs*.png` has a matching `cs*.csv` file containing its attributes (face shape, hair, eye color, glasses, etc.). The mapping from numeric attribute values to words is defined in `configs/attribute_mappings.yaml` and is used to build the captions.

> The `data/` folder is git-ignored and must be created locally.

---

## Reproducibility Guide

All commands must be run **from the repository root**, because configuration and data paths are relative. The global seed is set in `configs/config.yaml` (`seed: 42`, `training.seed: 42`) and is used for the data split, the noise generator and the data loaders.

### Step 1: Preprocess the dataset

```bash
python scripts/preprocess.py
```

This reads `data/raw/cartoonset10k`, then for each avatar:

- crops `crop_size` pixels (default 64) from every border,
- resizes the result to `image_size x image_size` (default 32) with the configured `strategy` (default `bilinear`),
- generates a caption from the attributes,
- saves `cs_<idx>.png` and `cs_<idx>.txt` into `data/processed/`.

> Warning: the output directory is **deleted and recreated** at every run.

To visually inspect the effect of the different resampling strategies on a random avatar (with its caption), run:

```bash
python src/dataset/preprocessor.py
```

### Step 2: Create the dataset split and train

```bash
python scripts/train.py
```

The script is interactive and will prompt you for:

1. **Split**: type `new` to create a new split (you will be asked for a name), or the number of an existing split under `data/splits/`. The default split is 80% train / 10% validation / 10% test, shuffled with the configured seed.
2. **Model/tokenizer**: type `new` to build a new vocabulary from the training captions and create a new checkpoint folder (you will be asked for a name), or the number of an existing model folder in `checkpoints/`.
3. **Checkpoint**: type `new` to start from scratch, or select the number of an existing `.pt` file to resume training.

Artifacts saved under `checkpoints/<model_name>/`:

- `tokenizer_vocabulary.csv`: the vocabulary (`token, index`),
- `model_hyperparameters.yaml`: the architecture hyperparameters needed to rebuild the model,
- `last.pt`: model + optimizer state, saved every `validation_interval` epochs.
- `best.pt`: model + optimizer state, saved every time the validation loss improves.

### Step 3: Generate images

```bash
python scripts/generate.py
```

The script will ask which model folder and which checkpoint to load, and then enter a loop where you can type prompts such as:

```
a round face and light tan skin, a short chin, black short crop hair, thin black eyebrows, brown eyes
```

Type `exit` to quit. Each generated image is saved in the checkpoint directory as `generated/generated_image_<n>.png` and displayed on screen.

Classifier-Free Guidance can be enabled from `inference.guidance_scale > 0` in the configuration; it requires a model trained with caption dropout (`training.dropout > 0` in the configuration).

### Step 4: Evaluate the model

```bash
python scripts/evaluate.py
```

The script will execute the following tests:

1. Sampling time per image across different batch sizes.
2. Peak VRAM usage across different batch sizes.
3. FID/KID evaluation using images in `data/splits/<split_id>/test/` set.
4. Compositional OOD sample generations if the model in compositional.
5. Diversity across seeds sample generations.

All the results of the tests will be saved in the `checkpoints/<model_id>/` folder.

### Step 5: Display the results

```bash
python scripts/evaluate.py
```

The script will retrieve the evaluation results from all the models in `checkpoints/` and display them in tables and graphs.

### Reproducing a full run (summary)

```bash
pip install -r requirements.txt
# download Cartoon Set 10k into data/raw/cartoonset10k
python scripts/preprocess.py
python scripts/train.py             # -> new split, new model, new checkpoint
python scripts/generate.py          # -> choose the model and checkpoint to sample from
python scripts/evaluate.py          # -> choose the model and checkpoint to evaluate
python scripts/display_results.py   # -> display the results of the evaluations
```

To obtain deterministic results, keep the seeds in `configs/config.yaml` unchanged and use the same PyTorch/CUDA versions. Note that some GPU kernels are not bit-wise deterministic, so tiny numerical differences across hardware are possible.

---

## Configuration
 
All the hyperparameters live in `configs/config.yaml`. Values below are the ones currently set in the file.
 
**Global**
 
| Key            | Default               | Description                          |
| -------------- | --------------------- | ------------------------------------ |
| `project_name` | `pixel-avatar-ddpm`   | Project name                         |
| `seed`         | `42`                  | Global seed for reproducibility      |
 
**`dataset`**
 
| Key                          | Default                    | Description                                                         |
| ---------------------------- | -------------------------- | ------------------------------------------------------------------- |
| `raw_dataset_10k_path`       | `data/raw/cartoonset10k`   | Raw Cartoon Set (10,000 avatars)                                    |
| `processed_dataset_path`     | `data/processed`           | Output of the preprocessing step (images + captions)                |
| `split_dataset_path`         | `data/splits`              | Where train / validation / test splits are stored                   |
| `checkpoint_path`            | `checkpoints`              | Where models and checkpoints are stored                             |
| `metrics_path`               | `metrics.csv`              | File name for the evaluation metrics saved during inference         |
| `metrics_computational_path` | `metrics_computational.csv`| File name for the computational metrics saved during inference      |
| `image_extension`            | `png`                      | Image file extension                                                |
| `attributes_extension`       | `csv`                      | Attribute file extension                                            |
| `caption_extension`          | `txt`                      | Caption file extension                                              |
| `crop_size`                  | `64`                       | Pixels removed from each border (removes the white margin)          |
| `image_size`                 | `32`                       | Final square resolution (32x32 or 64x64)                            |
| `strategy`                   | `bilinear`                 | Resampling: `nearest`, `bilinear`, `bicubic`                        |
| `held_out_combinations`      | 4 pairs (see below)        | Attribute pairs excluded from training for compositional evaluation |
 
Held-out attribute combinations (each pair is removed from the training data and used to test compositional generalization):
 
| # | Attribute 1  | Value 1                   | Attribute 2   | Value 2                  |
| - | ------------ | ------------------------- | ------------- | ------------------------ |
| 1 | `hair_color` | black                     | `facial_hair` | mustache with goatee     |
| 2 | `hair_color` | red                       | `eye_color`   | blue                     |
| 3 | `face_color` | fair pinkish              | `hair`        | top knot bun             |
| 4 | `glasses`    | cat eye dark sunglasses   | `hair_color`  | yellow                   |
 
**`model`**
 
| Key                          | Default | Description                                         |
| ---------------------------- | ------- | --------------------------------------------------- |
| `image_size`                 | `32`    | Size of the input images                            |
| `resnet_in_channels`         | `3`     | Input channels (RGB)                                |
| `resnet_out_channels`        | `3`     | Output channels (RGB)                               |
| `resnet_base_channels`       | `64`    | Base channel width of the denoiser                  |
| `resnet_spatial_resolutions` | `3`     | Number of down/up-sampling levels                   |
| `resnet_attention_heads`     | `4`     | Heads in the attention blocks                       |
| `time_embedding_multiplier`  | `4`     | Multiplier for the time-embedding dimension         |
| `sequence_length`            | `70`    | Maximum caption length (in tokens)                  |
| `text_encoder_layers`        | `4`     | Transformer layers of the text encoder              |
| `text_hidden_size`           | `128`   | Text embedding dimension                            |
| `dropout`                    | `0.1`   | Dropout rate of the text encoder                    |
 
**`scheduler`**
 
| Key                       | Default         | Description                                          |
| ------------------------- | --------------- | ---------------------------------------------------- |
| `timesteps`               | `1000`          | Number of diffusion steps used in training           |
| `beta_start` / `beta_end` | `1e-4` / `0.02` | Noise schedule range (linear schedule)               |
| `beta_max`                | `0.999`         | Maximum beta (cosine schedule clipping)              |
| `s`                       | `0.008`         | Offset of the cosine schedule                        |
| `type`                    | `cosine`        | `linear` or `cosine`                                 |
 
**`training`**
 
| Key                   | Default  | Description                                                          |
| --------------------- | -------- | -------------------------------------------------------------------- |
| `seed`                | `42`     | Seed used during training                                            |
| `epochs`              | `50`     | Number of epochs (suggested: 50 for 32x32, 25 for 64x64)             |
| `batch_size`          | `32`     | Batch size (suggested: 32 for 32x32, 10 for 64x64)                   |
| `learning_rate`       | `0.0002` | Optimizer learning rate                                              |
| `weight_decay`        | `0.01`   | Optimizer weight decay                                               |
| `dropout`             | `0.1`    | Model dropout and caption-dropout probability for CFG                |
| `validation_interval` | `1`      | Epochs between validation passes                                     |
| `patience`            | `10`     | Epochs without improvement before early stopping                     |
 
**`inference`**
 
| Key                 | Default | Description                                                    |
| ------------------- | ------- | -------------------------------------------------------------- |
| `timesteps`         | `250`   | Number of denoising steps at inference time                    |
| `guidance_scale`    | `3.5`   | Classifier-Free Guidance scale                                 |
| `batch_size`        | `32`    | Batch size for generation                                      |
| `metric_samples`    | `250`   | Number of samples generated for the evaluation metrics         |
| `diversity_samples` | `16`    | Number of samples generated for the diversity evaluation       |
 
The architecture hyperparameters are saved alongside each model, so `generate.py` can rebuild the exact same network.

---

## Model Architecture

- **Denoiser (`src/models/resnet`)**: a U-Net-like network built from `ResNetLayer` stages. Each encoder/decoder stage is made of a `ResidualBlock` (GroupNorm + SiLU + convolutions, with time-embedding injection), an `AttentionBlock` (self-attention, cross-attention over the text embedding, and a feed-forward network, all with residual connections), and a down-sampling (strided convolution) or up-sampling (bilinear interpolation + convolution) layer. Skip connections link the encoder to the decoder.
- **Time encoder**: sinusoidal embedding of the timestep followed by a two-layer MLP.
- **Text encoder (`src/models/text_encoder`)**: word embedding + sinusoidal positional encoding + a stack of Transformer encoder blocks. The word-level tokenizer (lowercasing, punctuation stripping, special tokens `[PAD]`, `[MASK]`, `[UNK]`, `[SOS]`, `[EOS]`) builds its vocabulary from the training captions.
- **Scheduler (`src/dataset/scheduler.py`)**: supports linear (in square-root space) and cosine beta schedules, forward noising (`add_noise`) and the DDPM reverse step (`remove_noise`) with predicted-x0 clipping to `[-1, 1]`, plus a configurable number of inference steps.

---

## Utility Scripts

- `scripts/extract_samples.py`: scans the raw dataset and copies one example image for every value of each attribute into `data/samples_by_attribute/<attribute>/`. Useful to verify and refine the textual mappings in `configs/attribute_mappings.yaml`.
