import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from matplotlib import pyplot as plt
from PIL import Image

from src.utils.helpers import get_models, load_config, get_checkpoints

mapping = {
    "unconditioned_ordinary_linear_32": "unconditioned-32",
    "conditioned_ordinary_linear_32": "conditioned-ordinary-32",
    "conditioned_compositional_linear_32": "compositional-linear-32",
    "conditioned_compositional_cosine_32": "compositional-cosine-32",
    "unconditioned_ordinary_linear_64": "unconditioned-64",
    "conditioned_ordinary_linear_64": "conditioned-ordinary-64",
    "conditioned_compositional_linear_64": "compositional-linear-64",
    "conditioned_compositional_cosine_64": "compositional-cosine-64",
}


def compositional_grid(
    size: int,
    held_out_combinations: list,
    available_models: list,
    checkpoints_path: Path,
) -> None:
    """Show a grid of compositional samples for the specified image size.

    Args:
        size (int): The size of the images (e.g., 32 or 64).
        held_out_combinations (list): A list of dictionaries containing the held-out attribute combinations.
        available_models (list): A list of available model names.
        checkpoints_path (Path): The path to the directory containing model checkpoints.
    """
    rows = len(held_out_combinations)
    cols = 2

    compositional_models = [
        model
        for model in available_models
        if "compositional" in model and str(size) in model
    ]

    fig, axes = plt.subplots(rows, cols + len(compositional_models))
    fig.suptitle(f"Compositional Samples ({size}x{size})", fontsize=14)

    # Get the metrics (MSE) for each model and combination
    metrics = {}
    for model in compositional_models:
        metrics_path = (
            Path(checkpoints_path)
            / model
            / "compositional_samples"
            / "compositional_metrics.csv"
        )
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    values = lines[1].strip().split(",")
                    metrics[model] = values
                else:
                    print(f"Warning: Metrics file for checkpoint '{model}' is empty.")
        else:
            print(f"Warning: Metrics file for checkpoint '{model}' not found.")

    for row_idx, combination in enumerate(held_out_combinations):
        combination_str = f"{combination['attribute_1']}-{combination['value_1']}---{combination['attribute_2']}-{combination['value_2']}"

        # Display the combination description in the first column
        combination_description = (
            f"$\\mathbf{{{combination['attribute_1'].replace('_', '\\_')}}}$: "
            f"$\\mathit{{{combination['value_1'].replace(' ', r'\ ')}}}$\n"
            f"$\\mathbf{{{combination['attribute_2'].replace('_', '\\_')}}}$: "
            f"$\\mathit{{{combination['value_2'].replace(' ', r'\ ')}}}$"
        )
        axes[row_idx, 0].text(
            0.5,
            0.5,
            combination_description,
            fontsize=12,
            ha="center",
            va="center",
        )
        axes[row_idx, 0].axis("off")

        # Display the original image from the first model in the second column
        original_model = compositional_models[0]
        original_image_path = (
            Path(checkpoints_path)
            / original_model
            / "compositional_samples"
            / f"real_{combination_str}.png"
        )

        if original_image_path.exists():
            img = Image.open(original_image_path)
            axes[row_idx, 1].imshow(img)
            if row_idx == 0:
                axes[row_idx, 1].set_title("Original Image", fontsize=12)
        else:
            axes[row_idx, 1].text(
                0.5,
                0.5,
                "Image not found",
                fontsize=12,
                ha="center",
                va="center",
            )

        for col_idx, model in enumerate(compositional_models):
            image_path = (
                Path(checkpoints_path)
                / model
                / "compositional_samples"
                / f"generated_{combination_str}.png"
            )

            if image_path.exists():
                img = Image.open(image_path)
                axes[row_idx, col_idx + 2].imshow(img)
                mse_value = metrics.get(model, ["N/A"])[row_idx]
                if mse_value != "N/A":
                    mse_value = f"{float(mse_value):.6f}"
                if row_idx == 0:
                    axes[row_idx, col_idx + 2].set_title(
                        f"{mapping.get(model, model)}", fontsize=12
                    )
                axes[row_idx, col_idx + 2].set_xlabel(f"MSE: {mse_value}", fontsize=10)
            else:
                axes[row_idx, col_idx + 2].text(
                    0.5,
                    0.5,
                    "Image not found",
                    fontsize=12,
                    ha="center",
                    va="center",
                )
                axes[row_idx, col_idx + 2].axis("off")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1)

    plt.subplots_adjust(wspace=0.01)
    plt.show()


def diversity_grid(
    size: int,
    cols: int,
    available_models: list,
    checkpoints_path: Path,
) -> None:
    """Show a grid of diversity samples for the specified image size.

    Args:
        size (int): The image size for which to display diversity samples.
        cols (int): The number of columns in the grid.
        available_models (list): A list of available model names.
        checkpoints_path (Path): The path to the directory containing model checkpoints.

    """
    diversity_models = [model for model in available_models if str(size) in model]
    diversity_models.reverse()  # Reverse the order of models to display unconditioned first
    rows = len(diversity_models)

    if rows == 0:
        print("No diversity models found for the specified image size.")
        return

    fig, axes = plt.subplots(rows, cols + 1, squeeze=False)
    fig.suptitle(f"Diversity Samples ({size}x{size})", fontsize=14)

    for row_idx in range(rows):
        axes[row_idx, 0].text(
            0.5,
            0.5,
            f"{mapping.get(diversity_models[row_idx], diversity_models[row_idx])}",
            fontsize=12,
            ha="center",
            va="center",
        )
        axes[row_idx, 0].axis("off")

    for row_idx, model in enumerate(diversity_models):
        image_paths = list(
            (Path(checkpoints_path) / model / "diversity_samples").glob("*.png")
        )

        for col_idx in range(1, cols + 1):
            axis = axes[row_idx, col_idx]

            axis.set_xticks([])
            axis.set_yticks([])

            image_path = image_paths[col_idx] if col_idx < len(image_paths) else None

            if image_path and image_path.exists():
                img = Image.open(image_path)
                axis.imshow(img)
            else:
                axis.text(
                    0.5,
                    0.5,
                    "Image not found",
                    fontsize=12,
                    ha="center",
                    va="center",
                )

            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(1)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()


def sampling_time_graph(sampling_times: dict[str, dict[int, float]]):
    """Plot sampling time against batch size for each model.

    Args:
        sampling_times (dict[str, dict[int, float]]): A dictionary where keys are model names and values are dictionaries mapping batch sizes to sampling times.
    """
    if not sampling_times:
        print("No sampling time data found.")
        return

    plt.figure(figsize=(8, 5))

    print("\nSampling Times:")
    print(f"{'Model':<50}", end="")

    batch_sizes_set = set()
    for model_times in sampling_times.values():
        batch_sizes_set.update(model_times.keys())
    batch_sizes_set = sorted(batch_sizes_set)

    for batch_size in batch_sizes_set:
        print(f"{batch_size:<10}", end="")
    print()

    for model, model_times in sampling_times.items():
        if not model_times:
            continue

        batch_sizes = sorted(model_times)
        times = [model_times[batch_size] for batch_size in batch_sizes]
        plt.plot(batch_sizes, times, marker="o", label=mapping.get(model, model))

        print(f"{model:<50}", end="")
        for batch_size in batch_sizes_set:
            time = model_times.get(batch_size, "N/A")
            if time != "N/A":
                time = f"{time:.4f}"
            print(f"{time:<10}", end="")
        print()

    plt.xlabel("Batch Size")
    plt.xticks(
        sorted(
            set(
                batch_size
                for model_times in sampling_times.values()
                for batch_size in model_times
            )
        )
    )
    plt.ylabel("Sampling Time per Image (s/img)")
    plt.title("Sampling Time per Image vs Batch Size")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def vram_peak_graph(vram_peaks: dict[str, dict[int, float]]):
    """Plot VRAM peak against batch size for each model.

    Args:
        vram_peaks (dict[str, dict[int, float]]): A dictionary where keys are model names and values are dictionaries mapping batch sizes to VRAM peaks.
    """
    if not vram_peaks:
        print("No VRAM peak data found.")
        return

    plt.figure(figsize=(8, 5))

    print("\nVRAM Peaks:")
    print(f"{'Model':<50}", end="")

    batch_sizes_set = set()
    for model_times in sampling_times.values():
        batch_sizes_set.update(model_times.keys())
    batch_sizes_set = sorted(batch_sizes_set)

    for batch_size in batch_sizes_set:
        print(f"{batch_size:<10}", end="")
    print()

    for model, model_vram in vram_peaks.items():
        if not model_vram:
            continue

        batch_sizes = sorted(model_vram)
        vram_values = [model_vram[batch_size] for batch_size in batch_sizes]
        plt.plot(batch_sizes, vram_values, marker="o", label=mapping.get(model, model))

        print(f"{model:<50}", end="")
        for batch_size in batch_sizes_set:
            vram = model_vram.get(batch_size, "N/A")
            if vram != "N/A":
                vram = f"{vram:.2f}"
            print(f"{vram:<10}", end="")
        print()

    plt.xlabel("Batch Size")
    plt.xticks(
        sorted(
            set(
                batch_size
                for model_vram in vram_peaks.values()
                for batch_size in model_vram
            )
        )
    )
    plt.ylabel("VRAM Peak Usage (MB)")
    plt.title("VRAM Peak Usage vs Batch Size")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    config = load_config()

    if config is None:
        print("Error: Configuration file not found or invalid.")
        sys.exit(1)

    checkpoints_path = config["dataset"]["checkpoint_path"]
    metrics_path = config["dataset"]["metrics_path"]
    metrics_computational_path = config["dataset"]["metrics_computational_path"]

    held_out_combinations = config["dataset"]["held_out_combinations"]

    # 1. Get available checkpoints
    available_models = get_models(checkpoints_path)

    if not available_models:
        print("No checkpoints found in the specified path.")
        sys.exit(1)

    print("Available models:")
    for idx, model in enumerate(available_models):
        print(f"{idx + 1}. {model}")

    metrics = {}

    # 2. Get metrics for each checkpoint
    for model in available_models:
        metrics_file = Path(checkpoints_path) / model / metrics_path
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    header = lines[0].strip().split(",")
                    values = lines[1].strip().split(",")
                    metrics[model] = dict(zip(header, values))
                else:
                    print(f"Warning: Metrics file for checkpoint '{model}' is empty.")
        else:
            print(f"Warning: Metrics file for checkpoint '{model}' not found.")

    # 3. Display metrics
    if metrics:
        # 3.1 FID and KID metrics
        print("\nFID and KID Metrics:")
        print(f"{'Model':<50} {'FID':<10} {'KID Mean':<10} {'KID Std':<10}")

        for model, metric_values in metrics.items():
            fid = metric_values.get("fid", "N/A")
            kid_mean = metric_values.get("kid_mean", "N/A")
            kid_std = metric_values.get("kid_std", "N/A")
            print(f"{model:<50} {fid:<10} {kid_mean:<10} {kid_std:<10}")

        # 3.2 Parameters count and sizes
        print("\nParameters Count and Sizes:")
        print(f"{'Model':<50} {'Parameters Count':<20} {'Size (MB)':<10}")

        for model, metric_values in metrics.items():
            param_count = metric_values.get("parameters_count", "N/A")
            size_mb = metric_values.get("parameters_size_bytes", "N/A")
            if size_mb != "N/A":
                size_mb = int(size_mb) / (1024**2)  # Convert bytes to MB
            print(f"{model:<50} {param_count:<20} {size_mb:<10}")

        # 3.3 Diversity Scores
        print("\nDiversity Scores:")
        print(f"{'Model':<50} {'Diversity Score':<20}")

        for model, metric_values in metrics.items():
            diversity_score = metric_values.get("diversity_score", "N/A")
            print(f"{model:<50} {diversity_score:<20}")

    # 4. Compositional grid for 32x32 images
    # compositional_grid(32, held_out_combinations, available_models, checkpoints_path)

    # 5. Compositional grid for 64x64 images
    # compositional_grid(64, held_out_combinations, available_models, checkpoints_path)

    # 6. Diversity grid for 32x32 images
    diversity_grid(32, 4, available_models, checkpoints_path)

    # 7. Diversity grid for 64x64 images
    diversity_grid(64, 4, available_models, checkpoints_path)

    # 8. Read computational metrics for sampling time and VRAM peaks
    sampling_times = {}
    vram_peaks = {}
    for model in available_models:
        metrics_computational_file = (
            Path(checkpoints_path) / model / metrics_computational_path
        )

        if metrics_computational_file.exists():
            with open(metrics_computational_file, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    batch_sizes = lines[0].strip().split(",")
                    time_values = lines[1].strip().split(",")
                    vram_values = lines[2].strip().split(",")

                    sampling_times[model] = {
                        int(batch_size): float(time)
                        for batch_size, time in zip(batch_sizes, time_values)
                    }
                    vram_peaks[model] = {
                        int(batch_size): float(vram)
                        for batch_size, vram in zip(batch_sizes, vram_values)
                    }

                else:
                    print(
                        f"Warning: Computational metrics file for model '{model}' is empty."
                    )
        else:
            print(f"Warning: Computational metrics file for model '{model}' not found.")

    # 9. Sampling Time Graphs
    sampling_time_graph(sampling_times)

    # 10. VRAM Peak Graphs
    vram_peak_graph(vram_peaks)
