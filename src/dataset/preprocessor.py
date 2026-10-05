import matplotlib.pyplot as plt
import numpy as np
import random

from pathlib import Path
from PIL import Image, ImageOps
from src.utils.helpers import load_config


def crop_avatar(image: Image.Image, crop_size: int = 64) -> Image.Image:
    """Crop an equal number of pixels from every border of an avatar image.

    Args:
        image: The source avatar image.
        crop_size: Number of pixels to remove from each border.

    Returns:
        A cropped image with the outer borders removed.
    """
    width, height = image.size

    # Remove crop_size pixels from each side.
    left = crop_size
    upper = crop_size
    right = width - crop_size
    lower = height - crop_size

    # Crop and return the image without its outer borders.
    return image.crop((left, upper, right, lower))


def resize_image(
    image: Image.Image, size: int = 32, strategy: str = "nearest"
) -> Image.Image:
    """Resize an image to a square using the selected resampling strategy.

    Args:
        image: The source image to resize.
        size: The width and height of the output image in pixels.
        strategy: Resampling method: ``"nearest"``, ``"bilinear"``, or
            ``"bicubic"``. Unrecognized values default to nearest-neighbor.

    Returns:
        The resized square image.
    """
    if strategy == "bilinear":
        return image.resize([size, size], Image.Resampling.BILINEAR)
    elif strategy == "bicubic":
        return image.resize([size, size], Image.Resampling.BICUBIC)
    else:
        return image.resize([size, size], Image.Resampling.NEAREST)


def get_random_image(data_path: str, extension: str = "png") -> Image.Image:
    """Load a randomly selected image with the requested extension.

    Args:
        data_path: Directory containing the image files.
        extension: File extension to filter by, without the leading dot.

    Returns:
        The randomly selected image as a PIL image.

    Raises:
        FileNotFoundError: If the directory contains no matching images.
    """
    image_paths = [
        path
        for path in Path(data_path).iterdir()
        if path.is_file() and path.suffix.lower() == f".{extension}"
    ]

    if not image_paths:
        raise FileNotFoundError(f"No .{extension} images found in {data_path}")

    return Image.open(random.choice(image_paths))


if __name__ == "__main__":
    # Load config
    config = load_config()

    if config:
        raw_data_path = config["data"]["raw_dataset_10k_path"]
        crop_size = config["data"]["crop_size"]
        img_size = config["data"]["image_size"]
    else:
        exit()

    original_image = crop_avatar(get_random_image(raw_data_path), crop_size)
    resized_image1 = resize_image(original_image, img_size)
    resized_image2 = resize_image(original_image, img_size, "bilinear")
    resized_image3 = resize_image(original_image, img_size, "bicubic")

    figure, axes = plt.subplots(1, 4)
    axes[0].imshow(original_image)
    axes[0].set_title("original")
    axes[1].imshow(resized_image1)
    axes[1].set_title(f"nearest {img_size}")
    axes[2].imshow(resized_image2)
    axes[2].set_title(f"bilinear {img_size}")
    axes[3].imshow(resized_image3)
    axes[3].set_title(f"bicubic {img_size}")

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()
    plt.show()
