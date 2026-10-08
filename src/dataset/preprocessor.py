import matplotlib.pyplot as plt
import numpy as np
import random
import textwrap
import yaml

from pathlib import Path
from PIL import Image


class Preprocessor:
    """A class for preprocessing images, including cropping and resizing."""

    def __init__(
        self, data_path: str, mappings: dict, crop_size: int = 64, image_size: int = 32
    ):
        """Initialize the ImagePreprocessor with the specified parameters.

        Args:
            data_path: Path to the directory containing the images.
            mapping: A dictionary mapping attribute names to their corresponding textual values.
            crop_size: Number of pixels to remove from each border of the image. Default is 64 pixels.
            image_size: The width and height of the output resized image in pixels. Default is 32 pixels.
        """
        self.data_path = data_path
        self.mappings = mappings
        self.crop_size = crop_size
        self.image_size = image_size

    def crop_avatar(self, image: Image.Image) -> Image.Image:
        """Crop an equal number of pixels from every border of an avatar image.

        Args:
            image: The source avatar image.

        Returns:
            A cropped image with the outer borders removed.
        """
        width, height = image.size

        # Remove crop_size pixels from each side.
        left = self.crop_size
        upper = self.crop_size
        right = width - self.crop_size
        lower = height - self.crop_size

        # Crop and return the image without its outer borders.
        return image.crop((left, upper, right, lower))

    def resize_image(
        self, image: Image.Image, strategy: str = "nearest"
    ) -> Image.Image:
        """Resize an image to a square using the selected resampling strategy.

        Args:
            image: The source image to be resized.
            strategy: The resampling strategy to use for resizing.
                Options are "nearest", "bilinear", or "bicubic". Default is "nearest".

        Returns:
            The resized square image.
        """
        if strategy == "bilinear":
            return image.resize(
                [self.image_size, self.image_size], Image.Resampling.BILINEAR
            )
        elif strategy == "bicubic":
            return image.resize(
                [self.image_size, self.image_size], Image.Resampling.BICUBIC
            )
        else:
            return image.resize(
                [self.image_size, self.image_size], Image.Resampling.NEAREST
            )

    def generate_caption(self, attributes: dict) -> str:
        """Generate a caption based on the provided attributes.

        Args:
            attributes: A dictionary containing attribute names and their corresponding values.

        Returns:
            A string caption generated based on the provided attributes.
        """
        parts = []

        # 1. Skin and Face
        face_shape = self._get_mapped_value("face_shape", attributes)
        face_color = self._get_mapped_value("face_color", attributes)
        if face_shape and face_color:
            parts.append(f"a {face_shape} face and {face_color} skin")
        elif face_shape:
            parts.append(f"a {face_shape} face")
        elif face_color:
            parts.append(f"{face_color} skin")

        # 2. Chin
        chin_length = self._get_mapped_value("chin_length", attributes)
        if chin_length:
            parts.append(f"a {chin_length} chin")

        # 3. Eyes (color, angle, lashes and eyelids)
        eye_color = self._get_mapped_value("eye_color", attributes)
        eye_angle = self._get_mapped_value("eye_angle", attributes)
        eye_slant = self._get_mapped_value("eye_slant", attributes)
        eye_lashes = self._get_mapped_value("eye_lashes", attributes)

        eye_desc = []
        if eye_color:
            eye_desc.append(eye_color)
        if eye_slant and eye_slant != "straight":
            eye_desc.append(eye_slant)
        if eye_angle and eye_angle != "neutral":
            eye_desc.append(eye_angle)

        if eye_desc:
            parts.append(f"{' '.join(eye_desc)} eyes")
        else:
            parts.append("eyes")

        if eye_lashes == "visible":
            parts.append("visible eyelashes")

        # 4. Eyebrows
        eyebrow_shape = self._get_mapped_value("eyebrow_shape", attributes)
        eyebrow_weight = self._get_mapped_value("eyebrow_weight", attributes)
        eyebrow_thickness = self._get_mapped_value("eyebrow_thickness", attributes)
        eyebrow_width = self._get_mapped_value("eyebrow_width", attributes)
        hair_color = self._get_mapped_value("hair_color", attributes)

        brow_desc = []
        if eyebrow_thickness:
            brow_desc.append(eyebrow_thickness)
        if eyebrow_width:
            brow_desc.append(eyebrow_width)
        if eyebrow_shape:
            brow_desc.append(eyebrow_shape)
        if hair_color:
            brow_desc.append(f"{hair_color}")

        if brow_desc:
            parts.append(f"{' '.join(brow_desc)} eyebrows")

        # 5. Facial Hair
        facial_hair = self._get_mapped_value("facial_hair", attributes)
        if facial_hair and facial_hair != "no":
            if hair_color:
                parts.append(f"a {hair_color} {facial_hair}")
            else:
                parts.append(f"a {facial_hair}")

        # 6. Glasses
        glasses = self._get_mapped_value("glasses", attributes)
        glasses_color = self._get_mapped_value("glasses_color", attributes)
        if glasses and glasses != "no":
            if glasses_color:
                parts.append(f"{glasses_color} {glasses}")
            else:
                parts.append(f"{glasses}")

        # Final caption assembly
        if not parts:
            return "A cartoon avatar."

        if len(parts) == 1:
            caption = f"A cartoon avatar with {parts[0]}."
        else:
            caption = f"A cartoon avatar with {', '.join(parts[:-1])}, and {parts[-1]}."

        return caption

    def get_random_image(
        self, image_extension: str = "png", attributes_extension: str = "csv"
    ) -> tuple[Image.Image, str]:
        """Load a randomly selected image with the requested extension.

        Args:
            image_extension: The file extension of the images to load. Default is "png".
            attributes_extension: The file extension of the attributes files to load. Default is "csv".

        Returns:
            A tuple containing the randomly selected image and its corresponding caption.

        Raises:
            FileNotFoundError: If the directory contains no matching images.
        """
        image_paths = [
            path
            for path in Path(self.data_path).iterdir()
            if path.is_file() and path.suffix.lower() == f".{image_extension}"
        ]

        if not image_paths:
            raise FileNotFoundError(
                f"No .{image_extension} images found in {self.data_path}"
            )

        image_path = random.choice(image_paths)
        attributes_path = image_path.with_suffix(f".{attributes_extension}")

        attributes = self.get_attributes(attributes_path)
        caption = self.generate_caption(attributes)
        image = Image.open(image_path)

        return image, caption

    def _get_mapped_value(self, attribute_name: str, attributes: dict) -> str | None:
        """Helper method to look up textual value from YAML mapping.

        Args:
            attribute_name: The name of the attribute to look up.
            attributes: A dictionary containing attribute names and their corresponding values.

        Returns:
            The textual value corresponding to the attribute name, or None if not found.
        """
        if attribute_name not in attributes.keys():
            return None

        val_idx = attributes[attribute_name]
        if (
            attribute_name in self.mappings
            and self.mappings[attribute_name] is not None
        ):
            return self.mappings[attribute_name].get(val_idx, None)

        return None

    def get_attributes(self, attributes_path) -> dict:
        """Load the attributes of an image from a CSV file.

        Args:
            attributes_path: The path to the CSV file containing the attributes.

        Returns:
            A dictionary containing attribute names and their corresponding values.
        """
        attributes = {}
        with open(attributes_path, "r") as file:
            for line in file:
                key, value, _ = line.strip().split(",")
                key = key.strip().strip('"')
                attributes[key] = int(value)
        return attributes


if __name__ == "__main__":
    raw_data_path = "data/raw/cartoonset10k"
    mappings_path = "configs/attribute_mappings.yaml"
    crop_size = 64
    img_size = 32

    with open(mappings_path, "r") as file:
        mappings = yaml.safe_load(file)

    preprocessor = Preprocessor(raw_data_path, mappings, crop_size, img_size)
    original_image, caption = preprocessor.get_random_image()
    cropped_image = preprocessor.crop_avatar(original_image)
    resized_image1 = preprocessor.resize_image(original_image)
    resized_image2 = preprocessor.resize_image(original_image, "bilinear")
    resized_image3 = preprocessor.resize_image(original_image, "bicubic")

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

    wrapped_caption = textwrap.fill(caption, width=70)
    figure.text(0.5, 0.02, wrapped_caption, ha="center", va="bottom")
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    plt.show()
