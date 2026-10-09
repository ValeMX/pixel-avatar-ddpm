import matplotlib.pyplot as plt
import numpy as np
import random
import textwrap
import tqdm
import yaml

from pathlib import Path
from PIL import Image


class Preprocessor:
    """A class for preprocessing images, including cropping and resizing."""

    def __init__(self, data_path: str, mappings: dict):
        """Initialize the ImagePreprocessor with the specified parameters.

        Args:
            data_path: Path to the directory containing the images.
            mapping: A dictionary mapping attribute names to their corresponding textual values.
        """
        self.data_path = data_path
        self.mappings = mappings

    def crop_image(self, image: Image.Image, crop_size: int = 64) -> Image.Image:
        """Crop an equal number of pixels from every border of an avatar image.

        Args:
            image: The source avatar image.
            crop_size: The number of pixels to remove from each border. Default is 64.

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
        self,
        image: Image.Image,
        image_size: int = 32,
        strategy: str = "nearest",
    ) -> Image.Image:
        """Resize an image to a square using the selected resampling strategy.

        Args:
            image: The source image to be resized.
            image_size: The desired size of the square image. Default is 32.
            strategy: The resampling strategy to use for resizing.
                Options are "nearest", "bilinear", or "bicubic". Default is "nearest".

        Returns:
            The resized square image.
        """
        if strategy == "bilinear":
            return image.resize([image_size, image_size], Image.Resampling.BILINEAR)
        elif strategy == "bicubic":
            return image.resize([image_size, image_size], Image.Resampling.BICUBIC)
        else:
            return image.resize([image_size, image_size], Image.Resampling.NEAREST)

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

        # 3. Hair
        hair = self._get_mapped_value("hair", attributes)
        hair_color = self._get_mapped_value("hair_color", attributes)

        if hair and hair_color:
            parts.append(f"{hair_color} {hair} hair")
        elif hair:
            parts.append(f"{hair} hair")

        # 4. Eyebrows (shape, weight, thickness, width, color)
        eyebrow_shape = self._get_mapped_value("eyebrow_shape", attributes)
        eyebrow_weight = self._get_mapped_value("eyebrow_weight", attributes)
        eyebrow_thickness = self._get_mapped_value("eyebrow_thickness", attributes)
        eyebrow_width = self._get_mapped_value("eyebrow_width", attributes)
        hair_color = self._get_mapped_value("hair_color", attributes)

        brow_desc = []
        if eyebrow_weight:
            brow_desc.append(eyebrow_weight)
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

        # 5. Eye - Eyebrow Distance
        eye_eyebrow_distance = self._get_mapped_value(
            "eye_eyebrow_distance",
            attributes,
        )
        if eye_eyebrow_distance:
            parts.append(f"{eye_eyebrow_distance} distance between eyes and eyebrows")

        # 6. Eyes (color, angle, slant, lashes, lid)
        eye_color = self._get_mapped_value("eye_color", attributes)
        eye_angle = self._get_mapped_value("eye_angle", attributes)
        eye_slant = self._get_mapped_value("eye_slant", attributes)
        eye_lashes = self._get_mapped_value("eye_lashes", attributes)
        eye_lid = self._get_mapped_value("eye_lid", attributes)

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

        if eye_lid:
            parts.append(f"{eye_lid} eyelids")

        if eye_lashes == "visible":
            parts.append("visible eyelashes")

        # 7. Facial Hair
        facial_hair = self._get_mapped_value("facial_hair", attributes)
        if facial_hair and facial_hair != "no":
            if hair_color:
                parts.append(f"a {hair_color} {facial_hair}")
            else:
                parts.append(f"a {facial_hair}")

        # 8. Glasses
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
        self,
        image_extension: str = "png",
        attributes_extension: str = "csv",
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
        attributes_name = image_path.with_suffix(f".{attributes_extension}").name

        attributes = self.get_attributes(attributes_name)
        caption = self.generate_caption(attributes)
        image = Image.open(image_path)

        return image, caption

    def get_image(self, image_name: str) -> Image.Image:
        """Load an image from the specified name.

        Args:
            image_name: The name of the image file.

        Returns:
            The loaded image.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
        """
        full_path = Path(self.data_path) / image_name
        if not full_path.is_file():
            raise FileNotFoundError(f"Image file {full_path} not found.")

        return Image.open(full_path)

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

    def get_attributes(self, attributes_name: str) -> dict:
        """Load the attributes of an image from a CSV file.

        Args:
            attributes_name: The name of the CSV file containing the attributes.

        Returns:
            A dictionary containing attribute names and their corresponding values.

        Raises:
            FileNotFoundError: If the specified attributes file does not exist.
        """
        full_path = Path(self.data_path) / attributes_name
        if not full_path.is_file():
            raise FileNotFoundError(f"Attributes file {full_path} not found.")

        attributes = {}

        with open(full_path, "r") as file:
            for line in file:
                key, value, _ = line.strip().split(",")
                key = key.strip().strip('"')
                attributes[key] = int(value)

        return attributes

    def preprocess_dataset(
        self,
        output_dir: str = "data/processed",
        output_prefix: str = "cs_",
        image_extension: str = "png",
        attributes_extension: str = "csv",
        caption_extension: str = "txt",
        crop_size: int = 64,
        image_size: int = 32,
        strategy: str = "nearest",
    ) -> None:
        """Preprocess the entire dataset by cropping and resizing images and
        by generating captions based on their attributes and
        saving them to the specified output directory.

        Args:
            output_dir: The directory where the processed images and captions will be saved. Default is "data/processed".
            image_extension: The file extension of the images to process. Default is "png".
            attributes_extension: The file extension of the attributes files to process. Default is "csv".
            caption_extension: The file extension of the caption files to process. Default is "txt".
            crop_size: The number of pixels to remove from each border during cropping. Default is 64.
            image_size: The desired size of the square image after resizing. Default is 32.
            strategy: The resampling strategy to use for resizing.
                Options are "nearest", "bilinear", or "bicubic". Default is "nearest".
        """
        # 1. Get all image names in the dataset directory with the specified extension.
        image_names = [
            path.name
            for path in Path(self.data_path).iterdir()
            if path.is_file() and path.suffix.lower() == f".{image_extension}"
        ]

        image_names.sort()  # Sort the image names for consistent processing order.

        print(f"Found {len(image_names)} images. Starting preprocessing...")

        # 2. Delete existing files in output directory to avoid duplicates.
        output_path = Path(output_dir)
        if output_path.exists():
            for file in output_path.iterdir():
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    for subfile in file.iterdir():
                        subfile.unlink()
                    file.rmdir()
            output_path.rmdir()
        output_path.mkdir(parents=True, exist_ok=True)

        # 3. Process each image: crop, resize, generate caption, and save.
        idx = 0
        for image_name in tqdm.tqdm(
            image_names,
            desc="Preprocessing images",
            unit="img",
            dynamic_ncols=True,
            leave=True,
            bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ):
            # 3.1. Load the image.
            image = self.get_image(image_name)

            # 3.2. Crop the image using the specified crop size.
            cropped_image = self.crop_image(image, crop_size)

            # 3.3. Resize the image using the specified strategy.
            resized_image = self.resize_image(cropped_image, image_size, strategy)

            # 3.4. Generate the corresponding attributes file name and load attributes.
            attributes_name = image_name.replace(
                f".{image_extension}",
                f".{attributes_extension}",
            )
            attributes = self.get_attributes(attributes_name)

            # 3.5. Generate a caption based on the loaded attributes.
            caption = self.generate_caption(attributes)

            # 3.6. Save the processed image and its caption to the output directory.
            num_digits = len(str(len(image_names)))
            output_image_path = (
                Path(output_dir)
                / f"{output_prefix}{idx:0{num_digits}d}.{image_extension}"
            )

            resized_image.save(output_image_path)

            output_caption_path = output_image_path.with_suffix(f".{caption_extension}")
            with open(output_caption_path, "w") as caption_file:
                caption_file.write(caption)

            idx += 1

        print(f"Processed {idx} images and captions. Saved to {output_dir}.")


if __name__ == "__main__":
    raw_data_path = "data/raw/cartoonset10k"
    mappings_path = "configs/attribute_mappings.yaml"
    crop_size = 64
    image_size = 32

    with open(mappings_path, "r") as file:
        mappings = yaml.safe_load(file)

    preprocessor = Preprocessor(raw_data_path, mappings)
    original_image, caption = preprocessor.get_random_image()
    cropped_image = preprocessor.crop_image(original_image)
    resized_image1 = preprocessor.resize_image(original_image, image_size, "nearest")
    resized_image2 = preprocessor.resize_image(original_image, image_size, "bilinear")
    resized_image3 = preprocessor.resize_image(original_image, image_size, "bicubic")

    figure, axes = plt.subplots(1, 4)
    axes[0].imshow(original_image)
    axes[0].set_title("original")
    axes[1].imshow(resized_image1)
    axes[1].set_title(f"nearest {image_size}")
    axes[2].imshow(resized_image2)
    axes[2].set_title(f"bilinear {image_size}")
    axes[3].imshow(resized_image3)
    axes[3].set_title(f"bicubic {image_size}")

    for axis in axes:
        axis.axis("off")

    wrapped_caption = textwrap.fill(caption, width=70)
    figure.text(0.5, 0.02, wrapped_caption, ha="center", va="bottom")
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    plt.show()
