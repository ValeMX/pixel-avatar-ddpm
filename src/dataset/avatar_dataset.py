from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class AvatarDataset(Dataset):
    """Custom dataset for loading avatar images and their corresponding captions."""

    def __init__(
        self,
        data_path: str,
        image_extension: str = "png",
        caption_extension: str = "txt",
        transform=None,
    ):
        """Initialize the AvatarDataset.

        Args:
            data_path (str): Path to the directory containing images and captions.
            image_extension (str): Extension of the image files. Default is "png".
            caption_extension (str): Extension of the caption files. Default is "txt".
            transform (callable, optional): Optional transform to be applied on an image.
        """
        image_paths = [
            path
            for path in Path(data_path).iterdir()
            if path.is_file() and path.suffix.lower() == f".{image_extension}"
        ]

        caption_paths = [
            path
            for path in Path(data_path).iterdir()
            if path.is_file() and path.suffix.lower() == f".{caption_extension}"
        ]

        self.image_paths = image_paths
        self.caption_paths = caption_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get the image and caption paths for the given index
        image_path = self.image_paths[idx]
        caption_path = self.caption_paths[idx]

        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Load the caption
        with open(caption_path, "r") as f:
            caption = f.read().strip()

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, caption
