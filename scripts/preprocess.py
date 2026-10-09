import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.dataset.preprocessor import Preprocessor
from src.utils.helpers import load_config, load_mappings

if __name__ == "__main__":
    config = load_config()
    mappings = load_mappings()

    if config is None:
        print("Error: Configuration file not found or invalid.")
        sys.exit(1)

    if mappings is None:
        print("Error: Mappings file not found or invalid.")
        sys.exit(1)

    raw_data_path = config["dataset"]["raw_dataset_10k_path"]
    processed_data_path = config["dataset"]["processed_dataset_path"]
    image_extension = config["dataset"]["image_extension"]
    attributes_extension = config["dataset"]["attributes_extension"]
    caption_extension = config["dataset"]["caption_extension"]
    crop_size = config["dataset"]["crop_size"]
    image_size = config["dataset"]["image_size"]
    strategy = config["dataset"]["strategy"]

    preprocessor = Preprocessor(data_path=raw_data_path, mappings=mappings)

    preprocessor.preprocess_dataset(
        output_dir=processed_data_path,
        image_extension=image_extension,
        attributes_extension=attributes_extension,
        caption_extension=caption_extension,
        crop_size=crop_size,
        image_size=image_size,
        strategy=strategy,
    )
