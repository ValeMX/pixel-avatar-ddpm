import glob
import shutil
from pathlib import Path
import pandas as pd


def extract_samples(
    data_dir="data/raw/cartoonset10k",
    output_dir="data/samples_by_attribute",
    target_features=None,
    no_glasses_val=11,
):
    """Find and extract sample images for each value of the specified features from the dataset."""

    if target_features is None:
        raise ValueError("target_features must be provided as a list of feature names.")

    data_path = Path(data_dir)
    out_path = Path(output_dir)

    # Create output directories for each target feature
    for feat in target_features:
        (out_path / feat).mkdir(parents=True, exist_ok=True)

    # Find all CSV files in the dataset directory
    csv_files = list(data_path.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}. Please check the dataset path.")
        return

    print(f"Found {len(csv_files)} CSV files. Starting scan to extract images...")

    found_values = {feat: set() for feat in target_features}

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, header=None, names=["feature", "num", "num_max"])
        except Exception:
            continue

        # Check if the corresponding image file exists
        img_path = csv_path.with_suffix(".png")
        if not img_path.exists():
            continue

        row_dict = dict(zip(df["feature"].str.strip(), df["num"]))

        for feat in target_features:
            if feat in row_dict:
                val = int(row_dict[feat])

                # For glasses_color, we only select avatars WITH glasses (glasses != 11)
                if feat == "glasses_color" or feat == "glasses":
                    glasses_val = int(row_dict.get("glasses", -1))
                    if glasses_val == no_glasses_val:
                        continue
                else:
                    glasses_val = int(row_dict.get("glasses", -1))
                    if glasses_val != no_glasses_val:
                        continue

                if val not in found_values[feat]:
                    found_values[feat].add(val)

                    dest_path = out_path / feat / f"{feat}_val_{val}.png"

                    # Copy the image to the destination path
                    shutil.copyfile(img_path, dest_path)
                    print(f" Saved example for [{feat} = {val}] -> {dest_path}")

    print("\n Extraction completed!")
    for feat in target_features:
        print(f"Feature '{feat}': Found values -> {sorted(found_values[feat])}")


if __name__ == "__main__":
    DATASET_PATH = "data/raw/cartoonset10k"
    OUTPUT_PATH = "data/samples_by_attribute"

    artwork = [
        "chin_length",
        "eye_angle",
        "eye_lashes",
        "eye_lid",
        "eyebrow_shape",
        "eyebrow_weight",
        "face_shape",
        "facial_hair",
        "glasses",
        "hair",
    ]

    colors = [
        "eye_color",
        "face_color",
        "glasses_color",
        "hair_color",
    ]

    proportions = [
        "eye_eyebrow_distance",
        "eye_slant",
        "eyebrow_thickness",
        "eyebrow_width",
    ]

    extract_samples(
        data_dir=DATASET_PATH, output_dir=OUTPUT_PATH, target_features=artwork
    )
