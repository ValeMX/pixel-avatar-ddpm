import csv
import json
import random
import shutil
from typing import Any
import torch

from collections import Counter
from itertools import combinations
from pathlib import Path


def clean_directory(directory_path: str) -> None:
    """Clean the specified directory by removing all files and subdirectories."""
    directory = Path(directory_path)
    if directory.exists():
        for file in directory.iterdir():
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                for subfile in file.iterdir():
                    subfile.unlink()
                file.rmdir()
        directory.rmdir()
    directory.mkdir(parents=True, exist_ok=True)


def build_ordinary_split(
    data_path: str,
    split_path: str,
    generator: torch.Generator,
    data_ratio: float = 1.0,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    image_extension: str = "png",
    caption_extension: str = "txt",
):
    """Build an ordinary split of the dataset.

    Args:
        data_path (str): Path to the dataset directory containing images and captions.
        split_path (str): Path to the directory where the split data will be saved.
        generator (torch.Generator): A PyTorch random number generator for reproducibility.
        data_ratio (float, optional): Ratio of the dataset to use. Default is 1.0 (use the entire dataset).
        train_ratio (float, optional): Ratio of the training set. Default is 0.8 (80% for training).
        validation_ratio (float, optional): Ratio of the validation set. Default is 0.1 (10% for validation).
        image_extension (str, optional): Extension of image files. Default is "png".
        caption_extension (str, optional): Extension of caption files. Default is "txt".
    """
    # Get all image and caption files in the dataset directory
    image_files = list(Path(data_path).glob(f"*.{image_extension}"))
    caption_files = list(Path(data_path).glob(f"*.{caption_extension}"))

    # Shuffle the files to ensure randomness with a fixed seed for reproducibility
    image_files.sort()
    caption_files.sort()

    permutation = torch.randperm(len(image_files), generator=generator).tolist()

    image_files = [image_files[i] for i in permutation]
    caption_files = [caption_files[i] for i in permutation]

    # Split the data according to the specified ratios
    split_image_files = image_files[: int(len(image_files) * data_ratio)]
    split_caption_files = caption_files[: int(len(caption_files) * data_ratio)]

    # Further split the data into train, validation, and test sets
    train_size = int(len(split_image_files) * train_ratio)
    val_size = int(len(split_image_files) * validation_ratio)

    train_image_files = split_image_files[:train_size]
    val_image_files = split_image_files[train_size : train_size + val_size]
    test_image_files = split_image_files[train_size + val_size :]

    train_caption_files = split_caption_files[:train_size]
    val_caption_files = split_caption_files[train_size : train_size + val_size]
    test_caption_files = split_caption_files[train_size + val_size :]

    # Clean the split directory before saving new splits
    clean_directory(split_path)

    # Copy the split image and caption files to the split directory
    split_dir = Path(split_path)
    split_dir.mkdir(parents=True, exist_ok=True)

    for img in train_image_files:
        (split_dir / "train").mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / img.name).write_bytes(img.read_bytes())

    for img in val_image_files:
        (split_dir / "validation").mkdir(parents=True, exist_ok=True)
        (split_dir / "validation" / img.name).write_bytes(img.read_bytes())

    for img in test_image_files:
        (split_dir / "test").mkdir(parents=True, exist_ok=True)
        (split_dir / "test" / img.name).write_bytes(img.read_bytes())

    for cap in train_caption_files:
        (split_dir / "train").mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / cap.name).write_text(cap.read_text())

    for cap in val_caption_files:
        (split_dir / "validation").mkdir(parents=True, exist_ok=True)
        (split_dir / "validation" / cap.name).write_text(cap.read_text())

    for cap in test_caption_files:
        (split_dir / "test").mkdir(parents=True, exist_ok=True)
        (split_dir / "test" / cap.name).write_text(cap.read_text())

    print(
        f"Ordinary data split completed. ",
        f"Train: {len(train_image_files)}, ",
        f"Validation: {len(val_image_files)}, ",
        f"Test: {len(test_image_files)}\n",
        f"Split data saved in: {split_dir.resolve()}",
        sep="",
    )


def load_metadata(
    metadata_path: str,
    string_columns: set[str] = {"image"},
) -> list[dict]:
    """Load the metadata from a CSV file and return it as a list of dictionaries.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        string_columns (set[str], optional): Set of column names that should be treated as strings.

    Returns:
        A list of dictionaries representing the metadata rows.

    Raises:
        ValueError: If the metadata file is empty or invalid.
    """
    with open(metadata_path, newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"Metadata file {metadata_path} is empty or invalid.")

    return [
        {k: (v if k in string_columns else int(v)) for k, v in row.items()}
        for row in rows
    ]


def resolve_combinations(combos: list[dict], mappings: dict) -> list[tuple]:
    """Resolve the held-out combinations into a list of tuples of (attribute, value_id).

    Args:
        combos (list[dict]): List of dictionaries representing the held-out combinations.
        mappings (dict): Dictionary mapping attributes to their corresponding value IDs.

    Returns:
        A list of tuples representing the resolved held-out combinations.

    Raises:
        ValueError: If an attribute or value in the combinations is unknown or invalid.
    """
    resolved = []
    for combo in combos:
        pair = []
        for i in (1, 2):
            attribute, value = combo[f"attribute_{i}"], combo[f"value_{i}"]

            if attribute not in mappings:
                raise ValueError(
                    f"Unknown attribute '{attribute}' in combination {combo}. "
                    f"Valid attributes: {sorted(mappings.keys())}"
                )

            ids = [k for k, v in mappings[attribute].items() if v == value]

            if len(ids) != 1:
                raise ValueError(
                    f"Value '{value}' is not valid (or ambiguous) for '{attribute}'. "
                    f"Valid values: {list(mappings[attribute].values())}"
                )

            pair.append((attribute, ids[0]))

        if pair[0][0] == pair[1][0]:
            raise ValueError(
                f"Both attributes in combination {combo} are the same: '{pair[0][0]}'. "
                f"Attributes must be different."
            )

        resolved.append(tuple(pair))
    return resolved


def is_held_out(row: dict, resolved: list[tuple]) -> bool:
    """Check if a given row contains any of the held-out combinations.

    Args:
        row (dict): A dictionary representing a single row of metadata.
        resolved (list[tuple]): A list of tuples representing the resolved held-out combinations.

    Returns:
        True if the row contains any held-out combination, False otherwise.
    """
    # Check if the row contains any of the held-out combinations
    # by comparing the attribute-value pairs in the row with those in the resolved combinations.
    return any(row[a1] == v1 and row[a2] == v2 for (a1, v1), (a2, v2) in resolved)


def get_text(mappings: dict, attribute: str, value_id: int) -> str:
    """Retrieve the text representation of a value ID for a given attribute.

    Args:
        mappings (dict): A dictionary mapping attributes to their corresponding value IDs.
        attribute (str): The attribute name.
        value_id (int): The value ID for which to retrieve the text representation.

    Returns:
        The text representation of the value ID for the specified attribute.
    """
    return mappings[attribute].get(value_id, str(value_id))


def create_compositional_split(
    rows: list[dict],
    resolved: list[tuple],
    mappings: dict,
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> tuple[dict[str, list[str]], dict]:
    """Build a compositional split of the data.

    Args:
        rows (list[dict]): List of dictionaries representing the data.
        resolved (list[tuple]): List of tuples representing the resolved held-out combinations.
        mappings (dict): Dictionary mapping attributes to their corresponding value IDs.
        seed (int): Random seed for reproducibility. Default is 42.
        train_ratio (float): Ratio of training data. Default is 0.8 (80% for training).
        validation_ratio (float): Ratio of validation data. Default is 0.1 (10% for validation).
        min_single_count (int): Minimum number of times a single value must appear in the training set. Default is 30.

    Returns:
        A tuple containing the splits and the report.

    Raises:
        ValueError: If the input data is invalid or if the split cannot be constructed.
    """

    if not resolved:
        raise ValueError("No held-out combinations specified.")

    ood_rows = [r for r in rows if is_held_out(r, resolved)]
    pool_rows = [r for r in rows if not is_held_out(r, resolved)]

    # Generate a list of stems (image names without extensions) and shuffle them using the provided seed for reproducibility.
    stems = sorted(r["image"] for r in pool_rows)
    random.Random(seed).shuffle(stems)

    # Split the stems into training, validation, and test sets based on the specified ratios.
    train_size = int(len(stems) * train_ratio)
    val_size = int(len(stems) * validation_ratio)

    splits = {
        "train": stems[:train_size],
        "validation": stems[train_size : train_size + val_size],
        "test": stems[train_size + val_size :],
        "ood": sorted(r["image"] for r in ood_rows),
    }

    by_name = {r["image"]: r for r in rows}
    train_rows = [by_name[s] for s in splits["train"]]

    combo_reports = []
    for (a1, v1), (a2, v2) in resolved:
        n_ood = sum(1 for r in ood_rows if r[a1] == v1 and r[a2] == v2)
        n_t1 = sum(1 for r in train_rows if r[a1] == v1)
        n_t2 = sum(1 for r in train_rows if r[a2] == v2)

        if n_ood == 0:
            raise ValueError(
                f"Combination {a1}={get_text(mappings, a1, v1)} + {a2}={get_text(mappings, a2, v2)} "
                f"does not appear in ood: cannot create a compositional split."
            )

        combo_reports.append(
            {
                "attribute_1": a1,
                "value_1": get_text(mappings, a1, v1),
                "attribute_2": a2,
                "value_2": get_text(mappings, a2, v2),
                "n_ood_images": n_ood,
                "n_train_images_with_value_1": n_t1,
                "n_train_images_with_value_2": n_t2,
            }
        )

    report = {
        "seed": seed,
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
        "counts": {name: len(files) for name, files in splits.items()},
        "held_out_combinations": combo_reports,
    }

    return splits, report


def find_candidate_combinations(
    rows: list[dict],
    mappings: dict,
    attributes: list[str],
    min_pair: int = 60,
    max_pair: int = 200,
    min_single: int = 300,
) -> list[dict]:
    """Find candidate combinations of attribute-value pairs that meet specified frequency criteria.

    Args:
        rows (list[dict]): List of dictionaries representing the data.
        mappings (dict): Dictionary mapping attributes to their corresponding value IDs.
        attributes (list[str]): List of attribute names to consider for combinations.
        min_pair (int): Minimum frequency for a pair of attribute-value combinations to be considered. Default is 60.
        max_pair (int): Maximum frequency for a pair of attribute-value combinations to be considered. Default is 200.
        min_single (int): Minimum frequency for a single attribute-value combination to be considered. Default is 300.

    Returns:
        A list of dictionaries representing candidate combinations that meet the specified criteria.
    """
    # Count the occurrences of each single attribute-value pair in the dataset.
    singles = {a: Counter(r[a] for r in rows) for a in attributes}

    # Find candidate combinations of attribute-value pairs that meet the specified frequency criteria.
    candidates = []
    for a1, a2 in combinations(attributes, 2):
        # Count the occurrences of each pair of attribute-value combinations in the dataset.
        for (v1, v2), n in Counter((r[a1], r[a2]) for r in rows).items():
            if (
                min_pair <= n <= max_pair
                and singles[a1][v1] >= min_single
                and singles[a2][v2] >= min_single
            ):
                candidates.append(
                    {
                        "attribute_1": a1,
                        "value_1": get_text(mappings, a1, v1),
                        "attribute_2": a2,
                        "value_2": get_text(mappings, a2, v2),
                        "n_pair": n,
                        "n_value_1": singles[a1][v1],
                        "n_value_2": singles[a2][v2],
                    }
                )
    return sorted(
        candidates, key=lambda c: (c["attribute_1"], c["attribute_2"], c["n_pair"])
    )


def write_split(
    data_path: str,
    split_path: str,
    splits: dict[str, list[str]],
    report: dict,
    image_extension: str = "png",
    caption_extension: str = "txt",
) -> None:
    """Write the split data (images and captions) to the specified split directory and save the split definition as a JSON file.

    Args:
        data_path (str): Path to the dataset directory containing images and captions.
        split_path (str): Path to the directory where the split data will be saved.
        splits (dict[str, list[str]]): Dictionary containing the splits (train, validation, test, ood).
        report (dict): Dictionary containing information about the split (e.g., counts, held-out combinations).
        image_extension (str, optional): Extension of image files. Default is "png".
        caption_extension (str, optional): Extension of caption files. Default is "txt".
    """
    split_dir = Path(split_path)
    if split_dir.exists() and any(split_dir.iterdir()):
        shutil.rmtree(split_dir)

    for name, stems in splits.items():
        out_dir = split_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            for ext in (image_extension, caption_extension):
                shutil.copy2(
                    Path(data_path) / f"{stem}.{ext}", out_dir / f"{stem}.{ext}"
                )

    with open(split_dir / "split.json", "w") as file:
        json.dump({**report, "files": splits}, file, indent=2)


def build_compositional_split(
    data_path: str,
    split_path: str,
    mappings: dict[str, Any],
    combos: list[dict],
    seed: int = 42,
    metadata_name: str = "metadata.csv",
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    image_extension: str = "png",
    caption_extension: str = "txt",
) -> dict:
    """Build a compositional split of the dataset based on held-out combinations of attribute-value pairs.

    Args:
        data_path (str): Path to the dataset directory containing images and captions.
        split_path (str): Path to the directory where the split data will be saved.
        mappings (dict): Dictionary mapping attributes to their corresponding value IDs.
        combos (list[dict]): List of dictionaries representing the held-out combinations.
        seed (int, optional): Random seed for reproducibility. Default is 42.
        metadata_name (str, optional): Name of the metadata CSV file. Default is "metadata.csv".
        train_ratio (float, optional): Ratio of training data. Default is 0.8 (80% for training).
        validation_ratio (float, optional): Ratio of validation data. Default is 0.1 (10% for validation).
        image_extension (str, optional): Extension of image files. Default is "png".
        caption_extension (str, optional): Extension of caption files. Default is "txt".

    Returns:
        A dictionary containing information about the split (e.g., counts, held-out combinations).
    """
    # 1. Load the metadata from the specified CSV file in the dataset directory.
    rows = load_metadata(data_path + "/" + metadata_name)

    # 2. Resolve the held-out combinations into a list of tuples of (attribute, value_id) using the provided mappings.
    resolved = resolve_combinations(combos, mappings)

    # 3. Create the compositional split of the data based on the resolved held-out combinations and specified ratios.
    splits, report = create_compositional_split(
        rows,
        resolved,
        mappings,
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )

    # 4. Write the split data (images and captions) to the specified split directory and save the split definition as a JSON file.
    write_split(
        data_path,
        split_path,
        splits,
        report,
        image_extension,
        caption_extension,
    )

    # 5. Print a summary of the split information, including the number of images in each split and details about the held-out combinations.
    print(f"Split summary for {split_path}:", report["counts"], "images")
    for c in report["held_out_combinations"]:
        print(
            f"  held-out: {c['attribute_1']}={c['value_1']} + {c['attribute_2']}={c['value_2']} "
            f"-> {c['n_ood_images']} images in ood, "
            f"{c['n_train_images_with_value_1']} images in train with {c['attribute_1']}={c['value_1']}, "
            f"{c['n_train_images_with_value_2']} images in train with {c['attribute_2']}={c['value_2']}"
        )

    return report
