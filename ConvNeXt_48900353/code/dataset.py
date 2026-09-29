"""
File for dataset loading, loading images, connecting images to patients, creating patient splits, and applying preprocessing steps ready to be inputted to models.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# import constants from config file
from config import CLASS_NAMES, DATASET_SPLIT_NAMES, DATASET_ROOT, METADATA_PATH, SEED, LABEL_MAP

PATIENT_PATTERN = re.compile(r"ADNI_(\d{3}_S_\d{4})", re.IGNORECASE)


def patient_id_from_record(record):
    """Extract patient ID from a metadata record"""
    match = PATIENT_PATTERN.search(record.get("raw"))
    return match.group(1).upper()

def build_image_table(dataset_root, scan_to_patient):
    """Build a table containing image paths, classes, and patient IDs"""

    rows = []

    # iterate through all folders in dataset
    for split in DATASET_SPLIT_NAMES:
        for class_name in CLASS_NAMES:
            folder = dataset_root / split / class_name # get path to specific split folder

            for path in sorted(folder.rglob("*")): # get path to all images in folder
                scan_id = path.name.split("_", 1)[0] # get scan id from image file name

                # append image_path, class_name, and patient_id as row to table
                rows.append({
                    "image_path": str(path.resolve()),
                    "class_name": class_name,
                    "patient_id": scan_to_patient.get(scan_id),
                })

    # convert table to pandas dataframe and return
    table = pd.DataFrame(rows)
    return table

def create_splits(dataset_root, metadata_path, seed=SEED):
    """Create patient splits and print dataset summaries"""

    with Path(metadata_path).expanduser().open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    # create mapping of scan_id -> patient_id
    scan_to_patient = {
        str(scan_id): patient_id_from_record(record)
        for scan_id, record in metadata.items()
    }

    # create table where each row is exactly one unique patient
    dataset_df = build_image_table(dataset_root, scan_to_patient)
    patients = dataset_df[["patient_id", "class_name"]].drop_duplicates("patient_id")

    # split patients into 70% training and 30% for validation/testing
    train, remaining = train_test_split(
        patients,
        test_size=0.3,
        stratify=patients["class_name"],
        random_state=seed,
    )

    # split the remaining 30% into 20% validation and 10% testing
    val, test = train_test_split(
        remaining,
        test_size=1 / 3, # 10% of 30%
        stratify=remaining["class_name"],
        random_state=seed,
    )

    print(f"Total: {len(dataset_df)} images from {len(patients)} patients")
    splits = []

    # place images into their splits with their associated patient
    for name, group in [("Train", train), ("Validation", val), ("Test", test)]:
        frame = dataset_df[dataset_df["patient_id"].isin(group["patient_id"])].reset_index(drop=True)
        splits.append(frame)
        print(
            f"\n{name}: {len(frame)} images ({len(frame) / len(dataset_df):.2%}), "
            f"{len(group)} patients"
        )
        print(frame["class_name"].value_counts().to_string())

    return tuple(splits)

def preprocess_image(image):
    """Returns a preprocessed image as a (3, 256, 256) tensor"""

    image = image.convert("L")
    width, height = image.size

    # pad image with black pixels to upscale it to 256 x 256
    left = (256 - width) // 2
    top = (256 - height) // 2
    right = 256 - width - left
    bottom = 256 - height - top
    image = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)

    # copy grayscale values into each of the three RGB channels
    image = image.convert("RGB")

    image_array = np.asarray(image, dtype=np.float32) / 255.0 # convert to numpy array and normalise pixels value to 0 - 1
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1) # convert numpy array to tensor
    return image_tensor

def load_batch(batch):
    """Load and preprocess a batch of (image_path, class_name) pairs"""

    images = []
    labels = []

    for image_path, class_name in batch:
        with Image.open(image_path) as image:
            images.append(preprocess_image(image))
        labels.append(LABEL_MAP[class_name])

    return torch.stack(images), torch.tensor(labels, dtype=torch.long)

def create_dataloaders(dataset_root, metadata_path, batch_size=32, num_workers=0, seed=SEED):
    """Returns train, validation and test loaders that load images per batch"""

    # convert path strings to actual paths
    dataset_root = Path(dataset_root).expanduser()
    metadata_path = Path(metadata_path).expanduser()

    splits = create_splits(dataset_root, metadata_path, seed)
    loaders = []

    for split_index, frame in enumerate(splits):
            # store only image path and classes, load pixels when a batch is requested
            samples = list(zip(frame["image_path"], frame["class_name"]))
            loader = DataLoader(
                samples,
                batch_size=batch_size,
                shuffle=(split_index == 0),  # shuffle only the training split
                num_workers=num_workers,
                collate_fn=load_batch,
                generator=torch.Generator().manual_seed(seed), # allow reproducibility
            )

            loaders.append(loader)

    return tuple(loaders)

if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders(DATASET_ROOT, METADATA_PATH)
    images, labels = next(iter(train_loader))
    print(f"\nImage batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Pixel range: {images.min().item():.3f} to {images.max().item():.3f}")