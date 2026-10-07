"""
File for dataset loading, loading images, connecting images to patients, creating patient splits, and applying preprocessing steps ready to be inputted to models.
"""

import json
import re
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageStat
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import v2

# import constants from config file
from config import CLASS_NAMES, DATASET_SPLIT_NAMES, DATASET_ROOT, METADATA_PATH, SEED, LABEL_MAP

PATIENT_PATTERN = re.compile(r"ADNI_(\d{3}_S_\d{4})", re.IGNORECASE)

IMAGE_SIZE = (224, 224)
RESIZE_IMAGE = v2.Resize(IMAGE_SIZE)

TRAIN_AUGMENTATION = v2.Compose([
    v2.RandomResizedCrop(
        size=IMAGE_SIZE,
        scale=(0.8, 1.0),
        ratio=(0.95, 1.05),
    ),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomAffine(
        degrees=10,
        translate=(0.15, 0.15),
        scale=(0.9, 1.1),
    ),
])

IMAGE_RESIZE = v2.Compose([
    RESIZE_IMAGE,
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])


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
                if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
                    continue
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
        test_size=1 / 3, # one third of the remaining 30% = 10% overall
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

def calculate_image_statistics(train_frame):
    """Calculate mean and standard deviation of training set images"""

    pixel_sum = 0.0
    pixel_squared_sum = 0.0
    pixel_count = 0

    for index, image_path in enumerate(train_frame["image_path"], start=1):
        with Image.open(image_path) as image:
            # resize images
            image = RESIZE_IMAGE(image.convert("L"))
            statistics = ImageStat.Stat(image)

        pixel_sum += statistics.sum[0]
        pixel_squared_sum += statistics.sum2[0]
        pixel_count += statistics.count[0]

    # convert the mean/std from [0, 255] to [0, 1].
    mean_255 = pixel_sum / pixel_count
    variance_255 = max(pixel_squared_sum / pixel_count - mean_255**2, 0.0)
    mean = mean_255 / 255.0
    std = variance_255**0.5 / 255.0

    return mean, std


def create_preprocessing(mean, std):
    """Create preprocessing that resizes image and normalises"""

    return v2.Compose([
        IMAGE_RESIZE,
        v2.Normalize(mean=(mean,) * 3, std=(std,) * 3),
    ])


def preprocess_image(image, augment=False, mean=None, std=None):
    """Return an optionally augmented image as a normalized (3, 224, 224) tensor"""

    image = image.convert("L")

    if augment:
        image = TRAIN_AUGMENTATION(image)

    # copy grayscale values into each of the three RGB channels
    image = image.convert("RGB")
    return create_preprocessing(mean, std)(image)


def denormalize_image(image, mean, std):
    """Convert a normalized image tensor back to a displayable [0, 1] tensor"""

    mean = torch.tensor(mean, dtype=image.dtype, device=image.device)
    std = torch.tensor(std, dtype=image.dtype, device=image.device)
    mean = mean.view(1, 1, 1)
    std = std.view(1, 1, 1)
    return (image * std + mean).clamp(0, 1)

def load_batch(batch, augment=False, mean=None, std=None):
    """Load and preprocess a batch of (image_path, class_name) pairs"""

    images = []
    labels = []

    for image_path, class_name in batch:
        with Image.open(image_path) as image:
            images.append(preprocess_image(image, augment=augment, mean=mean, std=std))
        labels.append(LABEL_MAP[class_name])

    return torch.stack(images), torch.tensor(labels, dtype=torch.long)

def create_dataloaders(dataset_root, metadata_path, batch_size=32, num_workers=0, seed=SEED):
    """Returns train, validation and test loaders that load images per batch"""

    # convert path strings to actual paths
    dataset_root = Path(dataset_root).expanduser()
    metadata_path = Path(metadata_path).expanduser()

    splits = create_splits(dataset_root, metadata_path, seed)
    mean, std = calculate_image_statistics(splits[0])
    loaders = []

    for split_index, frame in enumerate(splits):
        # Store only paths and classes; load pixels when a batch is requested.
        samples = list(zip(frame["image_path"], frame["class_name"]))
        loader = DataLoader(
            samples,
            batch_size=batch_size,
            shuffle=(split_index == 0),  # shuffle only the training split
            num_workers=num_workers,
            collate_fn=partial(
                load_batch,
                augment=(split_index == 0),
                mean=mean,
                std=std,
            ),
            generator=torch.Generator().manual_seed(seed), # allow reproducibility
        )

        loaders.append(loader)

    return tuple(loaders)

def show_examples(images, labels, mean, std, count=6):
    """Display a few images and their class labels from a batch"""

    count = min(count, images.size(0))

    # arrange up to three images per row
    columns = min(count, 3)
    rows = (count + columns - 1) // columns

    # create one subplot for each position in the image grid
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    axes = axes.ravel()

    for index in range(count):
        display_image = denormalize_image(images[index], mean, std)
        axes[index].imshow(display_image.permute(1, 2, 0).numpy())

        # use the label to retrieve and display the class name
        axes[index].set_title(CLASS_NAMES[labels[index].item()])
        axes[index].axis("off")

    # hide any unused subplot positions in the grid
    for axis in axes[count:]:
        axis.axis("off")

    figure.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders(DATASET_ROOT, METADATA_PATH)
    # Reuse the statistics already calculated when creating the loaders.
    mean = train_loader.collate_fn.keywords["mean"]
    std = train_loader.collate_fn.keywords["std"]
    images, labels = next(iter(train_loader))
    print(f"\nImage batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Pixel range: {images.min().item():.3f} to {images.max().item():.3f}")
    show_examples(images, labels, mean, std)
