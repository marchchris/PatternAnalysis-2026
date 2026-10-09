"""
File for dataset loading, loading images, connecting images to patients, creating patient splits, 
and applying preprocessing steps ready to be inputted to models.

Images are collected from each dataset split and class folder, then each scan ID is 
matched to its patient ID using themetadata file. The resulting image table is split 
into training, validation, and test sets by patient, ensuring that images from the same 
patient remain in only one split.

"""

import json
import re
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms import v2

from config import (
    CLASS_NAMES,
    DATASET_SPLIT_NAMES,
    DATASET_ROOT,
    METADATA_PATH,
    SEED,
    LABEL_MAP,
)

PATIENT_PATTERN = re.compile(r"ADNI_(\d{3}_S_\d{4})", re.IGNORECASE)
IMAGE_SIZE = (224, 224)

def normalise_images(image):
    """Normalise images to mean 0 and standard deviation 1"""
    values = TF.pil_to_tensor(image.convert("L")).to(torch.float32)
    mean = values.mean()
    std = values.std(correction=0)
    values.sub_(mean).div_(std + 1e-5)

    minimum, maximum = torch.aminmax(values)
    values.sub_(minimum).div_(maximum - minimum + 1e-5).mul_(255)
    return TF.to_pil_image(values.to(torch.uint8))


TRAIN_TRANSFORM = v2.Compose([
    v2.Lambda(normalise_images),
    v2.Resize(IMAGE_SIZE),
    v2.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.9, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.2),
    # v2.RandomRotation(degrees=10),
    v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),

    v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
    v2.ColorJitter(brightness=0.2, contrast=0.2),

    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),

    v2.RandomErasing(p=0.25),


    v2.Normalize(mean=[0.5], std=[0.5]),
])

EVAL_TRANSFORM = v2.Compose([
    v2.Lambda(normalise_images),
    v2.Resize(IMAGE_SIZE),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5], std=[0.5]),
])


def patient_id_from_record(record):
    """Extract a patient ID from a metadata record"""
    match = PATIENT_PATTERN.search(record.get("raw") or "")
    return match.group(1).upper()


def build_image_table(dataset_root, scan_to_patient):
    """Build a table containing image paths, classes, and patient IDs"""
    dataset_root = Path(dataset_root).expanduser().resolve()
    rows = []

    # iterate through all folders in dataset
    for split in DATASET_SPLIT_NAMES:
        for class_name in CLASS_NAMES:
            folder = dataset_root / split / class_name

             # get paths to all images in folder
            paths = sorted(
                path for path in folder.rglob("*")
                if path.suffix.lower() == ".jpeg" and path.is_file()
            )

            # append (image path, class, patient id)
            for path in paths:
                scan_id = path.name.split("_", 1)[0]
                patient_id = scan_to_patient.get(scan_id)
                rows.append((str(path), class_name, patient_id))

    return pd.DataFrame.from_records(rows, columns=["image_path", "class_name", "patient_id"])


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

    # split patients into 70% training and 30% for validation/testing
    patients = dataset_df[["patient_id", "class_name"]].drop_duplicates("patient_id")
    train, remaining = train_test_split(
        patients,
        test_size=0.3,
        stratify=patients["class_name"],
        random_state=seed,
    )
    val, test = train_test_split(
        remaining,
        test_size=1 / 3, # one third of the remaining 30% is 10% overall
        stratify=remaining["class_name"],
        random_state=seed,
    )

    # count all images
    image_count = len(dataset_df)
    print(f"Total: {image_count} images from {len(patients)} patients")
    splits = []

    # build each split from patient IDs so images from one patient stay together
    for name, group in (("Train", train), ("Validation", val), ("Test", test)):
        frame = dataset_df.loc[
            dataset_df["patient_id"].isin(group["patient_id"])
        ].reset_index(drop=True)
        splits.append(frame)

        # print image and patient counts, and class distributions
        print(
            f"\n{name}: {len(frame)} images ({len(frame) / image_count:.2%}), "
            f"{len(group)} patients"
        )
        print(frame["class_name"].value_counts().to_string())

    return tuple(splits)


def preprocess_image(image, augment=False):
    """Returns an optionally augmented float32 tensor with shape (1, 224, 224)."""
    transform = TRAIN_TRANSFORM if augment else EVAL_TRANSFORM
    return transform(image)


def load_batch(batch, augment=False):
    """Load and preprocess (image_path, class_name) pairs."""

    # use TRAIN_TRANSFORM for training set only
    transform = TRAIN_TRANSFORM if augment else EVAL_TRANSFORM

    images = []
    labels = []

    # build batches
    for image_path, class_name in batch:
        with Image.open(image_path) as image:
            images.append(transform(image))
        labels.append(LABEL_MAP[class_name])

    
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def create_dataloaders(
    dataset_root,
    metadata_path,
    batch_size=32,
    num_workers=0,
    seed=SEED,
):
    """Return train, validation, and test loaders"""

    splits = create_splits(dataset_root, metadata_path, seed)

    loaders = []
    for split_index, frame in enumerate(splits):
        samples = list(frame[["image_path", "class_name"]].itertuples(index=False, name=None))
        is_training = split_index == 0
        loaders.append(DataLoader(
            samples,
            batch_size=batch_size,
            shuffle=is_training,
            num_workers=num_workers,
            collate_fn=partial(load_batch, augment=is_training),
            generator=torch.Generator().manual_seed(seed),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        ))
    return tuple(loaders)


def show_examples(images, labels, count=6):
    """Display a sample of images and their class labels from a batch"""

    import matplotlib.pyplot as plt

    count = min(count, images.size(0))

    images = images[:count].detach().cpu()
    labels = labels[:count].detach().cpu()
    
    columns = min(count, 3)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False
    )
    axes = axes.ravel()

    for index in range(count):
        axes[index].imshow(images[index, 0].numpy(), cmap="gray")
        axes[index].set_title(CLASS_NAMES[labels[index].item()])
        axes[index].axis("off")
    for axis in axes[count:]:
        axis.axis("off")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders(
        DATASET_ROOT, METADATA_PATH
    )
    images, labels = next(iter(train_loader))
    print(f"\nImage batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Pixel range: {images.min().item():.3f} to {images.max().item():.3f}")
    show_examples(images, labels)
