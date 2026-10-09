"""
File for dataset loading, loading images, connecting images to patients, creating patient splits, 
and applying preprocessing steps ready to be inputted to models.

Images are collected from each dataset split and class folder, then each scan ID is
extracted from the filename before its slice index. The resulting image table is split
into training, validation, and test sets by scan, ensuring that slices from the same
scan remain in only one split.

"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import v2

from config import (
    CLASS_NAMES,
    DATASET_SPLIT_NAMES,
    DATASET_ROOT,
    SEED,
    LABEL_MAP,
)

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
    # v2.Lambda(normalise_images),
    # v2.Resize(IMAGE_SIZE),
    # v2.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.9, 1.0)),
    # v2.RandomHorizontalFlip(p=0.5),
    # v2.RandomVerticalFlip(p=0.2),
    # # v2.RandomRotation(degrees=10),
    # v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),

    # v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
    # v2.ColorJitter(brightness=0.2, contrast=0.2),

    # v2.ToImage(),
    # v2.ToDtype(torch.float32, scale=True),

    # v2.RandomErasing(p=0.25),


    # v2.Normalize(mean=[0.5], std=[0.5]),

    v2.Lambda(normalise_images),
    v2.Resize((256, 256)),
    v2.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomAffine(
        degrees=15, translate=(0.1, 0.1), scale=(0.9, 1)
    ),
    v2.ColorJitter(
        brightness=0.1, contrast=0.15,
    ),
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5], std=[0.5]),
])

EVAL_TRANSFORM = v2.Compose([
    v2.Lambda(normalise_images),
    v2.Resize(IMAGE_SIZE),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5], std=[0.5]),
])


def scan_id_from_filename(path):
    """Extract the scan ID from a filename containing a slice index."""
    parts = Path(path).stem.split("_")
    return parts[0]


def build_image_table(dataset_root):
    """Build a table containing image paths, classes, and scan IDs."""
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

            # append (image path, class, scan id)
            for path in paths:
                scan_id = scan_id_from_filename(path)
                rows.append((str(path), class_name, scan_id))

    return pd.DataFrame.from_records(
        rows, columns=["image_path", "class_name", "scan_id"]
    )


def create_splits(dataset_root, seed=SEED):
    """Create patient splits and print dataset summaries"""
    # create table where each row is exactly one image with its scan ID
    dataset_df = build_image_table(dataset_root)

    # split scans into 70% training and 30% for validation/testing
    scans = dataset_df[["scan_id", "class_name"]].drop_duplicates("scan_id")
    train, remaining = train_test_split(
        scans,
        test_size=0.3,
        stratify=scans["class_name"],
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
    print(f"Total: {image_count} images from {len(scans)} scans")
    splits = []

    # build each split from scan IDs so all slices from one scan stay together
    for name, group in (("Train", train), ("Validation", val), ("Test", test)):
        frame = dataset_df.loc[
            dataset_df["scan_id"].isin(group["scan_id"])
        ].reset_index(drop=True)
        splits.append(frame)

        # print image and patient counts, and class distributions
        print(
            f"\n{name}: {len(frame)} images ({len(frame) / image_count:.2%}), "
            f"{len(group)} scans"
        )
        print(frame["class_name"].value_counts().to_string())

    return tuple(splits)


def preprocess_image(image, augment=False):
    """Returns an optionally augmented float32 tensor with shape (1, 224, 224)."""
    transform = TRAIN_TRANSFORM if augment else EVAL_TRANSFORM
    return transform(image)


class ADNIDataset(Dataset):
    """Dataset for loading and preprocessing a split of ADNI images"""

    def __init__(self, frame, augment=False):
        self.samples = list(
            frame[["image_path", "class_name"]].itertuples(
                index=False, name=None
            )
        )
        self.transform = TRAIN_TRANSFORM if augment else EVAL_TRANSFORM

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, class_name = self.samples[index]
        with Image.open(image_path) as image:
            image_tensor = self.transform(image)
        label = torch.tensor(LABEL_MAP[class_name], dtype=torch.long)
        return image_tensor, label


def create_dataloaders(dataset_root, batch_size=32, num_workers=0, seed=SEED,):
    """Return train, validation, and test loaders"""

    splits = create_splits(dataset_root, seed)

    datasets = []
    for split_index, frame in enumerate(splits):
        is_training = split_index == 0
        datasets.append(ADNIDataset(frame, augment=is_training))

    loaders = []
    for split_index, dataset in enumerate(datasets):
        loaders.append(DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_index == 0, # only shuffle training set
            num_workers=num_workers,
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
        DATASET_ROOT
    )
    images, labels = next(iter(train_loader))
    print(f"\nImage batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Pixel range: {images.min().item():.3f} to {images.max().item():.3f}")
    show_examples(images, labels)
