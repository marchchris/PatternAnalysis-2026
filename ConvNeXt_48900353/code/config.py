"""Config file for storing paths, defining constants and hyperparameters."""

from pathlib import Path

# paths for ADNI dataset
DATASET_ROOT = Path("~/Documents/Datasets/ADNI/AD_NC").expanduser()
METADATA_PATH = Path("~/Documents/Datasets/ADNI/meta_data_with_label.json").expanduser()

# constants for files
SEED = 42
CLASS_NAMES = ["AD", "NC"]
LABEL_MAP = {"NC" : 0, "AD" : 1} # label 0 = Normal Congnitive Ability, label 1 = Alzheimer's Disease
DATASET_SPLIT_NAMES = ["train", "test"]


