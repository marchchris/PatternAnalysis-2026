"""Config file for storing paths, defining constants and hyperparameters."""

# paths for ADNI dataset
DATASET_ROOT = "~/Documents/Datasets/ADNI/AD_NC"
METADATA_PATH = "~/Documents/Datasets/ADNI/meta_data_with_label.json"

# constants for files
SEED = 42
CLASS_NAMES = ["NC", "AD"]
LABEL_MAP = {"NC" : 0, "AD" : 1} # label 0 = Normal Congnitive Ability, label 1 = Alzheimer's Disease
DATASET_SPLIT_NAMES = ["train", "test"]


