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
from config import CLASS_NAMES, DATASET_SPLIT_NAMES

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
                scan_id = str(path).split("/")[-1].split("_")[0] # get scan id from image file name

                # append image_path, class_name, and patient_id as row to table
                rows.append({
                    "image_path": str(path.resolve()),
                    "class_name": class_name,
                    "patient_id": scan_to_patient.get(scan_id),
                })

    # convert table to pandas dataframe and return
    table = pd.DataFrame(rows)
    return table