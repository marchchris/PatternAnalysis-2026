"""Loads trained models and run inference on hold out images.

Run from the code/ directory:
python predict.py --model [model_name] --weights [saved_model_path]
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch

from config import DATASET_ROOT, LABEL_MAP, METADATA_PATH, SEED
from dataset import create_splits, preprocess_image
from modules import build_model

def load_model(model_name, weights_path, device):
    """Load the saved model"""
    model = build_model(model_name, num_classes=len(LABEL_MAP))

    state_dict = torch.load(
        Path(weights_path).expanduser(), map_location="cpu", weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["resnet18", "convnext"], required=True)
    parser.add_argument("--weights", required=True, help="Path to the saved .pth file")
    args = parser.parse_args()

    if args.num_images < 1:
        parser.error("--num-images must be at least 1")

    # use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the saved model
    model = load_model(args.model, args.weights, device)
    index_to_class = {index: name for name, index in LABEL_MAP.items()}
    print(f"Model: {args.model} | Device: {device}")