import time
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.transforms import Normalize

from config import LABEL_MAP, SEED
from dataset import create_dataloaders
from modules import build_model

MODEL_NAME = "convnext"
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
