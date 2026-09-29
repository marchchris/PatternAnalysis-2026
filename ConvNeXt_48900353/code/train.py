import time
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import nn

from config import LABEL_MAP, SEED
from dataset import create_dataloaders
from modules import build_model

MODEL_NAME = "convnext"
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

def run_epoch(model, loader, criterion, device, optimizer=None):
    """Train when an optimizer is provided, otherwise evaluate the model"""

    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    # validation and testing do not need gradient calculations
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            # move images and labels to GPU
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images) # forward pass
            loss = criterion(outputs, labels) # calculate loss

            # update weights if training model
            if training:
                loss.backward()
                optimizer.step()

            # weight by batch size so final batch is counted correctly
            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_images += labels.size(0)

    return total_loss / total_images, total_correct / total_images