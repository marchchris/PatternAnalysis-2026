"""
This file provides a single function for creating randomly initialized ResNet-18 or ConvNeXt-Tiny models and 
replacing their final classification layers with the specified number of output classes.
"""

import torch.nn as nn
from torchvision.models import convnext_tiny, resnet18

def build_model(model_name, num_classes):
    """Build a classification model with a output layer with `num_classes` outputs.

    Args:
        model_name: Model architecture to create: ``"resnet18"`` or
            ``"convnext"``.
        num_classes: Number of output classes.

    Returns:
        A randomly initialized model configured for ``num_classes`` outputs.

    Raises:
        ValueError: If ``model_name`` is not supported.
    """
    if model_name == "resnet18":
        model = resnet18(weights=None) # create non pretrained resnet18 model

        # replace final classifcation layer
        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )

    elif model_name == "convnext":
        model = convnext_tiny(weights=None) # create non pretrained convnext_tiny model

        # replace only the final linear layer
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes,
        )

    else:
        # raise ValueError if passed model_name is not supported
        raise ValueError("model name must be 'resnet18' or 'convnext'.")

    return model