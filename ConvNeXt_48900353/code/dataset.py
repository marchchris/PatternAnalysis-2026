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

