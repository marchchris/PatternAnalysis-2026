"""
ResNet-18 and ConvNeXt-Tiny implementations with PyTorch

Both models start with random weights and return raw classification logits.
Input: (batch_size, 1, 224, 224), for single-channel grayscale images.
Output: (batch_size, 2), for AD/NC classification.

Architecture references:
- ResNet: https://arxiv.org/abs/1512.03385
- ConvNeXt: https://arxiv.org/abs/2201.03545
"""

import torch
from torch import nn

class ResNetBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # first convolution
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # second convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # match the skip connection to the main branch when its shape changes
        self.downsample = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            # project the shortcut so it can be added to the main branch
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        # save the residual path before applying the main convolutions
        shortcut = self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        # combine both paths and apply the final activation
        return self.relu(x + shortcut)

class ResNet18(nn.Module):
    """ResNet-18 implementation of four stages with [2, 2, 2, 2] residual blocks"""

    def __init__(self, num_classes=2):
        super().__init__()

        # The stem extracts low-level features and reduces 224x224 inputs to 56x56.
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=7,
            stride=2, padding=3, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_stage(64, 64, stride=1)    # (64, 64, 64)
        self.layer2 = self._make_stage(64, 128, stride=2)   # (128, 32, 32)
        self.layer3 = self._make_stage(128, 256, stride=2)  # (256, 16, 16)
        self.layer4 = self._make_stage(256, 512, stride=2)  # (512, 8, 8)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _make_stage(in_channels, out_channels, stride):
        # only the first block changes the spatial size or channel count
        return nn.Sequential(
            ResNetBlock(in_channels, out_channels, stride=stride),
            ResNetBlock(out_channels, out_channels),
        )

    @staticmethod
    def _init_weights(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(
                layer.weight, mode="fan_out", nonlinearity="relu",
            )
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        # The first stage keeps the resolution; later stages downsample it.
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


class LayerNorm2d(nn.Module):
    """Normalize channels at each spatial position of a tensor"""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x):
        # layerNorm operates on the last dimension, so temporarily move C last
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)  # NHWC -> NCHW

class DropPath(nn.Module):
    """Randomly drop a samples residual branch during training"""

    def __init__(self, probability=0.0):
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("Drop-path probability must be in [0, 1).")
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0.0:
            return x

        keep_probability = 1.0 - self.probability
        # one random decision per image
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(mask_shape).bernoulli_(keep_probability)
        return x * mask / keep_probability

class ConvNeXtBlock(nn.Module):
    """Depthwise convolution, channel expansion, and a residual connection"""

    def __init__(self, channels, drop_path=0.0):
        super().__init__()

        # groups=channels applies a separate spatial filter to each channel
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels,
        )
        self.norm = nn.LayerNorm(channels, eps=1e-6)

        # applied to NHWC tensors these linear layers act like 1x1 convolutions
        self.expand = nn.Linear(channels, 4 * channels)
        self.activation = nn.GELU()
        self.project = nn.Linear(4 * channels, channels)

        # one learnable scale per channel, initialized close to zero
        self.layer_scale = nn.Parameter(torch.full((channels,), 1e-6))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        shortcut = x
        x = self.depthwise(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        x = self.project(self.activation(self.expand(x)))
        x = x * self.layer_scale
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return shortcut + self.drop_path(x)

class ConvNeXt(nn.Module):
    """ConvNeXt with depths [3, 3, 9, 3] and widths [96, 192, 384, 768].
    drop_path_rate is the maximum stochastic-depth probability, it increases linearly across the 18 blocks.
    """

    def __init__(self, num_classes=2, drop_path_rate=0.1):
        super().__init__()

        depths = (3, 3, 9, 3)
        widths = (96, 192, 384, 768)

        # the stem reduces spatial size by four 224x224 -> 56x56.
        self.stem = nn.Sequential(
            nn.Conv2d(1, widths[0], kernel_size=4, stride=4),
            LayerNorm2d(widths[0]),
        )

        # stage output shapes (C, H, W) for 224x224 inputs:
        # (96, 56, 56), (192, 28, 28), (384, 14, 14), (768, 7, 7)
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        block_index = 0
        total_blocks = sum(depths)

        for stage_index in range(4):
            channels = widths[stage_index]
            blocks = []

            for _ in range(depths[stage_index]):
                probability = drop_path_rate * block_index / (total_blocks - 1)
                blocks.append(ConvNeXtBlock(channels, drop_path=probability))
                block_index += 1

            self.stages.append(nn.Sequential(*blocks))

            if stage_index < 3:
                # Normalize, halve H/W, and double the channel count.
                self.downsamples.append(nn.Sequential(
                    LayerNorm2d(channels),
                    nn.Conv2d(
                        channels, widths[stage_index + 1],
                        kernel_size=2, stride=2,
                    ),
                ))

        self.norm = nn.LayerNorm(widths[-1], eps=1e-6)
        self.head = nn.Linear(widths[-1], num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.stem(x)
        for stage_index in range(4):
            x = self.stages[stage_index](x)
            if stage_index < 3:
                x = self.downsamples[stage_index](x)

        x = x.mean(dim=(2, 3))  # global average pooling: (N, C, H, W) -> (N, C)
        x = self.norm(x)
        return self.head(x)

def build_model(model_name, num_classes=2, *, drop_path_rate=0.1):
    """Build a randomly initialized ResNet or ConvNeXt model"""

    if model_name == "resnet18":
        return ResNet18(num_classes=num_classes)

    if model_name == "convnext":
        return ConvNeXt(
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

    raise ValueError("model_name must be 'resnet18' or'convnext'.")