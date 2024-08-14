import torch
import math
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """LeNet model for MNIST."""

    def __init__(self, num_classes=10):
        """Initialize LeNet.

        Args:
            num_classes: Number of classes. Default is 10 for MNIST.
        """
        super().__init__()

        self.net = nn.Sequential(
            torch.nn.Conv2d(1, 6, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 4 * 4, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.net(x)
