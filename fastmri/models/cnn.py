"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    """
    PyTorch implementation of a CNN model.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 64,
        num_layers: int = 5,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_layers = num_layers
        self.drop_prob = drop_prob

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_chans, chans, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Dropout2d(drop_prob),
           )
        for _ in range(num_layers - 2):
            self.conv_layers.append(nn.Conv2d(chans, chans, kernel_size=1, padding=0, bias=False))
            self.conv_layers.append(nn.InstanceNorm2d(out_chans))
            self.conv_layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            # self.conv_layers.append(nn.Dropout2d(drop_prob))
        self.conv_layers.append(nn.Conv2d(chans, out_chans, kernel_size=1, padding=0, bias=False))
        self.conv_layers.append(nn.Sigmoid())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.conv_layers(image)
