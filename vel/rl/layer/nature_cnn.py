"""
Code based loosely on implementation:
https://github.com/openai/baselines/blob/master/baselines/ppo2/policies.py

Under MIT license.
"""
import numpy as np

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import vel.util.network as net_util

from vel.api import SizeHint, SizeHints
from vel.net.modular import Layer, LayerFactory


class NatureCnn(Layer):
    """ Neural network as defined in the paper 'Human-level control through deep reinforcement learning' """

    def __init__(self, name: str, input_width, input_height, input_channels, output_dim=512):
        super().__init__(name)

        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=(8, 8),
            stride=4
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(4, 4),
            stride=2
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=1
        )

        layer_series = [
            (8, 0, 4),
            (4, 0, 2),
            (3, 0, 1)
        ]

        self.final_width = net_util.convolutional_layer_series(input_width, layer_series)
        self.final_height = net_util.convolutional_layer_series(input_height, layer_series)

        self.linear_layer = nn.Linear(
            self.final_width * self.final_height * 64,  # 64 is the number of channels of the last conv layer
            self.output_dim
        )

    def reset_weights(self):
        """ Call proper initializers for the weights """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)

    def size_hints(self) -> SizeHints:
        return SizeHints(SizeHint(None, self.output_dim))

    def forward(self, image, state: dict = None, context: dict = None):
        result = image
        result = F.relu(self.conv1(result))
        result = F.relu(self.conv2(result))
        result = F.relu(self.conv3(result))
        flattened = result.view(result.size(0), -1)
        return F.relu(self.linear_layer(flattened))


class NatureCnnFactory(LayerFactory):
    """ Nature Cnn Network Factory """

    def __init__(self, output_dim: int = 512):
        self.output_dim = output_dim

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "nature_cnn"

    def instantiate(self, name: str, direct_input: SizeHints, context: dict) -> Layer:
        (b, c, w, h) = direct_input.assert_single(4)

        return NatureCnn(
            name=name,
            input_width=w,
            input_height=h,
            input_channels=c,
            output_dim=self.output_dim
        )


def create(output_dim=512):
    """ Vel factory function """
    return NatureCnnFactory(output_dim=output_dim)

