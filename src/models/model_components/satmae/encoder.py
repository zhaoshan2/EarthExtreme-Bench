import torch.nn as nn
from .training_utils import get_activation, get_normalization, SE_Block


class CoreCNNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        norm="batch",
        activation="relu",
        padding="same",
        residual=True,
    ):
        super(CoreCNNBlock, self).__init__()
        self.activation = get_activation(activation)
        self.residual = residual
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.squeeze = SE_Block(self.out_channels)
        self.match_channels = nn.Identity()
        if in_channels != out_channels:
            self.match_channels = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, padding=0, bias=False
                ),
                get_normalization(norm, out_channels),
            )
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, padding=0)
        self.norm1 = get_normalization(norm, self.out_channels)
        self.conv2 = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            3,
            padding=self.padding,
            groups=self.out_channels,
        )
        self.norm2 = get_normalization(norm, self.out_channels)

        self.conv3 = nn.Conv2d(
            self.out_channels, self.out_channels, 3, padding=self.padding, groups=1
        )
        self.norm3 = get_normalization(norm, self.out_channels)

    def forward(self, x):
        identity = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))
        x = x * self.squeeze(x)
        if self.residual:
            x = x + self.match_channels(identity)
        x = self.activation(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        out_channels,
        norm="batch",
        activation="relu",
        padding="same",
    ):
        super(EncoderBlock, self).__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.norm = norm
        self.padding = padding

        self.blocks = []
        for i in range(self.depth):
            _in_channels = self.in_channels if i == 0 else self.out_channels
            block = CoreCNNBlock(
                _in_channels,
                self.out_channels,
                norm=self.norm,
                activation=self.activation,
                padding=self.padding,
            )

            self.blocks.append(block)

        self.blocks = nn.Sequential(*self.blocks)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        for i in range(self.depth):
            x = self.blocks[i](x)

        x = self.downsample(x)

        return x
