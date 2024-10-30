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


class DecoderBlock(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        out_channels,
        *,
        norm="batch",
        activation="relu",
        padding="same",
    ):
        super(DecoderBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_blocks = activation
        self.activation = get_activation(activation)
        self.norm = norm
        self.padding = padding
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.match_channels = CoreCNNBlock(
            self.in_channels,
            self.out_channels,
            norm=self.norm,
            activation=self.activation_blocks,
            padding=self.padding,
        )

        self.blocks = []
        for _ in range(self.depth):
            block = CoreCNNBlock(
                self.out_channels,
                self.out_channels,
                norm=self.norm,
                activation=self.activation_blocks,
                padding=self.padding,
            )
            self.blocks.append(block)
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        x = self.upsample(x)
        x = self.match_channels(x)
        for i in range(self.depth):
            x = self.blocks[i](x)
        return x


class CoreDecoder(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim=10,
        output_dim=1,
        depths=None,
        dims=None,
        activation="relu",
        norm="batch",
        padding="same",
    ):
        super(CoreDecoder, self).__init__()
        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.dims = [96, 192, 384, 768] if dims is None else dims
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.activation = activation
        self.norm = norm
        self.padding = padding
        self.decoder_blocks = []
        assert len(self.depths) == len(
            self.dims
        ), "depths and dims must have the same length."
        for i in reversed(range(len(self.depths))):
            decoder_block = DecoderBlock(
                self.depths[i],
                self.dims[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
                norm=norm,
                activation=activation,
                padding=padding,
            )
            self.decoder_blocks.append(decoder_block)
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)
        self.decoder_downsample_block = nn.Identity()
        self.decoder_bridge = nn.Sequential(
            CoreCNNBlock(
                embedding_dim,
                self.dims[-1],
                norm=norm,
                activation=activation,
                padding=padding,
            ),
        )
        self.decoder_head = nn.Sequential(
            CoreCNNBlock(
                self.dims[0],
                self.dims[0],
                norm=norm,
                activation=activation,
                padding=padding,
            ),
            nn.Conv2d(self.dims[0], self.output_dim, kernel_size=1, padding=0),
        )

    def forward_decoder(self, x):
        for block in self.decoder_blocks:
            x = block(x)
        return x

    def forward(self, x):
        x = self.decoder_bridge(x)
        x = self.forward_decoder(x)
        x = self.decoder_head(x)
        return x
