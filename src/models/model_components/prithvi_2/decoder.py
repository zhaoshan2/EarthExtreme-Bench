import sys
from typing import List, Tuple, Union

import torch.nn as nn
import torch
from timm.models.vision_transformer import Block

from .training_utils import get_activation, get_normalization, SE_Block
from .encoder import TemporalEncoder, LocationEncoder
from .embed import get_3d_sincos_pos_embed

def _init_weights(module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

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

class MAEDecoder(nn.Module):
    """ Transformer Decoder used in the Prithvi MAE"""
    def __init__(self,
                 patch_size: Union[int, Tuple[int, int, int]] = (1, 16, 16),
                 grid_size: Union[List[int], Tuple[int, int, int]] = (3, 14, 14),
                 in_chans: int = 3,
                 encoder_embed_dim: int = 1024,
                 decoder_embed_dim: int = 512,
                 depth: int = 8,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.,
                 norm_layer: nn.Module = nn.LayerNorm,
                 coords_encoding: Union[List[str], None] = None,
                 coords_scale_learn: bool = False,
                 ):
        super().__init__()

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_dim = decoder_embed_dim
        self.grid_size = grid_size
        if isinstance(patch_size, int):
            patch_size = (1, patch_size, patch_size)
        self.patch_size = patch_size
        self.num_frames = self.grid_size[0] * patch_size[0]
        num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        # Optional temporal and location embedding
        coords_encoding = coords_encoding or []
        self.temporal_encoding = 'time' in coords_encoding
        self.location_encoding = 'location' in coords_encoding
        if self.temporal_encoding:
            self.temporal_embed_dec = TemporalEncoder(decoder_embed_dim, coords_scale_learn)
        if self.location_encoding:
            self.location_embed_dec = LocationEncoder(decoder_embed_dim, coords_scale_learn)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.register_buffer("decoder_pos_embed", torch.zeros(1, num_patches + 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim,
                                      patch_size[0] * patch_size[1] * patch_size[2] * in_chans,
                                      bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_3d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], self.grid_size, add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(_init_weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        ids_restore: torch.Tensor,
        temporal_coords: None | torch.Tensor = None,
        location_coords: None | torch.Tensor = None,
        input_size: list[int] = None,
    ):
        # embed tokens
        x = self.decoder_embed(hidden_states)

        t, h, w = input_size[-3:]
        decoder_pos_embed = torch.from_numpy(
            get_3d_sincos_pos_embed(
                self.decoder_embed_dim,
                (
                    t // self.patch_size[0],
                    h // self.patch_size[1],
                    w // self.patch_size[2],
                ),
                add_cls_token=True,
            )
        ).to(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # unshuffle
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]).to(x_.device))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # add pos embed
        x = x + decoder_pos_embed

        # remove cls token
        x_ = x[:, 1:, :]

        if self.temporal_encoding:
            num_tokens_per_frame = x_.shape[1] // self.num_frames
            temporal_encoding = self.temporal_embed_dec(temporal_coords, num_tokens_per_frame)
            # Add temporal encoding w/o cls token
            x_ = x_ + temporal_encoding
        if self.location_encoding:
            location_encoding = self.location_embed_dec(location_coords)
            # Add location encoding w/o cls token
            x_ = x_ + location_encoding

        # append cls token
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # apply Transformer layers (blocks)
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)

        # predictor projection
        pred = self.decoder_pred(x)

        # remove cls token
        pred = pred[:, 1:, :]

        return pred