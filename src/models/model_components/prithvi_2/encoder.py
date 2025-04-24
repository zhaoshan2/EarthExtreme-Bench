import sys
from typing import Union, Tuple, List, Optional
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from timm.layers import to_2tuple
from einops import rearrange
import numpy as np

from .embed import PatchEmbed, get_3d_sincos_pos_embed, _get_1d_sincos_embed_from_grid_torch

def _init_weights(module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

class TemporalEncoder(nn.Module):
    def __init__(self, embed_dim: int, trainable_scale: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.year_embed_dim = embed_dim // 2
        self.julian_day_embed_dim = embed_dim - self.year_embed_dim

        # If trainable, initialize scale with small number
        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer('scale', torch.ones(1))

    def forward(self, temporal_coords: torch.Tensor, tokens_per_frame: Union[int , None] = None):
        """
        temporal_coords: year and day-of-year info with shape (B, T, 2).
        tokens_per_frame: number of tokens for each frame in the sample. If provided, embeddings will be
            repeated over T dimension, and final shape is (B, T*tokens_per_frame, embed_dim).
        """
        shape = temporal_coords.shape[:2] + (-1,)  # B, T, -1

        year = _get_1d_sincos_embed_from_grid_torch(
            self.year_embed_dim, temporal_coords[:, :, 0].flatten()).reshape(shape)
        julian_day = _get_1d_sincos_embed_from_grid_torch(
            self.julian_day_embed_dim, temporal_coords[:, :, 1].flatten()).reshape(shape)

        embedding = self.scale * torch.cat([year, julian_day], dim=-1)

        if tokens_per_frame is not None:
            embedding = torch.repeat_interleave(embedding, tokens_per_frame, dim=1)

        return embedding  # B, T*tokens_per_frame, embed_dim


class LocationEncoder(nn.Module):
    def __init__(self, embed_dim: int, trainable_scale: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.lat_embed_dim = embed_dim // 2
        self.lon_embed_dim = embed_dim - self.lat_embed_dim

        # If trainable, initialize scale with small number
        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer('scale', torch.ones(1))

    def forward(self, location_coords: torch.Tensor):
        """
        location_coords: lat and lon info with shape (B, 2).
        """
        shape = location_coords.shape[:1] + (1, -1)  # B, 1, -1

        lat = _get_1d_sincos_embed_from_grid_torch(
                self.lat_embed_dim, location_coords[:, 0].flatten()).reshape(shape)
        lon = _get_1d_sincos_embed_from_grid_torch(
                self.lon_embed_dim, location_coords[:, 1].flatten()).reshape(shape)

        embedding = self.scale * torch.cat([lat, lon], dim=-1)

        return embedding  # B, 1, embed_dim

class PrithviViT(nn.Module):
    """ Prithvi ViT Encoder"""
    def __init__(self,
                 img_size: Union[int, Tuple[int, int]] = 224,
                 patch_size: Union[int, Tuple[int, int, int]] = (1, 16, 16),
                 num_frames: int = 1,
                 in_chans: int = 3,
                 embed_dim: int = 1024,
                 depth: int = 24,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.,
                 norm_layer: nn.Module = partial(torch.nn.LayerNorm, eps=1e-6),
                 coords_encoding: Union[List[str], None] = None,
                 coords_scale_learn: bool = False,
                 encoder_only: bool = True,  # needed for timm
                 out_indices: List = [5, 11, 17, 23],
                 ** kwargs,
                ):
        super().__init__()

        self.feature_info = []
        self.encoder_only = encoder_only
        self.in_chans = in_chans
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.img_size = to_2tuple(img_size)
        self.out_indices = out_indices
        if isinstance(patch_size, int):
            patch_size = (1, patch_size, patch_size)
        self.patch_size = patch_size

        # 3D patch embedding
        self.patch_embed = PatchEmbed(
            input_size=(num_frames,) + self.img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Optional temporal and location embedding
        coords_encoding = coords_encoding or []
        self.temporal_encoding = 'time' in coords_encoding
        self.location_encoding = 'location' in coords_encoding
        if self.temporal_encoding:
            assert patch_size[0] == 1, f"With temporal encoding, patch_size[0] must be 1, received {patch_size[0]}"
            self.temporal_embed_enc = TemporalEncoder(embed_dim, coords_scale_learn)
        if self.location_encoding:
            self.location_embed_enc = LocationEncoder(embed_dim, coords_scale_learn)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer("pos_embed", torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))

        # Transformer layers
        self.blocks = []
        for i in range(depth):
            self.blocks.append(Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
            self.feature_info.append(
                {"num_chs": embed_dim * self.patch_embed.patch_size[0], "reduction": 1, "module": f"blocks.{i}"}
            )
        self.blocks = nn.ModuleList(self.blocks)

        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, add_cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(_init_weights)

    def random_masking(self, sequence, mask_ratio, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.
        Args:
            sequence (`torch.FloatTensor` of shape `(batch_size, sequence_length, dim)`)
            mask_ratio (float): mask ratio to use.
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def _get_pos_embed(self, x):
        t, h, w = x.shape[-3:]

        pos_embed = torch.from_numpy(get_3d_sincos_pos_embed(
            self.embed_dim,
            (
                t // self.patch_embed.patch_size[0],
                h // self.patch_embed.patch_size[1],
                w // self.patch_embed.patch_size[2],
            ),
            add_cls_token=True,
        )).float().unsqueeze(0).to(x)

        return pos_embed


    def forward(
        self, x: torch.Tensor,
        temporal_coords: Union[None, torch.Tensor] = None,
        location_coords: Union[None, torch.Tensor] = None,
        mask_ratio=0.75
    ):
        if x.shape[-3:] != self.patch_embed.input_size:
            # changed input size
            pos_embed = self._get_pos_embed(x)
        else:
            pos_embed = self.pos_embed

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]

        if self.temporal_encoding:
            num_tokens_per_frame = x.shape[1] // self.num_frames
            temporal_encoding = self.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
            x = x + temporal_encoding
        if self.location_encoding:
            location_encoding = self.location_embed_enc(location_coords)
            x = x + location_encoding

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_features(
            self,
            x: torch.Tensor,
            temporal_coords: Optional[torch.Tensor] = None,
            location_coords: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:

        if len(x.shape) == 4 and self.patch_embed.input_size[0] == 1:
            # add time dim
            x = x.unsqueeze(2)

        if x.shape[-3:] != self.patch_embed.input_size:
            pos_embed = self._get_pos_embed(x)
        else:
            pos_embed = self.pos_embed

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]

        if self.temporal_encoding:
            num_tokens_per_frame = x.shape[1] // self.patch_embed.num_frames
            temporal_encoding = self.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
            x = x + temporal_encoding
        if self.location_encoding:
            location_encoding = self.location_embed_enc(location_coords)
            x = x + location_encoding

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        out = []
        for i, block in enumerate(self.blocks): # 24 blocks
            x = block(x) # [b,_, embed_dim]: (1, 197, 1024)
            out.append(x.clone())

        x = self.norm(x)
        out[-1] = x

        return out

    def prepare_features_for_image_model(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        out = []
        effective_time_dim = self.patch_embed.input_size[0] // self.patch_embed.patch_size[0]

        for i, x in enumerate(features):
            if i in self.out_indices:
                x_no_token = x[:, 1:, :]
                number_of_tokens = x_no_token.shape[1]
                tokens_per_timestep = number_of_tokens // effective_time_dim
                h = int(np.sqrt(tokens_per_timestep))
                encoded = rearrange(
                    x_no_token,
                    "batch (t h w) e -> batch (t e) h w",
                    e=self.embed_dim,
                    t=effective_time_dim,
                    h=h,
                ) # (b,c*t,h,h) [1, 1024, 14, 14] or [1, 2014, 14, 14] for 2 frames
                out.append(encoded)
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PrithviViT(
        img_size=224,
        patch_size=(1, 16, 16),
        num_frames=2,
        in_chans=6,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        coords_encoding=None,
        coords_scale_learn=False,
        encoder_only=True,  # needed for timm
        out_indices=[5, 11, 17, 23],
    )
    model = model.to(device)
    input = torch.randn((1, 6, 2, 224, 224)).to(device)
    features = model.forward_features(input)
    features_for_img = model.prepare_features_for_image_model(features)