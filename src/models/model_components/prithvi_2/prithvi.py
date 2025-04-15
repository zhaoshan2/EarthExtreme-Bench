from functools import partial
from typing import List, Tuple, Union
from pathlib import Path
import sys

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import to_2tuple

from .decoder import MAEDecoder
from .encoder import PrithviViT


class PrithviMAE(nn.Module):
    """ Prithvi Masked Autoencoder"""

    def __init__(self,
                 img_size: Union[int, Tuple[int, int]] = 224,
                 patch_size: Union[int, Tuple[int, int, int]] = (1, 16, 16),
                 num_frames: int = 3,
                 in_chans: int = 3,
                 embed_dim: int = 1024,
                 depth: int = 24,
                 num_heads: int = 16,
                 decoder_embed_dim: int = 512,
                 decoder_depth: int = 8,
                 decoder_num_heads: int = 16,
                 mlp_ratio: float = 4.,
                 norm_layer: nn.Module = partial(torch.nn.LayerNorm, eps=1e-6),
                 norm_pix_loss: bool = False,
                 coords_encoding:  Union[List[str] , None] = None,
                 coords_scale_learn: bool = False,
                 encoder_only: bool = False,
                 out_indices: List = [5, 11, 17, 23],
                 **kwargs,
                 ):
        super().__init__()

        self.encoder = PrithviViT(
            img_size=img_size,
            num_frames=num_frames,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            coords_encoding=coords_encoding,
            coords_scale_learn=coords_scale_learn,
            out_indices=out_indices
        )

        self.encoder_only = encoder_only

        if not encoder_only:
            self.decoder = MAEDecoder(
                patch_size=patch_size,
                grid_size=self.encoder.patch_embed.grid_size,
                in_chans=in_chans,
                encoder_embed_dim=embed_dim,
                decoder_embed_dim=decoder_embed_dim,
                depth=decoder_depth,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                coords_encoding=coords_encoding,
                coords_scale_learn=coords_scale_learn,
            )
        else:
            self.decoder = nn.Identity()

        self.norm_pix_loss = norm_pix_loss

    def patchify(self, pixel_values):
        """
        Args:
            pixel_values (torch.FloatTensor of shape `(batch_size, num_channels, time, height, width)`):
                Pixel values.
        Returns:
            torch.FloatTensor of shape `(batch_size, num_patches, patch_size[0]*patch_size[1]*patch_size[2] * num_channels)`:
                Patchified pixel values.
        """
        patch_size_t, patch_size_h, patch_size_w = self.encoder.patch_embed.patch_size
        num_channels = self.encoder.in_chans

        # patchify
        patchified_pixel_values = rearrange(pixel_values, 'b c (t s) (h p) (w q) -> b (t h w) (s p q c)',
                                            c=num_channels, s=patch_size_t, p=patch_size_h, q=patch_size_w)


        return patchified_pixel_values

    def unpatchify(self, patchified_pixel_values, image_size:  Union[Tuple[int, int] , None] = None):
        """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape
                `(batch_size, num_patches, patch_size[0]*patch_size[1]*patch_size[2] * num_channels)`:
                Patchified pixel values.
            image_size (`Tuple[int, int]`, *optional*):
                Original image size.
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                Pixel values.
        """
        patch_size_t, patch_size_h, patch_size_w = self.encoder.patch_embed.patch_size
        image_size = to_2tuple(image_size) if image_size is not None else self.encoder.img_size
        original_height, original_width = image_size
        num_patches_h = original_height // patch_size_h
        num_patches_w = original_width // patch_size_w
        num_channels = self.encoder.in_chans

        pixel_values = rearrange(patchified_pixel_values, 'b (t h w) (s p q c) -> b c (t s) (h p) (w q)',
                                 c=num_channels, h=num_patches_h, w=num_patches_w,
                                 s=patch_size_t, p=patch_size_h, q=patch_size_w)
        return pixel_values

    def forward_loss(self, pixel_values, pred, mask):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, time, height, width)`):
                Pixel values.
            pred (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size[0]*patch_size[1]*patch_size[2] * num_channels)`:
                Predicted pixel values.
            mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).
        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss.
        """
        target = self.patchify(pixel_values)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(
        self,
        pixel_values: torch.Tensor,
        temporal_coords: Union[None, torch.Tensor] = None,
        location_coords: Union[None, torch.Tensor] = None,
        mask_ratio: float = 0.75
    ):
        if len(pixel_values.shape) == 4 and self.encoder.patch_embed.input_size[0] == 1:
            # add time dim
            pixel_values = pixel_values.unsqueeze(2)

        latent, mask, ids_restore = self.encoder(pixel_values, temporal_coords, location_coords, mask_ratio)
        pred = self.decoder(latent, ids_restore, temporal_coords, location_coords, input_size=pixel_values.shape)
        loss = self.forward_loss(pixel_values, pred, mask)
        return loss, pred, mask

    def forward_features(
        self,
        x: torch.Tensor,
        temporal_coords: Union[None, torch.Tensor] = None,
        location_coords: Union[None, torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        return self.encoder.forward_features(x, temporal_coords, location_coords)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PrithviMAE(
        img_size = 224,
        patch_size = (1, 16, 16),
        num_frames = 3,
        in_chans = 3,
        embed_dim  = 1024,
        depth = 24,
        num_heads = 16,
        decoder_embed_dim = 512,
        decoder_depth = 8,
        decoder_num_heads = 16,
        mlp_ratio = 4.,
        norm_layer = partial(torch.nn.LayerNorm, eps=1e-6),
        norm_pix_loss = False,
        coords_encoding = None,
        coords_scale_learn = False,
        encoder_only = False,
    )
    model = model.to(device)
    input = torch.randn((1, 6, 1, 224, 224)).to(device)
    model.forward(input)
