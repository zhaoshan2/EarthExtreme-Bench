import sys
import warnings
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import PrithviViT
from mmseg.models.necks import Feature2Pyramid
from mmseg.models.decode_heads import UPerHead, FCNHead


def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=True,
):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class UperNet(torch.nn.Module):
    def __init__(self, backbone, neck, decode_head, aux_head):
        super(UperNet, self).__init__()
        self.encoder = backbone
        self.neck = neck
        self.decode_head = decode_head
        self.aux_head = aux_head

    def forward(self, x, temporal_coords, location_coords):
        feat = self.encoder.forward_features(x, temporal_coords, location_coords)
        feat = self.encoder.prepare_features_for_image_model(feat)
        # for i, f in enumerate(feat):
        #     print(i, f.shape) #0, (2,1024, 14, 14) 1, #(2,1024, 14, 14)...
        feat = self.neck(feat)
        out = self.decode_head(feat)
        out = resize(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return out

def vit_base_patch16(img_size, patch_size, in_chans, num_frames, embed_dim, **kwargs):
    model = PrithviViT(
        img_size=img_size,
        patch_size=(1, patch_size, patch_size),
        num_frames=num_frames,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        mlp_ratio=4.,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        coords_encoding=None,
        coords_scale_learn=False,
        encoder_only=True,  # needed for timm
        out_indices=[5, 11, 17, 23],
        **kwargs
    )

    return model

class Prithvi(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=3,
        in_chans=3,
        output_dim=1,
        embed_dim=1024,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # encoder specifics
        self.encoder = vit_base_patch16(img_size, patch_size, in_chans, num_frames, embed_dim)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # --------------------------------------------------------------------------
        self.output_dim = output_dim
        edim = self.encoder.embed_dim

        self.neck = Feature2Pyramid(
            embed_dim=edim,
            rescales=[4, 2, 1, 0.5],
        )

        self.decoder = UPerHead(
            in_channels=[edim, edim, edim, edim],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=output_dim,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
            ),
        )
        self.aux_head = FCNHead(
            in_channels=edim,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=output_dim,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        )

        self.seg_model = UperNet(
            self.encoder, self.neck, self.decoder, self.aux_head
        )

    def forward(self, x):
        temporal_coords, location_coords = None, None
        if isinstance(x, Tuple):
            x, temporal_coords, location_coords= x[0], x[1], x[2]
        if x.dim() == 4:  # if input has no T dim, expand it
            x = x[:, :, None, :, :]

        x = self.seg_model(x, temporal_coords, location_coords)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Prithvi(
        img_size=224,
        num_frames=2,
        in_chans=6,
        output_dim=1,
        embed_dim=1024,
    )
    model = model.to(device)
    input = torch.randn((2, 6, 2, 512, 512)).to(device)
    out = model.forward(input)
    print(out.shape) # ([2,1,512,512]): this can't output temporal information
