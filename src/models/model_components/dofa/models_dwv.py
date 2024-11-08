# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from .wave_dynamic_layer import Dynamic_MLP_OFA, Dynamic_MLP_Decoder

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from .decoder import CoreDecoder


class OFAViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=512,
        patch_size=16,
        drop_rate=0.0,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        wv_planes=128,
        num_classes=45,
        global_pool=False,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        wave_list=[0.665, 0.56, 0.49],
    ):
        super().__init__()

        self.wv_planes = wv_planes
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)

        self.patch_embed = Dynamic_MLP_OFA(
            wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim
        )
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.wave_list = wave_list

    def forward_features(self, x):
        # embed patches
        wave_list = self.wave_list
        wavelist = torch.tensor(wave_list, device=x.device).float()
        self.waves = wavelist

        x, _ = self.patch_embed(x, self.waves)

        x = x + self.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits=False):
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)  # (1, 1025, 768)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            x = self.fc_norm(x)
        else:
            x = x[:, 0]

        x = self.forward_head(x)
        return x


def vit_small_patch16(wave_list, img_size, **kwargs):
    model = OFAViT(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        wave_list=wave_list,
        **kwargs
    )
    return model


def vit_base_patch16(wave_list, img_size, **kwargs):
    model = OFAViT(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        wave_list=wave_list,
        **kwargs
    )
    return model


def vit_large_patch16(wave_list, img_size, **kwargs):
    model = OFAViT(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        wave_list=wave_list,
        **kwargs
    )
    return model


def vit_huge_patch14(wave_list, img_size, **kwargs):
    model = OFAViT(
        img_size=img_size,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        wave_list=wave_list,
        **kwargs
    )
    return model


class Dofa(nn.Module):

    def __init__(
        self,
        wave_list=[0.665, 0.56, 0.49],
        img_size=512,
        output_dim=1,
        decoder_norm="batch",
        decoder_padding="same",
        decoder_activation="relu",
        decoder_depths=[2, 2, 8, 2],
        decoder_dims=[160, 320, 640, 1280],
        **kwargs
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # encoder specifics
        self.vit_encoder = vit_base_patch16(wave_list=wave_list, img_size=img_size)
        # self.vit_encoder = DOFAViT(
        #     img_size=img_size,
        #     patch_size=16,
        #     wave_list=wave_list,
        #     embed_dims=768,
        #     num_layers=12,
        #     num_heads=12,
        #     mlp_ratio=4,
        # )

        # --------------------------------------------------------------------------
        # CNN Decoder Blocks:
        self.depths = decoder_depths
        self.dims = decoder_dims
        self.output_dim = output_dim

        self.decoder_head = CoreDecoder(
            embedding_dim=768,
            output_dim=output_dim,
            depths=decoder_depths,
            dims=decoder_dims,
            activation=decoder_activation,
            padding=decoder_padding,
            norm=decoder_norm,
        )

        self.decoder_downsample_block = nn.Identity()

    def reshape(self, x):
        # Separate channel axis
        N, L, D = x.shape
        x = x.permute(0, 2, 1)
        x = x.view(N, D, int(L**0.5), int(L**0.5))

        return x

    def forward(self, x):
        x = self.vit_encoder.forward_features(x)  # (1, 1025, 768)

        # remove cls token
        x = x[:, 1:, :]
        # # reshape into 2d features
        x = self.reshape(x)
        x = self.decoder_downsample_block(x)
        x = self.decoder_head(x)
        return x


if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Dofa(wave_list=[0.665, 0.56, 0.49]).to(device)
    # The model accepts remote sensing data in a video format (B, C, H, W)
    x = torch.randn(1, 3, 512, 512)
    x = x.to(device)
    y = model.forward(x)
    print("output", y.shape)  # (1,1,512,512)
