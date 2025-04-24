import sys
import torch
import torch.nn as nn

from pathlib import Path

from .decoder import CoreDecoder
from .encoder import PrithviEncoder


class Prithvi(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=3,
        tubelet_size=1,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        output_dim=1,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        decoder_norm="batch",
        decoder_padding="same",
        decoder_activation="relu",
        decoder_depths=[2, 2, 8, 2],
        decoder_dims=[160, 320, 640, 1280],
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # encoder specifics
        self.vit_encoder = PrithviEncoder(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            output_dim=output_dim,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            norm_pix_loss=norm_pix_loss,
        )
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # --------------------------------------------------------------------------
        # CNN Decoder Blocks:
        self.depths = decoder_depths
        self.dims = decoder_dims
        self.tubelet_size = tubelet_size
        self.output_dim = output_dim
        self.num_frames = num_frames

        self.decoder_head = CoreDecoder(
            embedding_dim=embed_dim*num_frames,
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
        """
        If use non-temporal version, switch here!
        """
        # x = x.view(N, D, int(L**0.5), int(L**0.5))
        x = x.reshape(N, D*self.num_frames, int((L/self.num_frames)**0.5), int((L/self.num_frames)**0.5))

        return x

    def forward(self, x):
        if x.dim() == 4:  # if input has no T dim, expand it
            x = x[:, :, None, :, :]
        x = self.vit_encoder(x)  # 1, 1025, 768 (b, sequence length (flattened_patches)+cls, embed_dim )
        # print("x",x.shape) # 1,589, 768 (temp)
        # remove cls token
        x = x[:, 1:, :]  # 1, 1024, 768
        # reshape into 2d features
        x = self.reshape(x)  # 1, 768, 32, 32 # 1, 768*3, 14, 14 (3*14*14 is 588)
        x = self.decoder_downsample_block(x)  # [1, 768, 32, 32]
        x = self.decoder_head(x)  # [1, 2, 512, 512]
        return x


class PrithviClassifier(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=3,
        tubelet_size=1,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        output_dim=1,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # encoder specifics
        self.vit_encoder = PrithviEncoder(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            output_dim=output_dim,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            norm_pix_loss=norm_pix_loss,
        )

        # --------------------------------------------------------------------------
        # CNN Decoder Blocks:

        self.classification_head = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=int(embed_dim / 2)),
            nn.LayerNorm(int(embed_dim / 2)),
            nn.ReLU(),
            nn.Linear(in_features=int(embed_dim / 2), out_features=output_dim),
        )

    def forward(self, x):
        x = x[:, :, None, :, :]
        x = self.vit_encoder(x)
        # select cls token
        x = x[:, 0, :]
        x = self.classification_head(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Prithvi(
        in_chans=6,
        output_dim=2,
        img_size=512,
        num_frames=1,
        decoder_norm="batch",
        decoder_padding="same",
        decoder_activation="relu",
        decoder_depths=[2, 2, 8, 2],
        decoder_dims=[160, 320, 640, 1280],
        depth=12,
        embed_dim=768,
        num_heads=3,
        patch_size=16,
        tubelet_size=1,
    )
    model = model.to(device)
    input = torch.randn((1, 6, 1, 512, 512)).to(device)
    model.forward(input)
