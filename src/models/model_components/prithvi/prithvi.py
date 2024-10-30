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

        # --------------------------------------------------------------------------
        # CNN Decoder Blocks:
        self.depths = decoder_depths
        self.dims = decoder_dims
        self.output_dim = output_dim

        self.decoder_head = CoreDecoder(
            embedding_dim=embed_dim,
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
        if x.dim() == 4:  # if input has no T dim, expand it
            x = x[:, :, None, :, :]
        x = self.vit_encoder(x)

        # remove cls token
        x = x[:, 1:, :]
        # reshape into 2d features
        x = self.reshape(x)
        x = self.decoder_downsample_block(x)
        x = self.decoder_head(x)
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
    # main()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CURR_FOLDER_PATH = Path(__file__).parent.parent  # "/home/EarthExtreme-Bench"
    SAVE_PATH = CURR_FOLDER_PATH / "results" / "Prithvi_100M"
    checkpoint = torch.load(
        "/home/data_storage_home/data/disaster/pretrained_model/Prithvi_100M.pt"
    )

    model = prithvi(
        checkpoint,
        output_dim=1,
        decoder_norm="batch",
        decoder_padding="same",
        decoder_activation="relu",
        decoder_depths=[2, 2, 8, 2],
        decoder_dims=[160, 320, 640, 1280],
        freeze_body=True,
        classifier=False,
        inference=False,
    )
    model.load_state_dict(torch.load(SAVE_PATH / "heatwave" / "best_model_200.pth"))

    model = model.to(device)

    import utils.dataset.era5_extreme_t2m_dataloader as ext

    heatwave = ext.HeateaveDataloader(
        batch_size=16,
        num_workers=0,
        pin_memory=False,
        horizon=28,
        chip_size=224,
        val_ratio=0.5,
        data_path="/home/EarthExtreme-Bench/data/weather",
        persistent_workers=False,
    )

    train_loader, records = heatwave.train_dataloader()
    val_loader, _ = heatwave.val_dataloader()

# train and test
