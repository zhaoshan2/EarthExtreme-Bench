import sys
import torch
import torch.nn as nn

sys.path.insert(0, "/home/EarthExtreme-Bench")

from config.settings import settings


class BaselineNet(nn.Module):
    def __init__(
        self,
        model_name,
        input_dim=4,
        output_dim=1,
        img_size=224,
        num_frames=1,
        wave_list=[0.665, 0.56, 0.49],
        freezing_body=True,
        *args,
        **kwargs,
    ):
        super(BaselineNet, self).__init__()
        # Define model
        self.model_name = model_name
        if model_name == "ibm-nasa-geospatial/prithvi":
            from .model_components.prithvi.prithvi import Prithvi

            # If segmentation task, to use encoder+semantic segmentation head
            checkpoint = torch.load(settings.ckp_path.prithvi_100M)

            model = Prithvi(
                in_chans=input_dim,
                output_dim=output_dim,
                img_size=img_size,
                num_frames=num_frames,
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

            if self.training:
                del checkpoint["pos_embed"]
                del checkpoint["decoder_pos_embed"]
            # Load pretrained weights of encoder
            msg = model.vit_encoder.load_state_dict(checkpoint, strict=False)
            print(msg)

            # Old patchembed weights and its mean
            original_patch_embed_weights = checkpoint["patch_embed.proj.weight"]
            mean_patch_embed_weights = original_patch_embed_weights.mean(
                dim=1, keepdim=True
            )

            for name, module in model.vit_encoder.named_modules():
                if isinstance(module, nn.Conv3d) and module.in_channels == input_dim:
                    # Copy the weight from checkpoint
                    print(f"copying the weights to {name}")
                    with torch.no_grad():  # original_conv1.weight.shape)
                        # Modify the conv layer to accept 6 channels
                        integ = input_dim // 6
                        remd = input_dim % 6
                        module.weight[:, : (integ * 6), :, :] = nn.Parameter(
                            original_patch_embed_weights.repeat(1, integ, 1, 1, 1)
                        )
                        # remaining dimensions are averaged from the original tensor
                        if remd != 0:
                            module.weight[:, (integ * 6) :, :, :] = nn.Parameter(
                                mean_patch_embed_weights.repeat(1, remd, 1, 1, 1) / 3.0
                            )
                    module.weight.requires_grad_(True)
            # Freeze the encoder anf finetune the semantic segmentation head
            if freezing_body:
                print("Freeze the encoder")
                for _, param in model.vit_encoder.named_parameters():
                    param.requires_grad = False
        elif model_name == "xshadow/dofa":
            from .model_components.DOFA.models_dwv import Dofa

            checkpoint = torch.load(settings.ckp_path.dofa)
            if self.training:
                del checkpoint["pos_embed"]
            model = Dofa(
                wave_list=wave_list,
                img_size=img_size,
                output_dim=output_dim,
                decoder_norm="batch",
                decoder_padding="same",
                decoder_activation="relu",
                decoder_depths=[2, 2, 8, 2],
                decoder_dims=[160, 320, 640, 1280],
            )
            msg = model.vit_encoder.load_state_dict(checkpoint, strict=False)
            print(msg)
            # Freeze the encoder and finetune the semantic segmentation head
            if freezing_body:
                print("Freeze the encoder")
                for _, param in model.vit_encoder.named_parameters():
                    param.requires_grad = False

        elif model_name == "stanford/satmae":
            # To do: working on satmae
            from .model_components.satmae.satmae import SatMAE
            from .model_components.satmae.training_utils import split_into_three_groups

            checkpoint_model = torch.load(settings.ckp_path.satmae)["model"]
            model = SatMAE(
                img_size=img_size,
                patch_size=8,
                in_chans=input_dim,
                output_dim=output_dim,
                channel_groups=((0, 1), (2, 3), (4, 5)),
                # order S2 bands: 0-B02, 1-B03, 2-B04, 3-B08, 4-B05, 5-B06, 6-B07, 7-B8A, 8-B11, 9-B12
                # groups: (i) RGB+NIR - B2, B3, B4, B8 (ii) Red Edge - B5, B6, B7, B8A (iii) SWIR - B11, B12,
                channel_embed=256,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4.0,
                norm_pix_loss=False,
                decoder_norm="batch",
                decoder_padding="same",
                decoder_activation="relu",
                decoder_depths=[2, 2, 8, 2],
                decoder_dims=[160, 320, 640, 1280],
            )
            # load pre-trained model weights
            msg = model.vit_encoder.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            # Old patchembed weights and its mean
            original_patch_embed_weights = checkpoint_model["patch_embed.proj.weight"]
            mean_patch_embed_weights = original_patch_embed_weights.mean(
                dim=1, keepdim=True
            )

            for name, module in model.vit_encoder.named_modules():
                if isinstance(module, nn.Conv2d) and module.in_channels == input_dim:
                    # Copy the weight from checkpoint
                    print(f"copying the weights to {name}")
                    with torch.no_grad():  # original_conv1.weight.shape)
                        integ = input_dim // 10
                        remd = input_dim % 10
                        module.weight[:, : (integ * 10), :, :] = nn.Parameter(
                            original_patch_embed_weights.repeat(1, integ, 1, 1, 1)
                        )
                        # remaining dimensions are averaged from the original tensor
                        if remd != 0:
                            module.weight[:, (integ * 10) :, :, :] = nn.Parameter(
                                mean_patch_embed_weights.repeat(1, remd, 1, 1, 1) / 3.0
                            )
                    module.weight.requires_grad_(True)
            # Freeze the encoder and finetune the semantic segmentation head
            if freezing_body:
                print("Freeze the encoder")
                for _, param in model.vit_encoder.named_parameters():
                    param.requires_grad = False

        else:
            raise ValueError(f"Can't find matched model {model_name}.")

        model.float()
        self.model = model

    def _initialize_weights(self, std=0.02):
        # for m in self.decoder:
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        raw_size = x.size() if isinstance(x, torch.Tensor) else None
        x = self.model(x)
        x = getattr(x, "logits", x)
        # Interpolate if the output size doesn't match the input size
        if isinstance(x, torch.Tensor) and x.size()[-2:] != raw_size[-2:]:
            x = nn.functional.interpolate(
                x, size=raw_size[-2:], mode="bilinear", align_corners=False
            )
        return x


if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineNet(
        model_name="xshadow/dofa",
        input_dim=6,
        output_dim=2,
        img_size=512,
        num_frames=1,
        wave_list=[1, 2, 3, 4, 5, 6],
    ).to(device)
    # The model accepts remote sensing data in a video format (B, C, T, H, W)
    x = torch.randn(1, 6, 512, 512)
    x = x.to(device)
    y = model.forward(x)
    print("output", y.shape)  # (1,1,512,512)
