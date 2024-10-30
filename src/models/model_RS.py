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
                in_chans=6,
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
            model.vit_encoder.load_state_dict(checkpoint, strict=False)
            # Modify the input layer to receive the input_dim
            model.vit_encoder.patch_embed.proj = nn.Conv3d(
                in_channels=input_dim,
                out_channels=768,
                kernel_size=(
                    1,
                    16,
                    16,
                ),
                stride=(
                    1,
                    16,
                    16,
                ),
                bias=True,
            )

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
                    # Freeze the encoder anf finetune the semantic segmentation head
                    for _, param in model.vit_encoder.named_parameters():
                        param.requires_grad = False

        elif model_name == "ibm-nasa-geospatial/prithvi_classifier":
            from .model_components.prithvi.prithvi import Prithvi, PrithviClassifier

            # If classification task, we use the encoder + classifier
            checkpoint = torch.load(settings.ckp_path.prithvi_100M)
            # for k, v in checkpoint.items():
            #     print(k)
            model = PrithviClassifier(
                in_chans=6,
                output_dim=output_dim,
                img_size=img_size,
                num_frames=num_frames,
                depth=12,
                embed_dim=768,
                num_heads=3,
                patch_size=16,
                tubelet_size=1,
            )
            if self.training:
                del checkpoint["pos_embed"]
                del checkpoint["decoder_pos_embed"]
            # Only load encoder weights
            model.vit_encoder.load_state_dict(checkpoint, strict=False)
            # Modify the input layer to receive the input_dim
            model.vit_encoder.patch_embed.proj = nn.Conv3d(
                in_channels=input_dim,
                out_channels=768,
                kernel_size=(
                    1,
                    16,
                    16,
                ),
                stride=(
                    1,
                    16,
                    16,
                ),
                bias=True,
            )

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

            # Freeze the encoder and train a new classifer
            for _, param in model.vit_encoder.named_parameters():
                param.requires_grad = False
        elif model_name == "xshadow/dofa":
            from .model_components.DOFA.models_dwv import vit_base_patch16

            checkpoint = torch.load(settings.ckp_path.dofa)
            model = vit_base_patch16(wave_list=[1, 2, 3])
            model.load_state_dict(checkpoint, strict=False)

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
        model_name="ibm-nasa-geospatial/prithvi",
        input_dim=6,
        output_dim=1,
        img_size=512,
        num_frames=1,
    ).to(device)
    # The model accepts remote sensing data in a video format (B, C, T, H, W)
    x = torch.randn(1, 6, 1, 512, 512)
    x = x.to(device)
    y = model.forward(x)
    print("output", y.shape)  # (1,1,512,512)
