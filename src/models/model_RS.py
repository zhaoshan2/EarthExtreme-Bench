import sys
import torch
import torch.nn as nn

import math

sys.path.insert(0, "/home/EarthExtreme-Bench")

from config.settings import settings


class BaselineNet(nn.Module):
    def __init__(
        self,
        disaster,
        model_name,
        input_dim=4,
        output_dim=1,
        img_size=224,
        num_frames=1,
        wave_list=[0.665, 0.56, 0.49],
        freezing_body=True,
        logger=None,
        *args,
        **kwargs,
    ):
        super(BaselineNet, self).__init__()
        # Define model
        self.disaster = disaster
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
            msg = model.vit_encoder.load_state_dict(checkpoint, strict=False)
            logger.info(msg)

            # Modify the input layer to receive the input_dim
            model.vit_encoder.patch_embed.proj = nn.Conv3d(
                in_channels=input_dim,
                out_channels=model.embed_dim,
                kernel_size=(
                    model.tubelet_size,
                    model.patch_size,
                    model.patch_size,
                ),
                stride=(
                    model.tubelet_size,
                    model.patch_size,
                    model.patch_size,
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
                    module.weight.requires_grad_(True)
            # Freeze the encoder anf finetune the semantic segmentation head
            if freezing_body:
                logger.info("Freeze the encoder")
                for _, param in model.vit_encoder.named_parameters():
                    param.requires_grad = False

        # 2025-01-20: To do: load prithvi 2.0 pretrained weights
        elif model_name == "ibm-nasa-geospatial/prithvi-2_upernet":
            from .model_components.prithvi_2.prithvi_upernet import Prithvi

            # If segmentation task, to use encoder+semantic segmentation head
            checkpoint = torch.load(settings.ckp_path.prithvi_eo_v2_300M)

            model = Prithvi(
                in_chans=6,
                output_dim=output_dim,
                img_size=img_size,
                num_frames=1,
            )
            # for n, p in model.named_parameters():
            #     print(n)
            # Remove keys containing "decoder" and rename encoder.* to *
            checkpoint = {
                key: value for key, value in checkpoint.items()
                if 'decoder' not in key
            }
            checkpoint = {
                key.replace("encoder.", ""): value for key, value in checkpoint.items()
            }
            if self.training:
                del checkpoint["pos_embed"]
                # del checkpoint["temporal_embed_enc"]
                # del checkpoint["location_embed_enc"]
            # Load pretrained weights of encoder
            msg = model.encoder.load_state_dict(checkpoint, strict=False)
            logger.info(msg)
            # print(msg)

            # Modify the input layer to receive the input_dim
            model.encoder.patch_embed.proj = nn.Conv3d(
                in_channels=input_dim,
                out_channels=model.embed_dim,
                kernel_size= model.encoder.patch_size,
                stride=model.encoder.patch_size,
                bias=True,
            )
            # Old patchembed weights and its mean
            original_patch_embed_weights = checkpoint["patch_embed.proj.weight"]
            mean_patch_embed_weights = original_patch_embed_weights.mean(
                dim=1, keepdim=True
            )

            for name, module in model.encoder.named_modules():
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
                            module.weight[:, (integ * 6):, :, :] = nn.Parameter(
                                mean_patch_embed_weights.repeat(1, remd, 1, 1, 1) / 3.0
                            )
                    module.weight.requires_grad_(True)
            # Freeze the encoder anf finetune the semantic segmentation head
            if freezing_body:
                logger.info("Freeze the encoder")
                for _, param in model.encoder.named_parameters():
                    param.requires_grad = False

        elif model_name == "xshadow/dofa":
            from .model_components.dofa.models_dwv import Dofa

            checkpoint = torch.load(settings.ckp_path.dofa)

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

            if "pos_embed" in checkpoint.keys():
                if model.vit_encoder.pos_embed.shape != checkpoint["pos_embed"].shape:
                    logger.info(
                        f"Resize the pos_embed shape from "
                        f'{checkpoint["pos_embed"].shape} to '
                        f"{model.vit_encoder.pos_embed.shape}"
                    )
                    h, w = img_size, img_size
                    pos_size = int(math.sqrt(checkpoint["pos_embed"].shape[1] - 1))
                    checkpoint["pos_embed"] = model.resize_pos_embed(
                        checkpoint["pos_embed"],
                        (
                            h // model.vit_encoder.patch_size,
                            w // model.vit_encoder.patch_size,
                        ),
                        (pos_size, pos_size),
                        "bicubic",
                    )

            msg = model.vit_encoder.load_state_dict(checkpoint, strict=False)
            logger.info(msg)
            # Freeze the encoder and finetune the semantic segmentation head
            if freezing_body:
                logger.info("Freeze the encoder")
                for _, param in model.vit_encoder.named_parameters():
                    param.requires_grad = False
        elif model_name == "xshadow/dofa_upernet":
            from .model_components.dofa.models_dwv_upernet import Dofa

            checkpoint = torch.load(settings.ckp_path.dofa)

            model = Dofa(
                wave_list=wave_list,
                img_size=img_size,
                output_dim=output_dim,
            )

            if "pos_embed" in checkpoint.keys():
                if model.vit_encoder.pos_embed.shape != checkpoint["pos_embed"].shape:
                    logger.info(
                        f"Resize the pos_embed shape from "
                        f'{checkpoint["pos_embed"].shape} to '
                        f"{model.vit_encoder.pos_embed.shape}"
                    )
                    h, w = img_size, img_size
                    pos_size = int(math.sqrt(checkpoint["pos_embed"].shape[1] - 1))
                    checkpoint["pos_embed"] = model.resize_pos_embed(
                        checkpoint["pos_embed"],
                        (
                            h // model.vit_encoder.patch_size,
                            w // model.vit_encoder.patch_size,
                        ),
                        (pos_size, pos_size),
                        "bicubic",
                    )

            msg = model.vit_encoder.load_state_dict(checkpoint, strict=False)
            logger.info(msg)
            # Freeze the encoder and finetune the semantic segmentation head
            if freezing_body:
                logger.info("Freeze the encoder")
                for _, param in model.vit_encoder.named_parameters():
                    param.requires_grad = False

        elif model_name == "stanford/satmae":
            # To do: working on satmae
            from .model_components.satmae.satmae import SatMAE
            from .model_components.satmae.training_utils import split_into_three_groups

            checkpoint_model = torch.load(settings.ckp_path.satmae)["model"]
            del checkpoint_model["pos_embed"]
            # channel_groups = ((0, 1, 2), (3, 4, 5), (6, 7))  # flood
            # channel_groups = ((0,), (1,), (2, 3))
            channel_groups = ((0,1), (2,3), (4, 5)) # fire
            model = SatMAE(
                img_size=img_size,
                patch_size=8,
                in_chans=input_dim,
                output_dim=output_dim,
                channel_groups=channel_groups,
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

            for i in range(len(channel_groups)):
                model.vit_encoder.patch_embed[i].proj = nn.Conv2d(
                    in_channels=len(channel_groups[i]),
                    out_channels=model.embed_dim,
                    kernel_size=model.patch_size,
                    stride=model.patch_size,
                    bias=True,
                )
                # Old patchembed weights and its mean
                original_patch_embed_weights = checkpoint_model[
                    f"patch_embed.{i}.proj.weight"
                ]
                mean_patch_embed_weights = original_patch_embed_weights.mean(
                    dim=1, keepdim=True
                )
                del checkpoint_model[f"patch_embed.{i}.proj.weight"]
                # load pre-trained model weights

                logger.info(f"copying the weights to {i}th patch embed.")
                with torch.no_grad():  # original_conv1.weight.shape)
                    inc = original_patch_embed_weights.shape[1]  # 4
                    tarc = model.vit_encoder.patch_embed[i].proj.weight.shape[1]  # 2
                    integ = tarc // inc  # 0
                    remd = tarc % inc  # 2
                    if integ != 0:
                        model.vit_encoder.patch_embed[i].proj.weight[
                            :, : (integ * inc), :, :
                        ] = nn.Parameter(
                            original_patch_embed_weights.repeat(1, integ, 1, 1)
                        )
                    # remaining dimensions are averaged from the original tensor
                    if remd != 0:
                        model.vit_encoder.patch_embed[i].proj.weight[
                            :, (integ * inc) :, :, :
                        ] = nn.Parameter(
                            mean_patch_embed_weights.repeat(1, remd, 1, 1) / 3.0
                        )
                    model.vit_encoder.patch_embed[i].proj.weight.requires_grad_(True)
            msg = model.vit_encoder.load_state_dict(checkpoint_model, strict=False)
            logger.info(msg)
            # Freeze the encoder and finetune the semantic segmentation head
            if freezing_body:
                logger.info("Freeze the encoder")
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
        if isinstance(x, torch.Tensor) and raw_size is not None and x.size()[-2:] != raw_size[-2:]:
            x = nn.functional.interpolate(
                x, size=raw_size[-2:], mode="bilinear", align_corners=False
            )
        return x


if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineNet(
        disaster="fire",
        model_name="ibm-nasa-geospatial/prithvi-2_upernet",
        input_dim=6,
        output_dim=1,
        img_size=224,
        num_frames=1,
        wave_list=[1, 2, 3, 4, 5, 6],
    ).to(device)
    # The model accepts remote sensing data in a video format (B, C, T, H, W)
    x = torch.randn(1, 6, 224, 224)
    x = x.to(device)
    y = model.forward(x)
    print("output", y.shape)  # (1,1,512,512)
