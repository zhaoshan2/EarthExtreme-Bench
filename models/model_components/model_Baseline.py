import json
import math
import sys

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from transformers import SegformerConfig, SegformerForSemanticSegmentation

from aurora.normalisation import locations, scales

from transformers import ConvNextConfig, UperNetConfig, UperNetForSemanticSegmentation
from aurora import AuroraHighRes, AuroraSmall, Aurora, Batch, Metadata


class BaselineNet(nn.Module):
    def __init__(self, *, input_dim=4, output_dim=1, model_name):
        super(BaselineNet, self).__init__()
        # define model
        # model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=output_dim)

        # backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        self.model_name = model_name
        if model_name == "openmmlab/upernet-convnext-tiny":
            original_model = UperNetForSemanticSegmentation.from_pretrained(model_name)
            original_conv1 = original_model.backbone.embeddings.patch_embeddings

            backbone_config = ConvNextConfig(
                out_features=["stage1", "stage2", "stage3", "stage4"],
                num_channels=input_dim,
            )
            config = UperNetConfig(
                backbone_config=backbone_config, num_labels=output_dim
            )
            model = UperNetForSemanticSegmentation.from_pretrained(
                model_name, config=config, ignore_mismatched_sizes=True
            )
            for name, module in model.backbone.named_modules():
                if isinstance(module, nn.Conv2d) and module.in_channels == input_dim:
                    # Modify the conv layer to accept 6 channels
                    print(f"copying the weights to {name}")
                    with torch.no_grad():  # original_conv1.weight.shape)
                        # Modify the conv layer to accept 6 channels
                        integ = input_dim // 3
                        remd = input_dim % 3
                        module.weight[:, : (integ * 3), :, :] = nn.Parameter(
                            original_conv1.weight.repeat(1, integ, 1, 1) / 3.0
                        )
                        # remaining dimensions are averaged from the original tensor
                        if remd != 0:
                            module.weight[:, (integ * 3) :, :, :] = nn.Parameter(
                                original_conv1.weight.mean(dim=1)
                                .unsqueeze(1)
                                .repeat(1, remd, 1, 1)
                                / 3.0
                            )
                    module.weight.requires_grad_(True)
        elif model_name == "nvidia/mit-b0":
            # https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation
            original_model = SegformerForSemanticSegmentation.from_pretrained(
                model_name, num_labels=output_dim
            )
            original_conv1 = original_model.segformer.encoder.patch_embeddings[0].proj

            config = SegformerConfig(num_channels=input_dim, num_labels=output_dim)
            model = SegformerForSemanticSegmentation.from_pretrained(
                model_name, config=config, ignore_mismatched_sizes=True
            )

            for name, module in model.segformer.encoder.named_modules():
                if isinstance(module, nn.Conv2d) and module.in_channels == input_dim:
                    print(f"copying the weights to {name}")
                    with torch.no_grad():  # original_conv1.weight.shape)
                        # Modify the conv layer to accept input_dim channels
                        integ = input_dim // 3
                        remd = input_dim % 3
                        module.weight[:, : (integ * 3), :, :] = nn.Parameter(
                            original_conv1.weight.repeat(1, integ, 1, 1) / 3.0
                        )
                        # remaining dimensions are averaged from the original tensor
                        if remd != 0:
                            module.weight[:, (integ * 3) :, :, :] = nn.Parameter(
                                original_conv1.weight.mean(dim=1)
                                .unsqueeze(1)
                                .repeat(1, remd, 1, 1)
                                / 3.0
                            )
                    module.weight.requires_grad_(True)
        elif model_name == "unet":
            model = torch.hub.load(
                "mateuszbuda/brain-segmentation-pytorch",
                model=model_name,
                in_channels=3,
                out_channels=1,
                init_features=32,
                pretrained=True,
            )
            original_model_conv = model.encoder1.enc1conv1
            # Modify the 1st and the last layer of the original model
            model.encoder1.enc1conv1 = nn.Conv2d(
                in_channels=input_dim,
                out_channels=32,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            )
            model.conv = nn.Conv2d(
                in_channels=32,
                out_channels=output_dim,
                kernel_size=(1, 1),
                stride=(1, 1),
            )

            for name, module in model.encoder1.named_modules():
                if isinstance(module, nn.Conv2d) and module.in_channels == input_dim:
                    print(f"copying the weights to {name}")
                    with torch.no_grad():  # original_conv1.weight.shape)
                        # Modify the conv layer to accept input_dim channels
                        integ = input_dim // 3
                        remd = input_dim % 3
                        module.weight[:, : (integ * 3), :, :] = nn.Parameter(
                            original_model_conv.weight.repeat(1, integ, 1, 1) / 3.0
                        )
                        # remaining dimensions are averaged from the original tensor
                        if remd != 0:
                            module.weight[:, (integ * 3) :, :, :] = nn.Parameter(
                                original_model_conv.weight.mean(dim=1)
                                .unsqueeze(1)
                                .repeat(1, remd, 1, 1)
                                / 3.0
                            )
                    module.weight.requires_grad_(True)
        elif model_name == "microsoft/aurora_small":
            model = AuroraSmall()
            model.load_checkpoint(
                "microsoft/aurora", "aurora-0.25-small-pretrained.ckpt"
            )
            for param in model.parameters():
                param.requires_grad = True
        elif model_name == "microsoft/aurora_highres":
            model = AuroraHighRes()
            model.load_checkpoint("wbruinsma/aurora", "aurora-0.1-finetuned.ckpt")
            for param in model.parameters():
                param.requires_grad = True
        elif model_name == "microsoft/aurora":
            model = Aurora(
                use_lora=False,
                surf_vars=("pcp",),
                static_vars=("n",),
                atmos_vars=("p",),
            )
            model.load_checkpoint_local(
                "/home/data_storage_home/data/disaster/pretrained_model/aurora-0.25-pretrained.ckpt",
                strict=False,
            )
            locations["pcp"] = 1.159494744
            scales["pcp"] = 0.900219525
            locations["p_0"] = 1.159494744
            scales["p_0"] = 0.900219525
            locations["n"] = 1.0
            scales["n"] = 1.0

            for param in model.parameters():
                param.requires_grad = True
            # microsoft-aurora 1.3.0 requires timm==0.6.13, but you have timm 0.9.2 (segmentation-models-pytorch 0.3.3 requires) which is incompatible.

        else:
            raise ValueError(f"Can't find matched model {model_name}.")

        self.model = model

    def _initialize_weights(self, std=0.02):
        # for m in self.decoder:
        for m in self.model:
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
    from datetime import datetime

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # ['msl', 'u10', 'v10']
    batch = Batch(
        surf_vars={k: torch.randn(1, 2, 17, 32) for k in ("2t", "10u", "10v", "msl")},
        static_vars={k: torch.randn(17, 32) for k in ("lsm", "z", "slt")},
        atmos_vars={k: torch.randn(1, 2, 4, 17, 32) for k in ("z", "u", "v", "t", "q")},
        metadata=Metadata(
            lat=torch.linspace(90, -90, 1800),
            lon=torch.linspace(0, 360, 3601)[:-1],
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=(100, 250, 500, 850),
        ),
    )
    batch = batch.to(device)

    model = BaselineNet(model_name="microsoft/aurora").to(device)
    prediction = model.forward(batch)
    output_test = prediction.surf_vars["msl"].detach().cpu().numpy()
    x = batch.surf_vars["msl"].detach().cpu().numpy()

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(5, 10))
    im = axes[0].imshow(x[0, 0])
    plt.colorbar(im, ax=axes[0])
    axes[0].set_title("input")

    im = axes[1].imshow(output_test[0, 0])
    plt.colorbar(im, ax=axes[1])
    axes[1].set_title("pred")
    plt.savefig("rollout_1_aurora_random")
