import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/home/EarthExtreme-Bench")
sys.path.insert(0, "/home/EarthExtreme-Bench/src/models/model_components")
from aurora import (
    Aurora,
    AuroraHighRes,
    AuroraSmall,
    Batch,
    Metadata,
)
from aurora.normalisation import locations, scales

from config.settings import settings


class BaselineNet(nn.Module):
    def __init__(self, model_name, *args, **kwargs):
        super(BaselineNet, self).__init__()
        # define model
        self.model_name = model_name
        if model_name == "microsoft/aurora_small":
            model = AuroraSmall()
            model.load_checkpoint(
                "microsoft/aurora", "aurora-0.25-small-pretrained.ckpt"
            )

        elif model_name == "microsoft/aurora_highres":
            model = AuroraHighRes()
            model.load_checkpoint("wbruinsma/aurora", "aurora-0.1-finetuned.ckpt")

        elif model_name == "microsoft/aurora":
            model = Aurora()
            model.load_checkpoint_local(
                settings.ckp_path.aurora,
                strict=False,
            )

        elif model_name == "microsoft/aurora_pcp":
            model = Aurora(
                use_lora=False,
                surf_vars=("pcp",),
                static_vars=("n",),
                atmos_vars=("p",),
            )
            model.load_checkpoint_local(
                settings.ckp_path.aurora,
                strict=False,
            )
            # The mean and stds needs to be updated when using different dataset
            locations["pcp"] = 1.824645369
            scales["pcp"] = 1.122875855
            locations["p_0"] = 1.824645369
            scales["p_0"] = 1.122875855
            locations["n"] = 0.7784
            scales["n"] = 0.4154

            # microsoft-aurora 1.3.0 requires timm==0.6.13, but you have timm 0.9.2 (segmentation-models-pytorch 0.3.3 requires) which is incompatible.

        else:
            raise ValueError(f"Can't find matched model {model_name}.")
        for param in model.parameters():
            param.requires_grad = True
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
