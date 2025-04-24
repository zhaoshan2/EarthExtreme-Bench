import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/home/EarthExtreme-Bench")
sys.path.insert(0, "/home/EarthExtreme-Bench/src/models/model_components")
# from aurora import (
#     Aurora,
#     AuroraHighRes,
#     AuroraSmall,
#     Batch,
#     Metadata,
# )
# from aurora.normalisation import locations, scales

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
        # define model
        self.disaster = disaster
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
            for param in model.parameters():
                param.requires_grad = True
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
            locations["pcp"] = settings[self.disaster]['normalization']['pcp_mean']#1.824645369
            scales["pcp"] = settings[self.disaster]['normalization']['pcp_std'] #4.5466495
            locations["p_0"] = settings[self.disaster]['normalization']['pcp_mean']#1.824645369
            scales["p_0"] = settings[self.disaster]['normalization']['pcp_std']
            locations["n"] = settings[self.disaster]['normalization']['noise_mean']
            scales["n"] = settings[self.disaster]['normalization']['noise_std']
            ## Statistics used for 30-min aurora model
            # locations["pcp"] = 1.159494744
            # scales["pcp"] = 0.900219525
            # locations["p_0"] = 1.159494744
            # scales["p_0"] = 0.900219525
            # locations["n"] = 1.0
            # scales["n"] = 1.0
            # for param in model.parameters():
            #     param.requires_grad = True

        elif model_name == "microsoft/aurora_t2m":
            model = Aurora(
                use_lora=False,
                surf_vars=("2t",),
                static_vars=("lsm", "slt", "z"),
                atmos_vars=("t",),
            )
            model.load_checkpoint_local(
                settings.ckp_path.aurora,
                strict=False,
            )
            # The mean and stds needs to be updated when using different dataset
            locations["2t"] = settings[self.disaster]['normalization']['mean'] #274.322479248046
            scales["2t"] = settings[self.disaster]['normalization']['std'] #13.129130363464355
            locations["t_0"] = settings[self.disaster]['normalization']['mean'] #
            scales["t_0"] = settings[self.disaster]['normalization']['std'] #
            locations["lsm"] = settings[self.disaster]['normalization']['mask_means'][0]#0.3388888888888889
            scales["lsm"] = settings[self.disaster]['normalization']['mask_stds'][0]# 0.4733320292105142
            locations["slt"] = settings[self.disaster]['normalization']['mask_means'][1]#0.6280021960240407
            scales["slt"] = settings[self.disaster]['normalization']['mask_stds'][1]#1.0399335522924775
            locations["z"] = settings[self.disaster]['normalization']['mask_means'][2]#3723.773681640625
            scales["z"] = settings[self.disaster]['normalization']['mask_stds'][2]# 8349.2705078125
            for param in model.parameters():
                param.requires_grad = True
            # microsoft-aurora 1.3.0 requires timm==0.6.13, but you have timm 0.9.2 (segmentation-models-pytorch 0.3.3 requires) which is incompatible.
        elif model_name == "microsoft/climax":
            from .model_components.climax.climax import ClimaX_CNN

            checkpoint = torch.load(settings.ckp_path.climax)["state_dict"]

            model = ClimaX_CNN(
                default_vars=[str(i) for i in range(input_dim)],
                # default_vars=["t2m", "lsm", "slt", "z"],
                img_size=[img_size // 4, img_size // 4],
                output_dim=output_dim,
                decoder_norm="batch",
                decoder_padding="same",
                decoder_activation="relu",
                decoder_depths=[2, 2, 8, 2],
                decoder_dims=[160, 320, 640, 1280],
            )
            if self.training:
                del checkpoint["net.pos_embed"]
                # del checkpoint["net.var_embed"]

            for i in range(input_dim):
                model.net.token_embeds[i].proj = nn.Conv2d(
                    in_channels=1,
                    out_channels=model.net.embed_dim,
                    kernel_size=model.net.patch_size,
                    stride=model.net.patch_size,
                    bias=True,
                )
                # Old patchembed weights and its mean
                original_patch_embed_weights = checkpoint[
                    f"net.token_embeds.{i}.proj.weight"
                ]
                mean_patch_embed_weights = original_patch_embed_weights.mean(
                    dim=1, keepdim=True
                )
                # load pre-trained model weights
                logger.info(f"copying the weights to {i}th patch embed.")
                with torch.no_grad():  # original_conv1.weight.shape)
                    inc = original_patch_embed_weights.shape[1]  # 4
                    tarc = model.net.token_embeds[i].proj.weight.shape[1]  # 2
                    integ = tarc // inc  # 0
                    remd = tarc % inc  # 2
                    if integ != 0:
                        model.net.token_embeds[i].proj.weight[
                            :, : (integ * inc), :, :
                        ] = nn.Parameter(
                            original_patch_embed_weights.repeat(1, integ, 1, 1)
                        )
                    # remaining dimensions are averaged from the original tensor
                    if remd != 0:
                        model.net.token_embeds[i].proj.weight[
                            :, (integ * inc) :, :, :
                        ] = nn.Parameter(
                            mean_patch_embed_weights.repeat(1, remd, 1, 1) / 3.0
                        )
                    model.net.token_embeds[i].proj.weight.requires_grad_(True)
                # After loading the token embed, delete them from the checkpoint
                del checkpoint[f"net.token_embeds.{i}.proj.weight"]
            # load pre-trained model weights
            msg = model.load_state_dict(checkpoint, strict=False)
            logger.info(msg)
            if freezing_body:
                logger.info("Freeze the encoder")
                for _, param in model.net.named_parameters():
                    param.requires_grad = False
        else:
            raise ValueError(f"Can't find matched model {model_name}.")

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
