import json
import math
import sys

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from transformers import SegformerConfig, SegformerForSemanticSegmentation

sys.path.insert(0, "/home/EarthExtreme-Bench")
from torchsummary import summary

from transformers import ConvNextConfig, UperNetConfig, UperNetForSemanticSegmentation


class BaselineNet(nn.Module):
    def __init__(self, *, input_dim=4, output_dim=1, model_name):
        super(BaselineNet, self).__init__()
        # define model
        # model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=output_dim)

        # backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        if model_name == "openmmlab/upernet-convnext-tiny":
            backbone_config = ConvNextConfig(
                out_features=["stage1", "stage2", "stage3", "stage4"],
                num_channels=input_dim,
            )
            config = UperNetConfig(
                backbone_config=backbone_config, num_labels=output_dim
            )
            model = UperNetForSemanticSegmentation(config=config)

        elif model_name == "nvidia/mit-b0":
            # https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation
            config = SegformerConfig(num_channels=input_dim, num_labels=output_dim)
            model = SegformerForSemanticSegmentation(config=config)

        elif model_name == "unet":
            model = torch.hub.load(
                "mateuszbuda/brain-segmentation-pytorch",
                model=model_name,
                in_channels=input_dim,
                out_channels=output_dim,
                init_features=32,
                pretrained=False,
            )

        else:
            raise ValueError(f"Can't find matched model {model_name}.")

        self.model = model

        self._initialize_weights()

    def _initialize_weights(self, std=0.02):
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        raw_size = x.size()
        x = self.model(x)
        try:
            x = x.logits
        except AttributeError:
            x = x
        if x.size()[-2:] != raw_size[-2:]:
            x = nn.functional.interpolate(
                x, size=raw_size[-2:], mode="bilinear", align_corners=False
            )
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.randn((1, 4, 128, 128)).to(device)
    labels = torch.randn((1, 2, 128, 128)).to(device)
    model = BaselineNet(input_dim=4, output_dim=2, model_name="unet")
    # model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-tiny")
    model = model.to(device)
    model.train()
    output = model(input)
    print("output shape", output.shape)
