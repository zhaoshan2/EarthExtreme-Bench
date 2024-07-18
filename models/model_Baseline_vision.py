from transformers import SegformerForSemanticSegmentation
import json
from huggingface_hub import hf_hub_download
import torch.nn as nn
import timm
import math
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/EarthExtreme-Bench')
from transformers import ConvNextConfig, UperNetConfig, UperNetForSemanticSegmentation


class BaselineNet(nn.Module):
    def __init__(self, *, input_dim=4, output_dim=1, model_name):
        super(BaselineNet, self).__init__()
        # define model
        # model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=output_dim)

        # backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        original_model = UperNetForSemanticSegmentation.from_pretrained(model_name)
        original_conv1 = original_model.backbone.embeddings.patch_embeddings

        backbone_config = ConvNextConfig(out_features=["stage1", "stage2", "stage3", "stage4"], num_channels=input_dim)
        config = UperNetConfig(backbone_config=backbone_config, num_labels=output_dim)
        model = UperNetForSemanticSegmentation.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)

        for name, module in model.backbone.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 6:
                # Modify the conv layer to accept 6 channels
                print(f"copying the weights to {name}")
                module.weight = nn.Parameter(original_conv1.weight.repeat(1, 2, 1, 1) / 2.0)

        self.model = model
        # configuration = model.config

        # self.last = nn.Conv2d(model.decode_head.classifier.out_channels, output_dim, kernel_size=1, stride=1, padding=0)
    def _initialize_weights(self, std=0.02):
        for m in self.decoder:
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
        x = x.logits
        if x.size()[-2:] != raw_size[-2:]:
            x = nn.functional.interpolate(x, size=raw_size[-2:], mode="bilinear",
                                                             align_corners=False)
        return x
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.randn((1, 6, 512, 512)).to(device)
    labels = torch.randn((1, 1, 512, 512)).to(device)
    model = BaselineNet(input_dim=6, output_dim=2, model_name = 'openmmlab/upernet-convnext-tiny')
    # model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-tiny")
    model = model.to(device)
    model.train()
    output = model(input)
    print("output shape", output.logits.shape)