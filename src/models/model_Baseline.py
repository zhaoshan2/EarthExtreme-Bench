import torch
import torch.nn as nn
import sys

sys.path.insert(0, "/home/EarthExtreme-Bench/src/models/model_components")
from transformers import (
    ConvNextConfig,
    SegformerConfig,
    SegformerForSemanticSegmentation,
    UperNetConfig,
    UperNetForSemanticSegmentation,
)
# from config.settings import settings

# transformers editable installation: https://huggingface.co/docs/transformers/installation#installing-from-source
class BaselineNet(nn.Module):
    def __init__(
        self,
        disaster,
        model_name,
        input_dim=4,
        output_dim=1,
        freezing_body=True,
        logger=None,
        *args,
        **kwargs,
    ):
        super(BaselineNet, self).__init__()
        # define model
        # model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=output_dim)

        # backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        self.disaster = disaster
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
                    logger.info(f"copying the weights to {name}")
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
            if freezing_body:
                logger.info("frozen the backbone")
                for _, param in model.backbone.named_parameters():
                    param.requires_grad = False
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
                    logger.info(f"copying the weights to {name}")
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
            if freezing_body:
                logger.info("Frozen the encoder")
                for _, param in model.segformer.encoder.named_parameters():
                    param.requires_grad = False

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
                    logger.info(f"copying the weights to {name}")
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
            if freezing_body:
                logger.info("Freeze the encoders")
                for _, param in model.encoder1.named_parameters():
                    param.requires_grad = False
                # for _, param in model.encoder2.named_parameters():
                #     param.requires_grad = False
                # for _, param in model.encoder3.named_parameters():
                #     param.requires_grad = False
                # for _, param in model.encoder4.named_parameters():
                #     param.requires_grad = False

        else:
            raise ValueError(f"Can't find matched model {model_name}.")

        self.model = model

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
    pass
