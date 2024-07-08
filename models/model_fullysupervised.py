import torch
import torch.nn as nn
import timm
import sys

sys.path.insert(0, '/home/EarthExtreme-Bench')
import numpy as np
from pathlib import Path
import os
import argparse
import math
class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class BaselineNet(nn.Module):
    def __init__(self, *, input_dim=4, output_dim=1, num_features=256, activation="relu", norm="batch", padding="same", model_name):
        super(BaselineNet, self).__init__()

        backbone = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=(3,))
        # Modify the input layer to accept four-channel images
        backbone.patch_embed.proj = nn.Conv2d(input_dim, backbone.patch_embed.proj.out_channels, kernel_size=(4, 4),
                                              stride=(4, 4))
        self.backbone = backbone
        # Modify the output layer to produce single-channel images
        self.decoder = nn.Sequential(
            nn.Conv2d(backbone.feature_info[-1]['num_chs'], num_features*2, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_features*2, num_features, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            Upsample(scale=32, num_feat=num_features),
            nn.Conv2d(num_features, output_dim, 3, 1, 1)
        )

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
        features = self.backbone(x)[-1]  # Get the last feature map from the backbone total length: 1
        features = torch.transpose(features, dim0=3, dim1=1)
        features = torch.transpose(features, dim0=-1, dim1=-2)

        x = self.decoder(features)
        return x




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--disaster', type=str, default="heatwave")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CURR_FOLDER_PATH = Path(__file__).parent.parent  # "/home/EarthExtreme-Bench"

    # checkpoint = torch.load('/home/data_storage_home/data/disaster/pretrained_model/Prithvi_100M.pt')
    model_name = 'swin_small_patch4_window7_224'
    SAVE_PATH = CURR_FOLDER_PATH / 'results' / model_name
    # model.load_state_dict(torch.load(SAVE_PATH / 'heatwave' / 'best_model_200.pth'))

    if args.disaster == "heatwave":

        from utils.dataset.era5_dataloader import ERA5Dataloader
        from utils.trainer.era5_train_and_test import train, test
        # dataset
        heatwave = ERA5Dataloader(batch_size=16,
                           num_workers=0,
                           pin_memory=False,
                           horizon=28,
                           chip_size=512,
                           val_ratio=0.5,
                           data_path='/home/EarthExtreme-Bench/data/weather',
                           persistent_workers=False)

        train_loader, records = heatwave.train_dataloader()
        print("length of training loader", len(train_loader))
        val_loader, _ = heatwave.val_dataloader()
        print("length of validation loader", len(val_loader))
        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)
        # model
        from transformers import SegformerForSemanticSegmentation
        import json
        from huggingface_hub import hf_hub_download

        # define model
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=1)

        # model = BaselineNet(input_dim=4, output_dim=1, model_name=model_name)
        model = model.to(device)

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-6)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,"min")
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 50], gamma=0.5)

        # training
        best_model_state_dict, best_epoch = train(model, train_loader, val_loader, device, save_path=SAVE_PATH, num_epochs=100, optimizer=optimizer, lr_scheduler=lr_scheduler)
        print(f"the model is saved at epoch {best_epoch}")
        # rename the best_model_state_dict with its epoch
        model_id = f"best_model" #current
        # ckp_path = SAVE_PATH / args.disaster / f"{model_id}.pth"

        # best_model_state_dict = torch.load(SAVE_PATH / args.disaster / f'{model_id}.pth')
        msg = model.load_state_dict(best_model_state_dict)
        print(msg)
        # testing
        test_loader, records = heatwave.test_dataloader()
        print("length of test loader", len(test_loader))
        _ = test(model, test_loader, device, stats=records.mean_std_dic, save_path=SAVE_PATH,  model_id = model_id)

    elif args.disaster == "fire":
        from utils.dataset.hls_fire_dataloader import HlsFireDataloader
        from utils.trainer.multispectral_train_and_test import train, test
        # dataset
        burned = HlsFireDataloader(batch_size=1,
            num_workers=0,
            pin_memory=False,
            chip_size=512,
            data_path= '/home/EarthExtreme-Bench/data/eo/hls_burn_scars',
            val_ratio=0.2,
            persistent_workers=False,
            transform=None)
        train_loader = burned.train_dataloader()
        val_loader = burned.val_dataloader()


        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)
        # model
        from transformers import SegformerForSemanticSegmentation
        import json
        from huggingface_hub import hf_hub_download

        # define model
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=2)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-6)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,"min")
        #best_model_state_dict = train(model, train_loader, val_loader, device, save_path=SAVE_PATH,
        #                                num_epochs = 100, optimizer=optimizer,
        #                                    lr_scheduler=lr_scheduler, disaster=args.disaster)
        model_id = 'best_model_60'
        best_model_state_dict = torch.load(SAVE_PATH / args.disaster / f"{model_id}.pth")
        msg = model.load_state_dict(best_model_state_dict)
        print(msg)

        test_loader = burned.test_dataloader()
        _ = test(model, test_loader, device, save_path=SAVE_PATH, disaster=args.disaster, model_id = model_id)