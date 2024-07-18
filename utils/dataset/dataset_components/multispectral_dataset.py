import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import pandas as pd
import rasterio
from typing import Any, Dict, List, Optional, Tuple
import sys
sys.path.insert(0, '/home/EarthExtreme-Bench')
from utils import score
import cv2


class BaseMultispectralDataset(torch.utils.data.Dataset):
    """
    1, Blue, B02
    2, Green, B03
    3, Red, B04
    4, NIR, B8A
    5, SW 1, B11
    6, SW 2, B12
    Further details in https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars
    """
    def __init__(self, data_path, split="train", val_ratio:float=0.1, bands:str=None, chip_size: int=256, transform=None, disaster="fire"):

        assert split in {"train", "val", "test"}
        self.split = split
        self.folder = "training" if split in {"train", "val"} else "validation"

        self.transform = transform
        self.chip_size = chip_size
        self.data_path = data_path
        self.disaster = disaster
        self.bands = bands
        self.val_ratio = val_ratio

        self.filenames = self._read_split()  # read train/valid/test splits
    def _normalization(self, image):
        if self.disaster == "fire":
            img_norm_cfg = dict(
                means=[
                    0.033349706741586264,
                    0.05701185520536176,
                    0.05889748132001316,
                    0.2323245113436119,
                    0.1972854853760658,
                    0.11944914225186566,
                ],
                stds=[
                    0.02269135568823774,
                    0.026807560223070237,
                    0.04004109844362779,
                    0.07791732423672691,
                    0.08708738838140137,
                    0.07241979477437814,
                ],
             )
        elif self.disaster == "flood":
            img_norm_cfg = dict(
                means=[
                    0.23651549,
                    0.31761484,
                    0.18514981,
                    0.26901252,
                    -14.57879175,
                    -8.6098158,
                    -14.29073382,
                    -8.33534564
                ],
                stds=[
                    0.16280619,
                    0.20849304,
                    0.14008107,
                    0.19767644,
                    4.07141682,
                    3.94773216,
                    4.21006244,
                    4.05494136
                ],
             )
        else:
            img_norm_cfg = dict(
                means=[
                    0.485,
                    0.456,
                    0.406
                ],
                stds=[
                    0.229,
                    0.224,
                    0.225
                ],
            )
        means = np.array(img_norm_cfg['means'])
        stds = np.array(img_norm_cfg['stds'])
        image = (image - means[:, None, None]) /stds[:, None, None]
        return image


    def _transform(self, x: Dict):
        if self.transform == 'resize':
            image = x['image'] #CHW
            mask = x['mask'] #HW
            new_chips = np.zeros((image.shape[0], self.chip_size, self.chip_size), dtype=np.float32)
            for i in range(image.shape[0]):
                # cv2.resize dsize receive the parameter(width, height), different from the img size of (H, W)
                new_slice = cv2.resize(image[i], (self.chip_size, self.chip_size), interpolation=cv2.INTER_NEAREST)
                new_chips[i, :, :] = new_slice
            new_mask = cv2.resize(mask, (self.chip_size, self.chip_size), interpolation=cv2.INTER_NEAREST)
            x['image'] = new_chips
            x['mask'] = new_mask
        return x

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        if self.disaster == "fire":
            image_path = os.path.join(self.data_path, self.folder, f"{filename}_merged.tif")
            mask_path = os.path.join(self.data_path, self.folder, f"{filename}.mask.tif")
        elif self.disaster == "flood":
            image_path = os.path.join(self.data_path, self.folder, f"{filename}_SAR.tif")
            mask_path = os.path.join(self.data_path, self.folder, f"{filename}_GT.tif")
        with rasterio.open(image_path) as dataset:
            if self.bands is None:
                image = dataset.read() # CHW
                image = self._normalization(image)

            elif self.bands == "rgb":
                red = dataset.read(3)
                green = dataset.read(2)
                blue = dataset.read(1)
                # Stack the bands into a single array with shape (height, width, 3)
                image = np.stack((red, green, blue), axis=-1)

        mask = np.array(Image.open(mask_path)) # (512,512)

        # convert missing data to nonfire
        mask[mask == -1] = 0
        sample = dict(image=image, mask=mask, id=filename)
        if self.transform is not None:
            sample = self._transform(sample)

        return sample

    def _read_split(self):
        split_filename = "validation_index.csv" if self.split == "test" else "training_index.csv"
        split_filepath = os.path.join(self.data_path, f"{split_filename[:-10]}", split_filename)
        split_data = pd.read_csv(split_filepath)
        filenames = split_data.iloc[:, 0]
        split = round(1/self.val_ratio)

        perfex = -11 if self.disaster=="fire" else -8
        if self.split == "train":  # 90% for train
            filenames = [x[:perfex] for i, x in enumerate(filenames) if i % split != 0]
        elif self.split == "val":  # 10% for validation
            filenames = [x[:perfex] for i, x in enumerate(filenames) if i % split == 0]
        elif self.split == "test":
            filenames = [x[:perfex] for i, x in enumerate(filenames)]

        return filenames


class MultispectralDataset(BaseMultispectralDataset):

    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = sample["image"]

        mask = np.array(Image.fromarray(sample["mask"]))

        # CHW
        sample["x"] = torch.tensor(image, dtype=torch.float32)
        # mask shape 1HW
        sample["y"] = torch.tensor(np.expand_dims(mask, 0), dtype=torch.float32)
        return sample

if __name__ == '__main__':
    dataset = BaseMultispectralDataset(data_path='/home/EarthExtreme-Bench/data/eo/flood', split="test", bands='rgb', transform=None, disaster="flood")
    x = dataset[0]
    img = x["x"]
    mas = x["y"]

    plt.figure()
    plt.imshow(score.tensor2uint(img))
    plt.colorbar()
    plt.title('sample')
    plt.savefig('input.png')

    plt.figure()
    plt.imshow(x["y"][0])
    plt.colorbar()
    plt.title('mask')
    plt.savefig('mask.png')
