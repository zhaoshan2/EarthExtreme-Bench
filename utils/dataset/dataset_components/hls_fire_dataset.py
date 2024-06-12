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
class BaseHlsFireDataset(torch.utils.data.Dataset):
    """
    1, Blue, B02
    2, Green, B03
    3, Red, B04
    4, NIR, B8A
    5, SW 1, B11
    6, SW 2, B12
    Further details in https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars
    """
    def __init__(self, data_path, split="train", val_ratio:float=0.1, bands:str=None, chip_size: int=256, transform=None):

        assert split in {"train", "val", "test"}
        self.split = split
        self.folder = "training" if split in {"train", "val"} else "validation"

        self.transform = transform
        self.chip_size = chip_size
        self.data_path = data_path
        self.bands = bands
        self.val_ratio = val_ratio

        self.filenames = self._read_split()  # read train/valid/test splits
    def _transform(self, x: Dict):
        if self.transform == 'resize':
            image = x['image'] #CHW
            mask = x['mask'] #HW
            new_chips = np.zeros((image.shape[0], self.chip_size, self.chip_size), dtype=np.float32)
            for i in range(image.shape[0]):
                # cv2.resize dsize receive the parameter(width, height), different from the img size of (H, W)
                new_slice = cv2.resize(image[i], (self.chip_size, self.chip_size), interpolation=cv2.INTER_LINEAR)
                new_chips[i, :, :] = new_slice
            new_mask = cv2.resize(mask, (self.chip_size, self.chip_size), interpolation=cv2.INTER_NEAREST)
            x['image'] = new_chips
            x['mask'] = new_mask
            return x

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.data_path, self.folder, f"{filename}_merged.tif")
        mask_path = os.path.join(self.data_path, self.folder, f"{filename}.mask.tif")

        with rasterio.open(image_path) as dataset:
            if self.bands is None:
                image = dataset.read() # CHW
            elif self.bands == "rgb":
                red = dataset.read(3)
                green = dataset.read(2)
                blue = dataset.read(1)
                # Stack the bands into a single array with shape (height, width, 3)
                image = np.stack((red, green, blue), axis=-1)

        mask = np.array(Image.open(mask_path)) # (512,512)
        sample = dict(image=image, mask=mask)
        if self.transform is not None:
            sample = self._transform(sample)

        return sample

    def _read_split(self):
        split_filename = "validation_index.csv" if self.split == "test" else "training_index.csv"
        split_filepath = os.path.join(self.data_path, f"{split_filename[:-10]}", split_filename)
        split_data = pd.read_csv(split_filepath)
        filenames = split_data.iloc[:, 0]
        split = round(1/self.val_ratio)
        if self.split == "train":  # 90% for train
            filenames = [x[:-11] for i, x in enumerate(filenames) if i % split != 0]
        elif self.split == "val":  # 10% for validation
            filenames = [x[:-11] for i, x in enumerate(filenames) if i % split == 0]
        elif self.split == "test":
            filenames = [x[:-11] for i, x in enumerate(filenames)]

        return filenames


class HlsFireDataset(BaseHlsFireDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = sample["image"]

        mask = np.array(Image.fromarray(sample["mask"]))

        # CHW
        sample["x"] = torch.tensor(image, dtype=torch.float32)
        # mask shape 1HW
        sample["y"] = torch.tensor(np.expand_dims(mask, 0), dtype=torch.int32)

        return sample

if __name__ == '__main__':
    dataset = HlsFireDataset(data_path='/home/EarthExtreme-Bench/data/eo/hls_burn_scars', split="test", bands='rgb', transform=None)
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
