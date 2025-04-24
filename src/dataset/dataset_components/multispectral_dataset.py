import os
import sys
from pathlib import Path
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image

sys.path.insert(0, "/home/EarthExtreme-Bench")
from config.settings import settings


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

    def __init__(
        self,
        split="train",
        bands: str = None,
        disaster="fire",
    ):

        assert split in {"train", "val", "test"}
        self.split = split
        self.folder = "training" if split in {"train", "val"} else "validation"
        self.settings = settings[disaster]
        self.chip_size = self.settings["dataloader"]["img_size"]
        self.data_path = Path(self.settings["data_path"]) / disaster
        self.disaster = disaster
        self.bands = bands
        self.val_ratio = self.settings["dataloader"]["val_ratio"]

        self.filenames = self._read_split()  # read train/valid/test splits

    def _normalization(self, image):
        # key = self.disaster if self.disaster in settings else "default"
        # means = np.array(settings[key]["normalization"]["means"])
        # stds = np.array(settings[key]["normalization"]["stds"])
        means = np.array(self.settings["normalization"]["means"])
        stds = np.array(self.settings["normalization"]["stds"])
        image = (image - means[:, None, None]) / stds[:, None, None]
        return image

    def _transform(self, x: Dict):
        image = x["image"]  # CHW
        spatial_coord = x['spatial_coords']
        # label = x["label"]  # HW
        new_chips = np.zeros(
            (image.shape[0], self.chip_size, self.chip_size), dtype=np.float32
        )
        for i in range(image.shape[0]):
            # cv2.resize dsize receive the parameter(width, height), different from the img size of (H, W)
            new_slice = cv2.resize(
                image[i],
                (self.chip_size, self.chip_size),
                interpolation=cv2.INTER_NEAREST,
            )
            new_chips[i, :, :] = new_slice
        new_coor = cv2.resize(
            spatial_coord, (self.chip_size, self.chip_size), interpolation=cv2.INTER_NEAREST
        )
        x["image"] = new_chips
        x["spatial_coords"] = new_coor
        return x

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if self.disaster not in settings:
            raise ValueError(f"{self.disaster} is not a valid disaster")
        filename = self.filenames[idx]
        base_path = Path(self.data_path) / self.folder
        image_path = base_path / (
            filename + settings[self.disaster]["image_file_suffix"]
        )
        label_path = base_path / (
            filename + settings[self.disaster]["label_file_suffix"]
        )
        with rasterio.open(image_path) as dataset:
            if self.bands is None:
                image = dataset.read()  # CHW
                image = self._normalization(image)

            elif self.bands == "rgb":
                red = dataset.read(3)
                green = dataset.read(2)
                blue = dataset.read(1)
                # Stack the bands into a single array with shape (height, width, 3)
                image = np.stack((red, green, blue), axis=-1)
            try:
                spatial_coords = dataset.lnglat()
            except:
                # Cannot read coords
                spatial_coords = None

        label = np.array(Image.open(label_path))  # (512,512)

        sample = dict(image=image, label=label, spatial_coords=spatial_coords, meta_info=filename)
        if self.chip_size != 512:
            sample = self._transform(sample)

        return sample

    def _read_split(self):
        split_filename = (
            "validation_index.csv" if self.split == "test" else "training_index.csv"
        )
        split_filepath = os.path.join(
            self.data_path, f"{split_filename[:-10]}", split_filename
        )
        split_data = pd.read_csv(split_filepath)
        filenames = split_data.iloc[:, 0]
        split = round(1 / self.val_ratio)

        prefix = settings[self.disaster]["prefix"]  # prefix of filename
        if self.split == "train":  # 90% for train
            filenames = [x[:prefix] for i, x in enumerate(filenames) if i % split != 0]
        elif self.split == "val":  # 10% for validation
            filenames = [x[:prefix] for i, x in enumerate(filenames) if i % split == 0]
        elif self.split == "test":
            filenames = [x[:prefix] for i, x in enumerate(filenames)]

        return filenames


class MultispectralDataset(BaseMultispectralDataset):

    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = sample["image"]
        spatial_coords = sample["spatial_coords"]

        label = np.array(Image.fromarray(sample["label"]))
        mask = label == -1
        # convert missing data to nonfire
        label[label == -1] = 0
        # CHW
        tensor_x = torch.tensor(image, dtype=torch.float32)
        sample["x"] = torch.nan_to_num(tensor_x, nan=0.0)
        # raise ValueError(f"Input tensor contains NaN values")
        # y shape 1HW
        sample["y"] = torch.tensor(np.expand_dims(label, 0), dtype=torch.float32).long()
        # To do: if the noise mask not impact the final performance, then remove it.
        sample["spatial_coords"] = torch.tensor(spatial_coords, dtype=torch.float32)
        sample["noise_mask"] = torch.tensor(np.expand_dims(~mask, 0))
        return sample


class Sentinel1Flood(MultispectralDataset):
    def __init__(
        self,
        split: str = "train",
    ):
        super().__init__(
            disaster="flood",
            split=split,
        )
        self.MetaInfo = {"disaster": "flood"}

    def bbb(self):
        print("UrbanSarSentinel")


class HlsFire(MultispectralDataset):
    def __init__(
        self,
        split: str = "train",
    ):
        super().__init__(
            disaster="fire",
            split=split,
        )
        self.MetaInfo = {"disaster": "fire"}

    def bbb(self):
        print("HLS")


if __name__ == "__main__":
    dataset = MultispectralDataset(
        split="train",
        bands=None,
        disaster="flood",
    )
    x = dataset[1]
    # print(x)

    img = x["x"]
    mas = x["y"]

    # plt.figure()
    # plt.imshow(score.tensor2uint(img))
    # plt.colorbar()
    # plt.title('sample')
    # plt.savefig('input.png')
    #
    # plt.figure()
    # plt.imshow(x["y"][0])
    # plt.colorbar()
    # plt.title('mask')
    # plt.savefig('mask.png')
