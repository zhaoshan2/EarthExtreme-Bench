import datetime
import json
import os
import pickle
import random
import sys
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import rasterio
import torch
import xarray as xr
from torch import Tensor
from torch.utils import data
from tqdm import tqdm

sys.path.insert(0, "/home/EarthExtreme-Bench")
from config.settings import settings


def crop_from_upper_left(img, base):
    """
    Parameters:
        img: torch.Tensor
        base: the multiple of base of cropped image
    Return:
        cropped_img: the spatial size is multiple of base
    """
    # Get original height and width
    height, width = img.shape[-2:]

    # Calculate the new height and width, closest multiples of 4
    new_height = (height // base) * base
    new_width = (width // base) * base

    # Crop from the upper-left corner
    cropped_img = img[..., :new_height, :new_width]

    return cropped_img


def resize_and_crop(img, dst_w, dst_h, lon=None, lat=None):
    # Get the dimensions of the image
    height, width = img.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = width / height
    shorter_side_length = min(dst_w, dst_h)

    # Resize the image
    if width < height:
        new_width = shorter_side_length
        new_height = int(shorter_side_length / aspect_ratio)
    else:
        new_height = shorter_side_length
        new_width = int(shorter_side_length * aspect_ratio)

    resized_img = cv2.resize(
        img, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    # Calculate scaling factors due to resizing
    scaling_factor = width / new_width

    # Randomly crop a 224x224 region from the resized image
    # x = random.randint(0, new_width - dst_h)
    # y = random.randint(0, new_height - dst_w)
    # Central crop a 224x224 region from the resized image
    x = (new_width - dst_w) // 2
    y = (new_height - dst_h) // 2
    cropped_img = resized_img[y : y + dst_h, x : x + dst_w, ...]

    if lon is not None:
        # Calculate the corresponding longitude and latitude of the cropped patch
        delta_lon = x * 0.25 * scaling_factor
        delta_lat = y * 0.25 * scaling_factor

        # The new upper-left corner coordinates in degrees
        if lon < 0:
            lon = 360 + lon
        new_lon = lon + delta_lon
        new_lat = lat - delta_lat  # latitude decreases as you move down
        # cropped image, (new lon, new lat, new spatial resolution)

        return cropped_img, (new_lon, new_lat, 0.25 * scaling_factor)
    else:
        return cropped_img


class Record:
    def __init__(
        self,
        data_path: str,
        disaster: str,
        size: int,
        split: str,
        val_ratio: float,
        mean_std_dict,
        debug: bool,
    ):
        self.file_path = Path(data_path) / f"{disaster}"
        self.df_train = pd.read_csv(
            self.file_path / f"{disaster}_records.csv", encoding="unicode_escape"
        )
        # To do: test on daily t2m over US and India
        self.df = pd.read_csv(
            self.file_path / f"{disaster}_records_test.csv",
            encoding="unicode_escape",
        )
        train_len = int((1 - val_ratio) * self.df_train.shape[0])
        if split == "train":
            self.df = self.df_train[:train_len]
        elif split == "val":
            self.df = self.df_train[train_len:]
        # elif split == 'test':
        #     self.df= self.df_test
        assert self.df.shape[0] > 0, "The split has 0 record!"
        self.disno = self.df["Disno."]
        self.max_w = size
        self.max_h = size
        self.min_w = np.min(self.df.W)
        self.min_h = np.min(self.df.H)
        self.mean_std_dic = mean_std_dict


class BaseWaveDataset(data.Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        disaster: str,
        split: str,
        debug: bool = False,
    ):
        self.disaster = disaster
        self.config = settings[self.disaster]
        self.horizon = self.config["dataloader"]["horizon"]
        self.transforms = self.config["dataloader"]["transforms"]
        self.chip_size = self.config["dataloader"]["img_size"]
        self.variable = self.config["variables"]["surface"]
        self.records = Record(
            disaster=disaster,
            data_path=settings[disaster]["data_path"],
            size=self.chip_size,
            split=split,
            val_ratio=self.config["dataloader"]["val_ratio"],
            mean_std_dict=self.config["normalization"],
            debug=debug,
        )
        self.chip_metadic = self._init_meta_info(self.records, self.horizon)
        self.MetaInfo = {"disaster": self.disaster, "variable": self.variable}

    def _init_meta_info(self, records, horizon):
        meta_info = {}
        for disno in records.disno:
            chips, data, _, new_space_info = self.resize_sequence(
                records.file_path, disno, records.max_w, records.max_h
            )
            if chips.shape[0] - horizon <= 0:
                continue
            for i in range(chips.shape[0] - horizon):
                meta_info[f"{disno}-{i:04d}"] = OrderedDict(
                    {
                        "input_time": pd.to_datetime(data.time[i].values).strftime(
                            "%Y-%m-%d"
                        ),
                        "target_time": (
                            pd.to_datetime(data.time[i].values)
                            + timedelta(days=horizon)
                        ).strftime("%Y-%m-%d"),
                        "latitude": new_space_info[1],
                        "longitude": new_space_info[0],
                        "spatial_res": new_space_info[2],
                    }
                )

        return meta_info

    def resize_sequence(self, file_path, disno, max_w, max_h):
        current_data = xr.open_dataset(file_path / disno / f"{disno}.nc")
        t2m = current_data[self.variable].values.astype(np.float32)
        start_lon = current_data.longitude.values.astype(np.float32)[0]
        start_lat = current_data.latitude.values.astype(np.float32)[0]

        if self.transforms == 0:
            new_chips = crop_from_upper_left(
                t2m, self.config["dataloader"]["model_patch"]
            )
            new_space_info = (start_lon, start_lat, 0.25)

        elif self.transforms == 1:
            new_chips = np.zeros(
                (t2m.shape[0], self.chip_size, self.chip_size), dtype=np.float32
            )
            for slice in range(t2m.shape[0]):
                new_t2m_slice, new_space_info = resize_and_crop(
                    t2m[slice], max_w, max_h, start_lon, start_lat
                )
                new_chips[slice] = new_t2m_slice

        # mask
        mask_types = ["land", "soil_type", "topography"]
        # Load and resize the masks
        masks = []
        for mask_type in mask_types:
            mask = np.load(file_path / disno / f"{mask_type}_{disno}.npy")
            mask_resized = cv2.resize(
                mask,
                (t2m.shape[-1], t2m.shape[-2]),
                interpolation=cv2.INTER_NEAREST,
            )
            masks.append(mask_resized[np.newaxis, ...])
        # Concatenate masks along a new axis
        mask = np.concatenate(masks, axis=0)

        if self.transforms == 0:
            mask = crop_from_upper_left(mask, self.config["dataloader"]["model_patch"])
        elif self.transforms == 1:
            mask = np.transpose(mask, (1, 2, 0))

            mask = resize_and_crop(mask, max_w, max_h)
            mask = np.transpose(mask, (2, 0, 1))

            assert mask.shape == (3, self.chip_size, self.chip_size)

        return new_chips, current_data, mask, new_space_info

    def __len__(self):
        # return len(list(self.chip_metadic.keys()))
        length = 0
        total_index = self.records.df.index
        # for i in range(len(self.records.df)):
        for i in total_index:
            if self.records.df.loc[i]["num_frames"] - self.horizon > 0:
                length += self.records.df.loc[i]["num_frames"] - self.horizon
        return length

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data, labels, field ids, and metadata at that index
        """
        # print(list(self.chip_metadic.keys())[:4])
        key = list(self.chip_metadic.keys())[index]
        # print("length of keys", len(list(self.chip_metadic.keys())))
        disno = key[:-5]
        frame = int(key[-4:])

        new_chips, _, mask, _ = self.resize_sequence(
            self.records.file_path, disno, self.records.max_w, self.records.max_h
        )
        img = new_chips[frame]
        label = new_chips[frame + self.horizon]

        # normalize the data
        stats = self.records.mean_std_dic
        img_normalized = (img - stats["mean"]) / stats["std"]
        label_normalized = (label - stats["mean"]) / stats["std"]
        # [land, soil, topography] mean and std
        mask_means = np.array(stats["mask_means"], dtype=np.float32)
        mask_stds = np.array(stats["mask_stds"], dtype=np.float32)
        mask_means = mask_means.reshape(mask.shape[0], 1, 1)
        mask_stds = mask_stds.reshape(mask.shape[0], 1, 1)
        mask_normalized = (mask - mask_means) / mask_stds
        sample = {
            # "x" (1, 128,128)
            "x": torch.tensor(img_normalized).unsqueeze(0),
            # "y" (1, 128, 128)
            "y": torch.tensor(label_normalized).unsqueeze(0),
            # mask (3, 128, 128)
            "mask": torch.tensor(mask_normalized),
            # "x" (1, 128,128)
            "x_u": torch.tensor(img).unsqueeze(0),
            # "y" (1, 128, 128)
            "y_u": torch.tensor(label).unsqueeze(0),
            # mask (3, 128, 128)
            "mask_u": torch.tensor(mask),
            "disno": disno,
            "meta_info": self.chip_metadic[key],
        }

        # if self.transforms is not None:
        #     sample = self.transforms(sample)

        return sample

    @abstractmethod
    def bbb(self):
        raise NotImplementedError


class Era5HeatWave(BaseWaveDataset):
    def __init__(
        self,
        split: str = "train",
    ):
        super().__init__(
            disaster="heatwave",
            split=split,
        )

    def bbb(self):
        print("heatwave")


class Era5ColdWave(BaseWaveDataset):
    def __init__(
        self,
        split: str = "train",
    ):
        super().__init__(
            disaster="coldwave",
            split=split,
        )

    def bbb(self):
        print("coldwave")


if __name__ == "__main__":
    path = "/home/EarthExtreme-Bench/data/weather"
    dataset = Era5HeatWave(split="test")
    print(f"The dataset length is {len(dataset)}")
    print(list(dataset.chip_metadic.keys())[40])
    x = dataset[40]
    print(f"The dataset has {len(list(dataset.chip_metadic.keys()))} keys")
    print(x["meta_info"]["input_time"])

    import matplotlib.pyplot as plt

    plt.figure()

    plt.imshow(x["x"], vmin=torch.min(x["y"]), vmax=torch.max(x["y"]))
    plt.colorbar()
    title_time = x["meta_info"]["input_time"]
    plt.title(f"t2m_{title_time}")
    plt.savefig("test_img.png")

    plt.figure()
    plt.imshow(x["y"], vmin=torch.min(x["y"]), vmax=torch.max(x["y"]))
    plt.colorbar()
    title_time = x["meta_info"]["target_time"]
    plt.title(f"t2m_{title_time}")
    plt.savefig("test_label.png")

    plt.figure()
    plt.imshow(x["mask"][1])
    plt.title("mask")
    plt.savefig("test_mask.png")
