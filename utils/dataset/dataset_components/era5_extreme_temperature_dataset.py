import datetime
import json
import os
import pickle
import random
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

# import earthextremebench as eb
import xarray as xr
from torch import Tensor
from torch.utils import data
from tqdm import tqdm


def resize_and_crop(img, dst_w, dst_h):
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

    # Randomly crop a 224x224 region from the resized image
    x = random.randint(0, new_width - dst_h)
    y = random.randint(0, new_height - dst_w)

    cropped_img = resized_img[y : y + dst_h, x : x + dst_w, ...]

    return cropped_img


class Record:
    def __init__(
        self,
        disaster: str,
        size: int,
        path: str,
        split: str,
        val_ratio: float,
        debug: bool,
    ):
        self.file_path = Path(path) / f"{disaster}-daily"
        self.df_train = pd.read_csv(
            self.file_path / f"{disaster}-daily_records.csv", encoding="unicode_escape"
        )
        # To do: test on daily t2m over US and India
        self.df = pd.read_csv(
            self.file_path / f"{disaster}-daily_records_test.csv",
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

        mean_std_dict_path = self.file_path / f"{disaster}-daily_records_stats.json"
        if mean_std_dict_path.exists():
            with open(mean_std_dict_path, "r") as fp:
                mean_std_dict = json.load(fp)
        self.mean_std_dic = mean_std_dict


class BaseWaveDataset(data.Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        chip_size: int,
        horizon: int,
        disaster: str,
        data_path: str,
        split: str,
        variable: str = "t2m",
        val_ratio: float = 0.2,
        debug: bool = False,
    ):
        self.horizon = horizon
        self.transforms = None
        self.chip_size = chip_size
        self.disaster = disaster
        self.variable = variable
        self.records = Record(disaster, chip_size, data_path, split, val_ratio, debug)
        self.chip_metadic = self._init_meta_info(self.records, horizon)

    def _init_meta_info(self, records, horizon):
        meta_info = {}
        for disno in records.disno:
            chips, data, _ = self.resize_sequence(
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
                        # "latitude": data.latitude.values.astype(np.float32),
                        # "longitude": data.longitude.values.astype(np.float32),
                        "disaster": self.disaster,
                        "variable": self.variable,
                    }
                )

        return meta_info

    def resize_sequence(self, file_path, disno, max_w, max_h):
        land_masks = np.load(file_path / disno / f"land_{disno}.npy")
        soil_type_masks = np.load(file_path / disno / f"soil_type_{disno}.npy")
        topography_masks = np.load(file_path / disno / f"topography_{disno}.npy")

        mask = np.concatenate(
            (
                land_masks[np.newaxis, ...],
                soil_type_masks[np.newaxis, ...],
                topography_masks[np.newaxis, ...],
            ),
            axis=0,
        )

        mask = np.transpose(mask, (1, 2, 0))

        mask = resize_and_crop(mask, max_w, max_h)
        mask = np.transpose(mask, (2, 0, 1))

        assert mask.shape == (3, self.chip_size, self.chip_size)

        current_data = xr.open_dataset(file_path / disno / f"{disno}.nc")
        t2m = current_data[self.variable].values.astype(np.float32)

        new_chips = np.zeros(
            (t2m.shape[0], self.chip_size, self.chip_size), dtype=np.float32
        )
        for slice in range(t2m.shape[0]):
            new_t2m_slice = resize_and_crop(t2m[slice], max_w, max_h)
            new_chips[slice] = new_t2m_slice

        return new_chips, current_data, mask

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

        new_chips, _, mask = self.resize_sequence(
            self.records.file_path, disno, self.records.max_w, self.records.max_h
        )
        img = new_chips[frame]
        label = new_chips[frame + self.horizon]

        # normalize the data
        stats = self.records.mean_std_dic
        img_normalized = (img - stats[f"{self.disaster}_mean"]) / stats[
            f"{self.disaster}_std"
        ]
        label_normalized = (label - stats[f"{self.disaster}_mean"]) / stats[
            f"{self.disaster}_std"
        ]
        # land, soil, topography mean and std
        mask_means = np.array(
            [
                stats["land_mask_mean"],
                stats["soil_type_mean"],
                stats["topography_mean"],
            ],
            dtype=np.float32,
        )
        mask_stds = np.array(
            [stats["land_mask_std"], stats["soil_type_std"], stats["topography_std"]],
            dtype=np.float32,
        )
        mask_means = mask_means.reshape(mask.shape[0], 1, 1)
        mask_stds = mask_stds.reshape(mask.shape[0], 1, 1)
        mask_normalized = (mask - mask_means) / mask_stds
        sample = {
            # "x" (128,128)
            "x": torch.tensor(img_normalized),
            # "y" (128, 128)
            "y": torch.tensor(label_normalized),
            # mask (3, 128, 128)
            "mask": torch.tensor(mask_normalized),
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
        chip_size: int = 128,
        horizon: int = 14,
        data_path: str = "",
        split: str = "train",
        val_ratio: float = 0.2,
    ):
        super().__init__(
            chip_size,
            horizon,
            "heatwave",
            data_path=data_path,
            split=split,
            val_ratio=val_ratio,
        )

    def bbb(self):
        print("heatwave")


class Era5ColdWave(BaseWaveDataset):
    def __init__(
        self,
        chip_size: int = 100,
        horizon: int = 14,
        data_path: str = "",
        split: str = "train",
        val_ratio: float = 0.2,
    ):
        super().__init__(
            chip_size,
            horizon,
            "coldwave",
            data_path=data_path,
            split=split,
            val_ratio=val_ratio,
        )

    def bbb(self):
        print("coldwave")


if __name__ == "__main__":
    path = "/home/EarthExtreme-Bench/data/weather"
    dataset = Era5HeatWave(
        horizon=28, chip_size=256, data_path=path, split="test", val_ratio=0.5
    )
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
