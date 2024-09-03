import ast
import datetime
import json
import os
import pickle
import random
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
import sys
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

sys.path.insert(0, "/home/EarthExtreme-Bench")
from config.settings import settings


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
        self, disaster: str, size: int, path: str, split: str, val_ratio: float
    ):
        self.file_path = Path(path) / f"{disaster}"
        self.df = pd.read_csv(
            self.file_path / f"{disaster}_surface_records_test.csv",
            encoding="unicode_escape",
        )
        self.df_upper = pd.read_csv(
            self.file_path / f"{disaster}_upper_records_test.csv",
            encoding="unicode_escape",
        )

        self.df_train = pd.read_csv(
            self.file_path / f"{disaster}_surface_records.csv",
            encoding="unicode_escape",
        )
        self.df_upper_train = pd.read_csv(
            self.file_path / f"{disaster}_upper_records.csv", encoding="unicode_escape"
        )

        train_len = int((1 - val_ratio) * self.df_train.shape[0])
        if split == "train":
            self.df = self.df_train[:train_len]
            self.df_upper = self.df_upper_train[:train_len]
        elif split == "val":
            self.df = self.df_train[train_len:]
            self.df_upper = self.df_upper_train[:train_len:]
        # elif split == 'test':
        #     self.df= self.df_test
        assert self.df.shape[0] > 0, "The split has 0 record!"

        self.disno = self.df["Disno."]  # TC_2019018S24033
        self.max_h = size
        self.max_w = size

        self.min_w = np.min(self.df.W)
        # To do: read from records
        self.surface_variables = settings[disaster]["variables"][
            "surface"
        ]  # ['msl', 'u10', 'v10']
        self.upper_variables = settings[disaster]["variables"][
            "upper"
        ]  # ['z', 'u', 'v']
        self.pressure_levels = settings[disaster]["variables"]["pressure_levels"]
        # surface mean and std
        self.mean_std_dic = {}
        self.mean_std_dic["means"] = np.array(
            settings[disaster]["normalization"]["surface_means"],
            dtype=np.float32,
        )  # shape (3,)
        self.mean_std_dic["stds"] = np.array(
            settings[disaster]["normalization"]["surface_stds"],
            dtype=np.float32,
        )
        # upper mean and std
        self.mean_std_dic_upper = {}
        # means: N, Z: (#variables x pressure levels) z, u, v at 1000, 850, ...
        self.mean_std_dic_upper["means"] = np.array(
            settings[disaster]["normalization"]["upper_means"],
            dtype=np.float32,
        )  # shape (3,5)
        self.mean_std_dic_upper["stds"] = np.array(
            settings[disaster]["normalization"]["upper_stds"],
            dtype=np.float32,
        )

        # land, soil, topography mean and std
        self.mean_std_dic_mask = {}
        self.mean_std_dic_mask["means"] = np.array(
            settings[disaster]["normalization"]["masks_means"],
            dtype=np.float32,
        )
        self.mean_std_dic_mask["stds"] = np.array(
            settings[disaster]["normalization"]["masks_stds"],
            dtype=np.float32,
        )


class TCDataset(data.Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        split: str,
        debug: bool = False,
    ):
        self.disaster = "tropicalCyclone"
        self.horizon = settings[self.disaster]["dataloader"]["horizon"]
        self.chip_size = settings[self.disaster]["dataloader"]["patch_size"]
        self.transforms = None
        self.records = Record(
            self.disaster,
            self.chip_size,
            settings[self.disaster]["data_path"],
            split,
            settings[self.disaster]["dataloader"]["val_ratio"],
        )
        self.surface_variables = self.records.surface_variables
        self.upper_variables = self.records.upper_variables
        self.pressure_levels = self.records.pressure_levels
        self.chip_metadic = self._init_meta_info(self.records, self.horizon)

    def _init_meta_info(self, records, horizon):
        meta_info = {}
        for disno in records.disno:
            chips, _, data, _ = self.resize_sequence(
                records.file_path, disno, records.max_w, records.max_h
            )
            current_record = self.records.df[self.records.df["Disno."] == disno]
            if chips.shape[1] - horizon <= 0:
                continue
            for i in range(chips.shape[1] - horizon):
                meta_info[f"{disno}-{i:04d}"] = OrderedDict(
                    {
                        "input_time": pd.to_datetime(data.time[i].values).strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                        "target_time": (
                            pd.to_datetime(data.time[i].values)
                            + timedelta(hours=horizon)
                        ).strftime("%Y-%m-%d %H:%M"),
                        "raw_H": current_record["H"].values[0],
                        "raw_W": current_record["W"].values[0],
                        "disaster": self.disaster,
                        "variable": self.surface_variables + self.upper_variables,
                        "pressures": self.pressure_levels,
                    }
                )

        return meta_info

    def nc2numpy(self, xr_upper: xr.Dataset, xr_surface: xr.Dataset, disno: str):
        """
        Input
            xr.Dataset upper, surface
        Return
            numpy array upper (N, T, Z, H, W), surface # (N, T, H, W)
        """
        # upper variables
        assert self.pressure_levels == list(xr_upper.level.values)
        upper_vars = []
        for var in self.upper_variables:
            upper_var = xr_upper[var].values.astype(np.float32)[
                np.newaxis, ...
            ]  # (1, 121, 5, 22, 23) (1, T, Z, H, W)
            upper_vars.append(upper_var)
        upper = np.concatenate(upper_vars, axis=0)  # (3, 121, 5, 22, 23)

        record_upper = self.records.df_upper[self.records.df_upper["Disno."] == disno]

        assert upper.shape == (
            len(self.upper_variables),
            record_upper["num_frames"].values[0],
            record_upper["Z"].values[0],
            record_upper["H"].values[0],
            record_upper["W"].values[0],
        )
        # levels in ? order, require new memery space
        # upper = upper[:, :, ::-1, :, :].copy()

        # surface variables
        surface_vars = []
        for var in self.surface_variables:
            surface_var = xr_surface[var].values.astype(np.float32)[
                np.newaxis, ...
            ]  # (3, 121, 22, 23)
            surface_vars.append(surface_var)
        surface = np.concatenate(surface_vars, axis=0)
        record_surface = self.records.df[self.records.df["Disno."] == disno]
        assert surface.shape == (
            len(self.records.surface_variables),
            record_surface["num_frames"].values[0],
            record_surface["H"].values[0],
            record_surface["W"].values[0],
        )

        return upper, surface

    def resize_sequence(self, file_path, disno, max_w, max_h):
        # mask
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

        # data
        current_xr_surface = xr.open_dataset(file_path / disno / f"{disno}_surface.nc")
        current_xr_upper = xr.open_dataset(file_path / disno / f"{disno}_upper.nc")

        upper_data, surface_data = self.nc2numpy(
            current_xr_upper, current_xr_surface, disno
        )
        # upper_data: (N, T, Z, H, W), surface # (N, T, H, W)

        new_chips = np.zeros(
            (surface_data.shape[0], surface_data.shape[1], max_w, max_h),
            dtype=np.float32,
        )
        for i in range(surface_data.shape[0]):
            for j in range(surface_data.shape[1]):
                # cv2.resize dsize receive the parameter(width, height), different from the img size of (H, W)
                new_surface_slice = resize_and_crop(surface_data[i, j], max_w, max_h)
                new_chips[i, j, :, :] = new_surface_slice

        new_upper_chips = np.zeros(
            (
                upper_data.shape[0],
                upper_data.shape[1],
                upper_data.shape[2],
                max_w,
                max_h,
            ),
            dtype=np.float32,
        )
        for i in range(upper_data.shape[0]):
            for j in range(upper_data.shape[1]):
                for k in range(upper_data.shape[2]):
                    new_upper_slice = resize_and_crop(upper_data[i, j, k], max_w, max_h)
                    new_upper_chips[i, j, k, :, :] = new_upper_slice
        # return surface_data, upper_data, current_xr_surface, mask
        return new_chips, new_upper_chips, current_xr_surface, mask

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
        new_chips, new_upper_chips, _, mask = self.resize_sequence(
            self.records.file_path, disno, self.records.max_w, self.records.max_w
        )
        # upper_data: (N, T, Z, H, W), surface # (N, T, H, W)
        img = new_chips[:, frame, ...]  # (N, W, H)
        img_upper = new_upper_chips[:, frame, ...]  # (N, Z, W, H)
        label = new_chips[:, frame + self.horizon, ...]  # (N, W, H)
        label_upper = new_upper_chips[:, frame + self.horizon, ...]  # (N, Z, W, H)

        # normalize the data
        stats = self.records.mean_std_dic
        img_means = stats["means"][:, np.newaxis, np.newaxis]  # shape (3,1,1)
        img_stds = stats["stds"][:, np.newaxis, np.newaxis]
        img_normalized = (img - img_means) / img_stds
        label_normalized = (label - img_means) / img_stds

        stats_upper = self.records.mean_std_dic_upper
        img_upper_means = stats_upper["means"][
            :, :, np.newaxis, np.newaxis
        ]  # shape (3,5,1,1)
        img_upper_stds = stats_upper["stds"][:, :, np.newaxis, np.newaxis]
        img_upper_normalized = (img_upper - img_upper_means) / img_upper_stds
        label_upper_normalized = (label_upper - img_upper_means) / img_upper_stds

        stats_mask = self.records.mean_std_dic_mask
        mask_means = stats_mask["means"].reshape(mask.shape[0], 1, 1)
        mask_stds = stats_mask["stds"].reshape(mask.shape[0], 1, 1)
        mask_normalized = (mask - mask_means) / mask_stds

        sample = {
            # x: shape (n, w, h)
            "x": torch.tensor(img_normalized),
            # x: shape (n, z, w, h)
            "x_upper": torch.tensor(img_upper_normalized),
            # y: shape (n, w, h)
            "y": torch.tensor(label_normalized),
            # y_upper: shape (n, z, w, h)
            "y_upper": torch.tensor(label_upper_normalized),
            # mask: shape (3, w, h)
            "mask": torch.tensor(mask_normalized),
            "disno": key[:-5],
            "meta_info": self.chip_metadic[key],
        }

        # if self.transforms is not None:
        #     sample = self.transforms(sample)

        return sample


if __name__ == "__main__":
    dataset = TCDataset(
        split="test",
    )

    print(f"The dataset length is {len(dataset)}")
    print(list(dataset.chip_metadic.keys())[0])
    x = dataset[-1]
    print(f"The dataset has {len(list(dataset.chip_metadic.keys()))} keys")

    label = x["y"][0].numpy()
    print(x["meta_info"]["raw_H"], x["meta_info"]["raw_W"])
    # label = cv2.resize(label, (x["meta_info"]['raw_W'], x["meta_info"]['raw_H']), interpolation=cv2.INTER_LINEAR)

    x_upper = x["x_upper"][0, 0].numpy()
    # x_upper = cv2.resize(x_upper, (x["meta_info"]['raw_W'], x["meta_info"]['raw_H']), interpolation=cv2.INTER_LINEAR)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(label)
    plt.colorbar()
    title_time = x["meta_info"]["target_time"]
    plt.title(f"surface_{title_time}_mslp")
    plt.savefig("test_label.png")

    plt.figure()
    plt.imshow(x_upper)
    plt.title("z 1000")
    plt.savefig("test_upper.png")
