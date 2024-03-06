import datetime
from abc import ABCMeta, abstractmethod
import ast
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import rasterio
import torch
from torch import Tensor
from tqdm import tqdm
import pandas as pd
# import earthextremebench as eb
import xarray as xr
import pickle
from collections import OrderedDict
from torch.utils import data

import json
from datetime import datetime, timedelta

EXT_PATH = Path('/home/code/EarthExtreme-Bench/data/weather')


class Record:
    def __init__(self, disaster: str, size: int):
        self.file_path = EXT_PATH / f"{disaster}"
        self.df = pd.read_csv(self.file_path / f'{disaster}_surface_records.csv', encoding='unicode_escape')
        self.df_upper = pd.read_csv(self.file_path / f'{disaster}_upper_records.csv', encoding='unicode_escape')
        self.disno = self.df['Disno.']
        self.max_w = size
        self.min_w = np.min(self.df.W)
        # To do: read from records
        self.surface_variables = ast.literal_eval(self.df['variables'][0])
        self.upper_variables = ast.literal_eval(self.df_upper['variables'][0]) #['z', 'u', 'v']
        # self.upper_variables = ['z', 'u', 'v']
        # self.surface_variables = ['msl', 'u10', 'v10']

class BaseShortDataset(data.Dataset, metaclass=ABCMeta):
    def __init__(self, chip_size: int, horizon: int, disaster: str):
        self.horizon = horizon
        self.transforms = None
        self.disaster = disaster
        self.records = Record(disaster, chip_size)
        self.surface_variables = self.records.surface_variables
        self.upper_variables = self.records.upper_variables
        self.chip_metadic = self._init_meta_info(self.records, horizon)

    def _init_meta_info(self, records, horizon):
        meta_info = {}
        for disno in records.disno:
            chips, _, data, _ = self.resize_sequence(records.file_path, disno, records.max_w)
            if chips.shape[1] - horizon <= 0:
                continue
            for i in range(chips.shape[1] - horizon):
                meta_info[f"{disno}-{i:04d}"] = OrderedDict({
                    "input_time": pd.to_datetime(data.time[i].values).strftime('%Y-%m-%d %H:%M'),
                    "target_time": (pd.to_datetime(data.time[i].values) + timedelta(hours=horizon)).strftime(
                        '%Y-%m-%d %H:%M'),
                    "latitude": data.latitude,
                    "longitude": data.longitude,
                })

        return meta_info

    def nc2numpy(self, xr_upper: xr.Dataset, xr_surface: xr.Dataset, disno: str):
        """
        Input
            xr.Dataset upper, surface
        Return
            numpy array upper, surface
        """
        # upper variables
        upper_vars = []
        for var in self.upper_variables:
            upper_var = xr_upper[var].values.astype(np.float32)[np.newaxis, ...]  # (1, 121, 5, 22, 23)
            upper_vars.append(upper_var)
        upper = np.concatenate(upper_vars, axis=0) #(3, 121, 5, 22, 23)

        record_upper = self.records.df_upper[self.records.df_upper['Disno.']==disno]

        assert upper.shape == (len(self.upper_variables), record_upper['num_frames'].values[0], record_upper['Z'].values[0], record_upper['W'].values[0], record_upper['H'].values[0])
        # levels in descending order, require new memery space
        # upper = upper[:, ::-1, :, :].copy()

        # surface variables
        surface_vars = []
        for var in self.surface_variables:
            surface_var = xr_surface[var].values.astype(np.float32)[np.newaxis, ...]  # (3, 121, 22, 23)
            surface_vars.append(surface_var)
        surface = np.concatenate(surface_vars, axis=0)
        record_surface = self.records.df[self.records.df['Disno.']==disno]

        assert surface.shape == (len(self.records.surface_variables), record_surface['num_frames'].values[0], record_surface['W'].values[0], record_surface['H'].values[0])

        return upper, surface

    def resize_sequence(self, file_path, disno, max_w):
        # mask
        land_masks = np.load(file_path / disno / f'land_{disno}.npy')
        soil_type_masks = np.load(file_path / disno / f'soil_type_{disno}.npy')
        topography_masks = np.load(file_path / disno / f'topography_{disno}.npy')

        mask = np.concatenate(
            (land_masks[np.newaxis, ...], soil_type_masks[np.newaxis, ...], topography_masks[np.newaxis, ...]), axis=0)

        mask = np.transpose(mask, (1, 2, 0))

        mask = cv2.resize(mask, (max_w, max_w), interpolation=cv2.INTER_CUBIC)
        mask = np.transpose(mask, (2, 0, 1))
        assert mask.shape == (3, max_w, max_w)

        # data
        current_xr_surface = xr.open_dataset(file_path / disno / f'{disno}_surface.nc')
        current_xr_upper = xr.open_dataset(file_path / disno / f'{disno}_upper.nc')

        upper_data, surface_data = self.nc2numpy(current_xr_upper, current_xr_surface, disno)

        new_chips = np.zeros((surface_data.shape[0], surface_data.shape[1], max_w, max_w), dtype=np.float32)
        for i in range(surface_data.shape[0]):
            for j in range(surface_data.shape[1]):
                new_surface_slice = cv2.resize(surface_data[i,j], (max_w, max_w), interpolation=cv2.INTER_CUBIC)
                new_chips[i,j,:,:] = new_surface_slice

        new_upper_chips = np.zeros((upper_data.shape[0], upper_data.shape[1], upper_data.shape[2], max_w, max_w),
                             dtype=np.float32)
        for i in range(upper_data.shape[0]):
            for j in range(upper_data.shape[1]):
                for k in range(upper_data.shape[2]):
                    new_upper_slice = cv2.resize(upper_data[i,j,k], (max_w, max_w), interpolation=cv2.INTER_CUBIC)
                    new_upper_chips[i, j, k, :,:] = new_upper_slice

        return new_chips, new_upper_chips, current_xr_surface, mask

    def __len__(self):
        # return len(list(self.chip_metadic.keys()))
        length = 0
        for i in range(len(self.records.df)):
            if self.records.df.loc[i]['num_frames'] - self.horizon > 0:
                length += (self.records.df.loc[i]['num_frames'] - self.horizon)
        return length

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data, labels, field ids, and metadata at that index
        """
        print(list(self.chip_metadic.keys())[:4])
        key = list(self.chip_metadic.keys())[index]
        print("length of keys", len(list(self.chip_metadic.keys())))
        disno = key[:-5]
        frame = int(key[-4:])
        new_chips, new_upper_chips, _, mask = self.resize_sequence(self.records.file_path, disno, self.records.max_w)
        img = new_chips[:, frame, ...]
        img_upper = new_upper_chips[:, frame, ...]
        label = new_chips[:, frame + self.horizon, ...]
        label_upper = new_upper_chips[:, frame + self.horizon, ...]

        sample = {
            "image": img,
            "x": torch.tensor(img),
            "x_upper": torch.tensor(img_upper),
            # y: shape (n, w, h)
            "y": torch.tensor(label),
            # y_upper: shape (n, z, w, h)
            "y_upper": torch.tensor(label_upper),
            # mask: shape (3, w, h)
            "mask": torch.tensor(mask),
            "disno": key[:-5],
            "meta_info": self.chip_metadic[key]
        }

        # if self.transforms is not None:
        #     sample = self.transforms(sample)

        return sample

    @abstractmethod
    def bbb(self):
        raise NotImplementedError

class Era5TropicalCyclone(BaseShortDataset):
    def __init__(self, chip_size: int = 128, horizon: int = 2):
        super().__init__(chip_size, horizon, "tropicalCyclone")

    def bbb(self):
        print("tropical cyclone")
if __name__ == "__main__":
    dataset = Era5TropicalCyclone(horizon=2)

    print(f"The dataset length is {len(dataset)}")
    print(list(dataset.chip_metadic.keys())[0])
    x = dataset[-1]
    print(f"The dataset has {len(list(dataset.chip_metadic.keys()))} keys")
    print(x["meta_info"]['input_time'])

    import matplotlib.pyplot as plt

    plt.figure()

    plt.imshow(x['image'][0], vmin=torch.min(x['y'][0]), vmax=torch.max(x['y'][0]))
    plt.colorbar()
    title_time = x["meta_info"]['input_time']
    plt.title(f'surface_{title_time}')
    plt.savefig('test_img.png')

    plt.figure()
    plt.imshow(x['y'][0], vmin=torch.min(x['y'][0]), vmax=torch.max(x['y'][0]))
    plt.colorbar()
    title_time = x["meta_info"]['target_time']
    plt.title(f'surface_{title_time}')
    plt.savefig('test_label.png')

    plt.figure()
    plt.imshow(x['mask'][2])
    plt.title('mask')
    plt.savefig('test_mask.png')
