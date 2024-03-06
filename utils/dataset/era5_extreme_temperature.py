import datetime
from abc import ABCMeta, abstractmethod
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
        self.file_path = EXT_PATH / f"{disaster}-daily"
        self.df = pd.read_csv(self.file_path / f'{disaster}-daily_records.csv', encoding='unicode_escape')
        self.disno = self.df['Disno.']
        self.max_w = size
        self.max_h = size
        self.min_w = np.min(self.df.W)
        self.min_h = np.min(self.df.H)


class BaseWaveDataset(data.Dataset, metaclass=ABCMeta):
    def __init__(self, chip_size: int, horizon: int, disaster: str, variable: str = "t2m"):
        self.horizon = horizon
        self.transforms = None
        self.chip_size = chip_size
        self.disaster = disaster
        self.variable = variable
        self.records = Record(disaster, chip_size)
        self.chip_metadic = self._init_meta_info(self.records, horizon)

    def _init_meta_info(self, records, horizon):
        meta_info = {}
        for disno in records.disno:
            chips, data, _ = self.resize_sequence(records.file_path, disno, records.max_w, records.max_h)
            if chips.shape[0] - horizon <= 0:
                continue
            for i in range(chips.shape[0] - horizon):
                meta_info[f"{disno}-{i:04d}"] = OrderedDict({
                    "input_time": pd.to_datetime(data.time[i].values).strftime('%Y-%m-%d'),
                    "target_time": (pd.to_datetime(data.time[i].values) + timedelta(days=horizon)).strftime(
                        '%Y-%m-%d'),
                    "latitude": data.latitude,
                    "longitude": data.longitude,
                })

        return meta_info

    def resize_sequence(self, file_path, disno, max_w, max_h):
        current_data = xr.open_dataset(file_path / disno / f'{disno}.nc')
        land_masks = np.load(file_path / disno / f'land_{disno}.npy')
        soil_type_masks = np.load(file_path / disno / f'soil_type_{disno}.npy')
        topography_masks = np.load(file_path / disno / f'topography_{disno}.npy')

        mask = np.concatenate(
            (land_masks[np.newaxis, ...], soil_type_masks[np.newaxis, ...], topography_masks[np.newaxis, ...]), axis=0)

        mask = np.transpose(mask, (1, 2, 0))

        mask = cv2.resize(mask, (max_w, max_h), interpolation=cv2.INTER_CUBIC)
        mask = np.transpose(mask, (2, 0, 1))
        assert mask.shape == (3, self.chip_size, self.chip_size)

        t2m = current_data[self.variable].values.astype(np.float32)

        new_chips = np.zeros((t2m.shape[0], self.chip_size, self.chip_size), dtype=np.float32)
        for slice in range(t2m.shape[0]):
            new_t2m_slice = cv2.resize(t2m[slice], (max_w, max_h), interpolation=cv2.INTER_CUBIC)
            new_chips[slice] = new_t2m_slice

        return new_chips, current_data, mask

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
        # print(list(self.chip_metadic.keys())[:4])
        key = list(self.chip_metadic.keys())[index]
        # print("length of keys", len(list(self.chip_metadic.keys())))
        disno = key[:-5]
        frame = int(key[-4:])
        new_chips, _, mask = self.resize_sequence(self.records.file_path, disno, self.records.max_w, self.records.max_h)
        img = new_chips[frame]
        label = new_chips[frame + self.horizon]

        sample = {
            "image": img,
            "x": torch.tensor(img),
            "y": torch.tensor(label),
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


class Era5HeatWave(BaseWaveDataset):
    def __init__(self, chip_size: int = 128, horizon: int = 14):
        super().__init__(chip_size, horizon, "heatwave")

    def bbb(self):
        print("heatwave")


class Era5ColdWave(BaseWaveDataset):
    def __init__(self, chip_size: int = 100, horizon: int = 14):
        super().__init__(chip_size, horizon, "coldwave")

    def bbb(self):
        print("coldwave")

# class Era5HeatWave(data.Dataset):
#     """Geo wrapper around heatwave dataset."""
#
#     def __init__(
#             self,
#             root: str = "data",
#             chip_size: int = 128,
#             stride: int = 64,
#             horizon: int = 14,
#             disaster: str = "heatwave",
#             variable: str = "t2m",
#     ) -> None:
#         """Initialize a new CV4A Kenya Crop Type Dataset instance.
#
#         Args:
#             root: root directory where dataset can be found
#             chip_size: size of chips
#             stride: spacing between chips, if less than chip_size, then there
#                 will be overlap between chips
#             bands: the subset of bands to load
#             download: if True, download dataset and store it in the root directory
#             api_key: a RadiantEarth MLHub API key to use for downloading the dataset
#             checksum: if True, check the MD5 of the downloaded files (may be slow)
#             verbose: if True, print messages when new tiles are loaded
#
#         Raises:
#             RuntimeError: if ``download=False`` but dataset is missing or checksum fails
#         """
#         super().__init__()
#         self.horizon = horizon
#         self.transforms = None
#         self.chip_size = chip_size
#         self.disaster = disaster
#         self.variable = variable
#         self.load_records(EXT_PATH)
#         self._chip_metainfo()
#
#     def load_records(self, path):
#         self.file_path = os.path.join(path, f"{self.disaster}-daily")
#         self.records = pd.read_csv(os.path.join(self.file_path, f'{self.disaster}-daily_records.csv'),
#                                    encoding='unicode_escape')
#         self.num_frames = self.records.num_frames.sum()
#         self.disno = self.records['Disno.']
#         self.max_w = self.chip_size
#         self.max_h = self.chip_size
#         self.min_w = np.min(self.records.W)
#         self.min_h = np.min(self.records.H)
#
#     def resize_sequence(self, disno):
#         curret_data = xr.open_dataset((os.path.join(self.file_path, disno, f'{disno}.nc')))
#         land_masks = np.load((os.path.join(self.file_path, disno, f'land_{disno}.npy')))
#         soil_type_masks = np.load((os.path.join(self.file_path, disno, f'soil_type_{disno}.npy')))
#         topography_masks = np.load((os.path.join(self.file_path, disno, f'topography_{disno}.npy')))
#
#         mask = np.concatenate(
#             (land_masks[np.newaxis, ...], soil_type_masks[np.newaxis, ...], topography_masks[np.newaxis, ...]), axis=0)
#
#         mask = np.transpose(mask, (1, 2, 0))
#
#         mask = cv2.resize(mask, (self.max_w, self.max_h), interpolation=cv2.INTER_CUBIC)
#         mask = np.transpose(mask, (2, 0, 1))
#         assert mask.shape == (3, self.chip_size, self.chip_size)
#
#         t2m = curret_data[self.variable].values.astype(np.float32)
#
#         new_chips = np.zeros((t2m.shape[0], self.chip_size, self.chip_size), dtype=np.float32)
#         for slice in range(t2m.shape[0]):
#             new_t2m_slice = cv2.resize(t2m[slice], (self.max_w, self.max_h), interpolation=cv2.INTER_CUBIC)
#             new_chips[slice] = new_t2m_slice
#
#         return new_chips, curret_data, mask
#
#     def _chip_metainfo(self, path: str = os.getcwd() + '/meta'):
#         # Path(path).mkdir(exist_ok=True, parents=True)
#         self.chip_metadic = {}
#         for disno in self.disno:
#             chips, data, _ = self.resize_sequence(disno)
#             if chips.shape[0] - self.horizon <= 0:
#                 continue
#             for i in range(chips.shape[0] - self.horizon):
#                 self.chip_metadic[f"{disno}-{i:04d}"] = OrderedDict({
#                     "input_time": pd.to_datetime(data.time[i].values).strftime('%Y-%m-%d'),
#                     "target_time": (pd.to_datetime(data.time[i].values) + timedelta(days=self.horizon)).strftime(
#                         '%Y-%m-%d'),
#                     "latitude": data.latitude,
#                     "longitude": data.longitude,
#                 })
#
#         return self.chip_metadic
#
#     def transforms(self, img):
#         pass
#
#     def __len__(self):
#         # return len(list(self.chip_metadic.keys()))
#         length = 0
#         for i in range(len(self.records)):
#             if self.records.loc[i]['num_frames'] - self.horizon > 0:
#                 length += (self.records.loc[i]['num_frames'] - self.horizon)
#         return length
#
#     def __getitem__(self, index: int) -> Dict[str, Tensor]:
#         """Return an index within the dataset.
#
#         Args:
#             index: index to return
#
#         Returns:
#             data, labels, field ids, and metadata at that index
#         """
#         print(list(self.chip_metadic.keys())[:4])
#         key = list(self.chip_metadic.keys())[index]
#         print("length of keys", len(list(self.chip_metadic.keys())))
#         disno = key[:-5]
#         frame = int(key[-4:])
#         new_chips, curret_data, mask = self.resize_sequence(disno)
#         img = new_chips[frame]
#         label = new_chips[frame + self.horizon]
#
#         sample = {
#             "image": img,
#             "x": torch.tensor(img),
#             "y": torch.tensor(label),
#             "mask": torch.tensor(mask),
#             "disno": key[:-5],
#             "meta_info": self.chip_metadic[key]
#         }
#
#         if self.transforms is not None:
#             sample = self.transforms(sample)
#
#         return sample
#
#
# class Era5ColdWave(Era5HeatWave):
#     """Geo wrapper around heatwave dataset."""
#
#     def __init__(
#             self,
#             root: str = "data",
#             chip_size: int = 100,
#             stride: int = 64,
#             horizon: int = 14,
#             variable: str = "t2m",
#             disaster: str = 'coldwave',
#
#     ) -> None:
#         """Initialize a new CV4A Kenya Crop Type Dataset instance.
#
#         Args:
#             root: root directory where dataset can be found
#             chip_size: size of chips
#             stride: spacing between chips, if less than chip_size, then there
#                 will be overlap between chips
#             bands: the subset of bands to load
#             download: if True, download dataset and store it in the root directory
#             api_key: a RadiantEarth MLHub API key to use for downloading the dataset
#             checksum: if True, check the MD5 of the downloaded files (may be slow)
#             verbose: if True, print messages when new tiles are loaded
#
#         Raises:
#             RuntimeError: if ``download=False`` but dataset is missing or checksum fails
#         """
#         super().__init__()  # load_records self._chip_metainfo()
#         # self.chip_metainfo: heatwave metainfo
#         self.horizon = horizon
#         self.chip_size = chip_size
#         self.disaster = disaster
#         self.variable = variable
#         self.load_records(EXT_PATH)  # new coldwave disno  heatwave metainfo
#         self._chip_metainfo()

    # def __getitem__(self, index: int) -> Dict[str, Tensor]:
    #     """Return an index within the dataset.
    #
    #     Args:
    #         index: index to return
    #
    #     Returns:
    #         data, labels, field ids, and metadata at that index
    #     """
    #     print(list(self.chip_metadic.keys())[:4])
    #     key = list(self.chip_metadic.keys())[index]
    #     print("length of keys", len(list(self.chip_metadic.keys())))
    #     disno = key[:-5]
    #     frame = int(key[-4:])
    #     new_chips, curret_data, mask = self.resize_sequence(disno)
    #     img = new_chips[frame]
    #     label = new_chips[frame + self.horizon]
    #
    #     sample = {
    #         "image": img,
    #         "x": torch.tensor(img),
    #         "y": torch.tensor(label),
    #         "mask": torch.tensor(mask),
    #         "disno": key[:-5],
    #         "meta_info": self.chip_metadic[key]
    #     }
    #
    #     if self.transforms is not None:
    #         sample = self.transforms(sample)
    #
    #     return sample


if __name__ == "__main__":
    dataset = Era5HeatWave(horizon=28)
    print(f"The dataset length is {len(dataset)}")
    print(list(dataset.chip_metadic.keys())[0])
    x = dataset[0]
    print(f"The dataset has {len(list(dataset.chip_metadic.keys()))} keys")
    print(x["meta_info"]['input_time'])

    import matplotlib.pyplot as plt

    plt.figure()

    plt.imshow(x['image'], vmin=torch.min(x['y']), vmax=torch.max(x['y']))
    plt.colorbar()
    title_time = x["meta_info"]['input_time']
    plt.title(f't2m_{title_time}')
    plt.savefig('test_img.png')

    plt.figure()
    plt.imshow(x['y'], vmin=torch.min(x['y']), vmax=torch.max(x['y']))
    plt.colorbar()
    title_time = x["meta_info"]['target_time']
    plt.title(f't2m_{title_time}')
    plt.savefig('test_label.png')

    plt.figure()
    plt.imshow(x['mask'][1])
    plt.title('mask')
    plt.savefig('test_mask.png')
