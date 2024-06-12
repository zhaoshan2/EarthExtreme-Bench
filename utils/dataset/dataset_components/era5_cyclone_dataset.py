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

class Record:
    def __init__(self, disaster: str, size: int, path: str, split: str, val_ratio: float):
        self.file_path = Path(path) / f"{disaster}-hourly"
        self.df = pd.read_csv(self.file_path / f'{disaster}_surface_records_test.csv', encoding='unicode_escape')
        self.df_upper = pd.read_csv(self.file_path / f'{disaster}_upper_records_test.csv', encoding='unicode_escape')

        self.df_train = pd.read_csv(self.file_path / f'{disaster}_surface_records.csv', encoding='unicode_escape')
        self.df_upper_train = pd.read_csv(self.file_path / f'{disaster}_upper_records.csv', encoding='unicode_escape')

        train_len = int((1 - val_ratio) * self.df_train.shape[0])
        if split == 'train':
            self.df = self.df_train[:train_len]
            self.df_upper = self.df_upper_train[:train_len]
        elif split == 'val':
            self.df = self.df_train[train_len:]
            self.df_upper = self.df_upper_train[:train_len:]
        # elif split == 'test':
        #     self.df= self.df_test
        assert self.df.shape[0] > 0, "The split has 0 record!"

        self.disno = self.df['Disno.'] #TC_2019018S24033
        self.max_h = size
        self.max_w = size

        self.min_w = np.min(self.df.W)
        # To do: read from records
        self.surface_variables = ast.literal_eval(self.df['variables'][0]) #['msl', 'u10', 'v10']
        self.upper_variables = ast.literal_eval(self.df_upper['variables'][0]) #['z', 'u', 'v']
        self.pressure_levels = [1000, 850, 700, 500, 200]
        # surface mean and std
        mean_std_dict_path = self.file_path / f'{disaster}-hourly_surface_records_stats.json'
        if mean_std_dict_path.exists():
            with open(mean_std_dict_path, 'r') as fp:
                mean_std_dict = json.load(fp)
        self.mean_std_dic = {}
        self.mean_std_dic['means'] = np.array([mean_std_dict[f"{disaster}_{var}_mean"] for var in self.surface_variables],
                              dtype=np.float32)
        self.mean_std_dic['stds'] = np.array([mean_std_dict[f"{disaster}_{var}_std"] for var in self.surface_variables],
                              dtype=np.float32)
        # upper mean and std
        mean_std_dict_path_upper = self.file_path / f'{disaster}-hourly_upper_records_stats.json'
        if mean_std_dict_path_upper.exists():
            with open(mean_std_dict_path_upper, 'r') as fp:
                mean_std_dict_upper = json.load(fp)
        self.mean_std_dic_upper = {}
        # means: N, Z: (#variables x pressure levels) z, u, v at 1000, 850, ...
        self.mean_std_dic_upper['means'] = np.array([mean_std_dict_upper[f"{disaster}_{var}_{p}_mean"] for var in self.upper_variables for p in self.pressure_levels],
                              dtype=np.float32).reshape(len(self.upper_variables), len(self.pressure_levels))
        self.mean_std_dic_upper['stds'] = np.array([mean_std_dict_upper[f"{disaster}_{var}_{p}_std"] for var in self.upper_variables for p in self.pressure_levels],
                              dtype=np.float32).reshape(len(self.upper_variables), len(self.pressure_levels))

        # land, soil, topography mean and std
        self.mean_std_dic_mask = {}
        self.mean_std_dic_mask['means'] = np.array([mean_std_dict["land_mask_mean"], mean_std_dict["soil_type_mean"], mean_std_dict["topography_mean"]],
                              dtype=np.float32)
        self.mean_std_dic_mask['stds'] = np.array([mean_std_dict["land_mask_std"], mean_std_dict["soil_type_std"], mean_std_dict["topography_std"]],
                             dtype=np.float32)

class TCDataset(data.Dataset, metaclass=ABCMeta):
    def __init__(self,  chip_size: int, horizon: int, disaster: str, data_path: str, split: str, val_ratio: float = 0.2, debug: bool = False):
        self.horizon = horizon
        self.transforms = None
        self.disaster = disaster
        self.records = Record(disaster, chip_size, data_path, split, val_ratio)
        self.surface_variables = self.records.surface_variables
        self.upper_variables = self.records.upper_variables
        self.pressure_levels = self.records.pressure_levels
        self.chip_size = chip_size

        self.chip_metadic = self._init_meta_info(self.records, horizon)

    def _init_meta_info(self, records, horizon):
        meta_info = {}
        for disno in records.disno:
            chips, _, data, _ = self.resize_sequence(records.file_path, disno, records.max_w, records.max_h)
            current_record = self.records.df[self.records.df['Disno.']==disno]
            if chips.shape[1] - horizon <= 0:
                continue
            for i in range(chips.shape[1] - horizon):
                meta_info[f"{disno}-{i:04d}"] = OrderedDict({
                    "input_time": pd.to_datetime(data.time[i].values).strftime('%Y-%m-%d %H:%M'),
                    "target_time": (pd.to_datetime(data.time[i].values) + timedelta(hours=horizon)).strftime(
                        '%Y-%m-%d %H:%M'),
                    "raw_H": current_record['H'].values[0],
                    "raw_W": current_record['W'].values[0],
                    'disaster': self.disaster,
                    'variable': self.surface_variables + self.upper_variables,
                    'pressures':self.pressure_levels
                })

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
            upper_var = xr_upper[var].values.astype(np.float32)[np.newaxis, ...]  # (1, 121, 5, 22, 23) (1, T, Z, H, W)
            upper_vars.append(upper_var)
        upper = np.concatenate(upper_vars, axis=0) #(3, 121, 5, 22, 23)

        record_upper = self.records.df_upper[self.records.df_upper['Disno.']==disno]

        assert upper.shape == (len(self.upper_variables), record_upper['num_frames'].values[0], record_upper['Z'].values[0], record_upper['H'].values[0], record_upper['W'].values[0])
        # levels in ? order, require new memery space
        # upper = upper[:, :, ::-1, :, :].copy()

        # surface variables
        surface_vars = []
        for var in self.surface_variables:
            surface_var = xr_surface[var].values.astype(np.float32)[np.newaxis, ...]  # (3, 121, 22, 23)
            surface_vars.append(surface_var)
        surface = np.concatenate(surface_vars, axis=0)
        record_surface = self.records.df[self.records.df['Disno.']==disno]
        assert surface.shape == (len(self.records.surface_variables), record_surface['num_frames'].values[0], record_surface['H'].values[0], record_surface['W'].values[0])

        return upper, surface

    def resize_sequence(self, file_path, disno, max_w, max_h):
        # mask
        land_masks = np.load(file_path / disno / f'land_{disno}.npy')
        soil_type_masks = np.load(file_path / disno / f'soil_type_{disno}.npy')
        topography_masks = np.load(file_path / disno / f'topography_{disno}.npy')

        mask = np.concatenate(
            (land_masks[np.newaxis, ...], soil_type_masks[np.newaxis, ...], topography_masks[np.newaxis, ...]), axis=0)

        mask = np.transpose(mask, (1, 2, 0))

        mask = cv2.resize(mask, (max_w, max_h), interpolation=cv2.INTER_LINEAR)
        mask = np.transpose(mask, (2, 0, 1))
        assert mask.shape == (3, self.chip_size, self.chip_size)

        # data
        current_xr_surface = xr.open_dataset(file_path / disno / f'{disno}_surface.nc')
        current_xr_upper = xr.open_dataset(file_path / disno / f'{disno}_upper.nc')

        upper_data, surface_data = self.nc2numpy(current_xr_upper, current_xr_surface, disno)
        #upper_data: (N, T, Z, H, W), surface # (N, T, H, W)

        new_chips = np.zeros((surface_data.shape[0], surface_data.shape[1], max_w, max_h), dtype=np.float32)
        for i in range(surface_data.shape[0]):
            for j in range(surface_data.shape[1]):
                # cv2.resize dsize receive the parameter(width, height), different from the img size of (H, W)
                new_surface_slice = cv2.resize(surface_data[i,j], (max_w, max_h), interpolation=cv2.INTER_LINEAR)
                new_chips[i,j,:,:] = new_surface_slice

        new_upper_chips = np.zeros((upper_data.shape[0], upper_data.shape[1], upper_data.shape[2], max_w, max_h),
                             dtype=np.float32)
        for i in range(upper_data.shape[0]):
            for j in range(upper_data.shape[1]):
                for k in range(upper_data.shape[2]):
                    new_upper_slice = cv2.resize(upper_data[i,j,k], (max_w, max_h), interpolation=cv2.INTER_LINEAR)
                    new_upper_chips[i, j, k, :,:] = new_upper_slice
        # return surface_data, upper_data, current_xr_surface, mask
        return new_chips, new_upper_chips, current_xr_surface, mask

    def __len__(self):
        # return len(list(self.chip_metadic.keys()))
        length = 0
        total_index = self.records.df.index
        # for i in range(len(self.records.df)):
        for i in total_index:
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
        new_chips, new_upper_chips, _, mask = self.resize_sequence(self.records.file_path, disno, self.records.max_w, self.records.max_w)
        # upper_data: (N, T, Z, H, W), surface # (N, T, H, W)
        img = new_chips[:, frame, ...]
        img_upper = new_upper_chips[:, frame, ...]
        label = new_chips[:, frame + self.horizon, ...]
        label_upper = new_upper_chips[:, frame + self.horizon, ...]

        # normalize the data
        stats = self.records.mean_std_dic
        img_means = stats['means'][:,np.newaxis, np.newaxis, np.newaxis]
        img_stds = stats['stds'][:,np.newaxis, np.newaxis, np.newaxis]
        img_normalized = (img - img_means) / img_stds
        label_normalized = (label - img_means) / img_stds

        stats_upper = self.records.mean_std_dic_upper
        img_upper_means = stats_upper['means'][:,np.newaxis,:,np.newaxis, np.newaxis]
        img_upper_stds = stats_upper['stds'][:,np.newaxis,:,np.newaxis, np.newaxis]
        img_upper_normalized = (img_upper - img_upper_means) / img_upper_stds
        label_upper_normalized = (label_upper - img_upper_means) / img_upper_stds

        stats_mask = self.records.mean_std_dic_mask
        mask_means = stats_mask["means"].reshape(mask.shape[0], 1, 1)
        mask_stds = stats_mask["stds"].reshape(mask.shape[0], 1, 1)
        mask_normalized = (mask - mask_means) / mask_stds

        sample = {
            "x": torch.tensor(img_normalized),
            "x_upper": torch.tensor(img_upper_normalized),
            # y: shape (n, t, w, h)
            "y": torch.tensor(label_normalized),
            # y_upper: shape (n, t, z, w, h)
            "y_upper": torch.tensor(label_upper_normalized),
            # mask: shape (3, w, h)
            "mask": torch.tensor(mask_normalized),
            "disno": key[:-5],
            "meta_info": self.chip_metadic[key]
        }

        # if self.transforms is not None:
        #     sample = self.transforms(sample)

        return sample

if __name__ == "__main__":
    dataset = TCDataset(horizon=2, chip_size=128, disaster="tropicalCyclone", data_path="/home/EarthExtreme-Bench/data/weather", split='test')

    print(f"The dataset length is {len(dataset)}")
    print(list(dataset.chip_metadic.keys())[0])
    x = dataset[-1]
    print(f"The dataset has {len(list(dataset.chip_metadic.keys()))} keys")

    label = x['y'][0, 0].numpy()
    print(x["meta_info"]['raw_H'], x["meta_info"]['raw_W'])
    label = cv2.resize(label, (x["meta_info"]['raw_W'], x["meta_info"]['raw_H']), interpolation=cv2.INTER_LINEAR)

    x_upper = x['x_upper'][0,0,0].numpy()
    x_upper = cv2.resize(x_upper, (x["meta_info"]['raw_W'], x["meta_info"]['raw_H']), interpolation=cv2.INTER_LINEAR)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.figure()
    plt.imshow(label)
    plt.colorbar()
    title_time = x["meta_info"]['target_time']
    plt.title(f'surface_{title_time}_mslp')
    plt.savefig('test_label.png')

    plt.figure()
    plt.imshow(x_upper)
    plt.title('z 1000')
    plt.savefig('test_upper.png')
