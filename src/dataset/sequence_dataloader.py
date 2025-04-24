import sys

sys.path.insert(0, "/home/EarthExtreme-Bench")
import os
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import pandas as pd

from .dataset_components.radar_storm_dataset import HDFIterator, infinite_batcher

sys.path.insert(0, "/home/EarthExtreme-Bench")
from config.settings import settings

class SEQDataloader:
    def __init__(
        self,
        disaster="storm",
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        filter_threshold=0,
    ):
        super().__init__()

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.disaster = disaster
        # if self.disaster not in settings:
        #     raise ValueError(f"{self.disaster} is not a valid disaster")
        self.settings = settings[disaster]["dataloader"]
        self.data_path = Path(settings[disaster]["data_path"]) / disaster
        self.filter_threshold = filter_threshold
        self.return_mask = True  # if self.disaster == "flood" else False
        self.all_data = h5py.File(
            os.path.join(self.data_path, f"all_data_{disaster}.hdf5"),
            "r",
            libver="latest",
        )
        self.outlier_mask = None
        if self.return_mask:
            self.outlier_mask = cv2.imread(
                os.path.join(self.data_path, "taasrad_mask.png"), 0
            )
        assert self._check_date_valid(), "Data split contains invalid date string"

    def _check_date_valid(self):
        try:
            for i in [0, 1]:
                for set in ["train", "val", "test"]:
                    datetime.strptime(self.settings[f"{set}_date"][i], "%Y-%m-%d")
            return True
        except ValueError:
            # If an error occurs, the date is invalid
            return False

    def train_dataloader(self):

        metadata = pd.read_csv(
            os.path.join(self.data_path, f"{self.disaster}_metadata.csv"),
            index_col="id",
        )
        metadata["start_datetime"] = pd.to_datetime(metadata["start_datetime"])
        metadata["end_datetime"] = pd.to_datetime(metadata["end_datetime"])

        metadata = metadata.loc[
            metadata["start_datetime"] >= self.settings["train_date"][0]
        ]
        metadata = metadata.loc[
            metadata["start_datetime"] < self.settings["train_date"][1]
        ]

        train_loader = infinite_batcher(
            data=self.all_data,
            metadata=metadata,
            mask=self.outlier_mask,
            disaster=self.disaster,
            in_seq_length=self.settings["in_seq_length"],
            out_seq_length=self.settings["out_seq_length"],
            batch_size=self.settings["batch_size"],
            model_patch=self.settings["model_patch"],
            stride=self.settings["stride"],
            shuffle=True,
            filter_threshold=self.filter_threshold,
            return_mask=self.return_mask,
            run_size=self.settings["run_size"],
        )
        return train_loader

    def val_dataloader(self):
        metadata = pd.read_csv(
            os.path.join(self.data_path, f"{self.disaster}_metadata.csv"),
            index_col="id",
        )
        metadata["start_datetime"] = pd.to_datetime(metadata["start_datetime"])
        metadata["end_datetime"] = pd.to_datetime(metadata["end_datetime"])

        metadata = metadata.loc[
            metadata["start_datetime"] >= self.settings["val_date"][0]
        ]
        metadata = metadata.loc[
            metadata["start_datetime"] < self.settings["val_date"][1]
        ]

        val_loader = HDFIterator(
            data=self.all_data,
            metadata=metadata,
            mask=self.outlier_mask,
            disaster=self.disaster,
            in_seq_length=self.settings["in_seq_length"],
            out_seq_length=self.settings["out_seq_length"],
            batch_size=self.settings["batch_size"],
            model_patch=self.settings["model_patch"],
            stride=self.settings["stride"],
            shuffle=False,
            filter_threshold=self.filter_threshold,
            return_mask=self.return_mask,
            run_size=self.settings["run_size"],
        )
        return val_loader

    def test_dataloader(self):
        metadata = pd.read_csv(
            os.path.join(self.data_path, f"{self.disaster}_metadata.csv"),
            index_col="id",
        )
        metadata["start_datetime"] = pd.to_datetime(metadata["start_datetime"])
        metadata["end_datetime"] = pd.to_datetime(metadata["end_datetime"])

        metadata = metadata.loc[
            metadata["start_datetime"] >= self.settings["test_date"][0]
        ]
        metadata = metadata.loc[
            metadata["start_datetime"] < self.settings["test_date"][1]
        ]

        test_loader = HDFIterator(
            data=self.all_data,
            metadata=metadata,
            mask=self.outlier_mask,
            disaster=self.disaster,
            in_seq_length=self.settings["in_seq_length"],
            out_seq_length=self.settings["out_seq_length"],
            batch_size=self.settings["batch_size"],
            model_patch=self.settings["model_patch"],
            stride=self.settings["stride"],
            shuffle=False,
            filter_threshold=self.filter_threshold,
            return_mask=self.return_mask,
            run_size=self.settings["run_size"],
        )
        return test_loader


if __name__ == "__main__":
    from datetime import timedelta

    import numpy as np
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = SEQDataloader(disaster="storm")
    loader = dataloader.val_dataloader()

    for id, train_data in enumerate(loader):
        for key, val in train_data.items():
            print(key, val.shape if isinstance(val, torch.Tensor) else val)
        break
    """
    expcp: 
    start_datetime    2022-07-01 00:00:00
    end_datetime      2022-07-03 00:00:00
    run_length                        144
    avg_cell_value                1.95812
    start_lat                        1100 --> lat_min, lat_max = 85 - lat * 0.1,  90 - lat * 0.1
    start_lon                        2900 --> lon_min, lon_max = lon * 0.1 - 180, lon * 0.1 - 175
    Name: 774, dtype: object
    x torch.Size([5, 2, 1, 50, 50]) # input_sequence_length, batch_size, channels, width, height
    y torch.Size([20, 2, 1, 50, 50]) # output_sequence_length, batch_size, channels, width, height
    meta_info [Timestamp('2022-07-01 00:00:00'), Timestamp('2022-07-01 00:05:00')

    storm:
    start_datetime    2018-01-08 00:00:00
    end_datetime      2018-01-08 23:55:00
    run_length                        288
    avg_cell_value               1.208008
    tags                  rain storm snow
    Name: 632, dtype: object
    x torch.Size([2, 4, 1, 480, 480])
    y torch.Size([5, 4, 1, 480, 480])
    meta_info [Timestamp('2018-01-08 00:00:00'), Timestamp('2018-01-08 00:05:00'), Timestamp('2018-01-08 00:10:00'), Timestamp('2018-01-08 00:15:00')]
    """
    # sample = next(loader)
    # data, label, datetime_clip = (
    #     sample["x"],
    #     sample["y"],
    #     sample["datetime_seqs"],
    # )
    # print(datetime_clip)
    # print(torch.amax(data))
    # print(torch.amax(label))
    # import matplotlib.pyplot as plt
    #
    # for i in range(2):
    #     plt.figure()
    #     plt.imshow(label[i, 0, 0], vmin=0, vmax=1, cmap="magma")
    #     plt.colorbar()
    #     title_time = datetime_clip[0] + timedelta(minutes=30 * i)
    #     plt.title(f"precipitation_{title_time}")
    #     plt.savefig(f"img_{i}.png", dpi=200)
