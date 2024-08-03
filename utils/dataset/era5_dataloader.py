import sys

sys.path.insert(0, "/home/EarthExtreme-Bench")

import torch
from torch.utils.data import DataLoader
from enum import Enum


class DataLoaderType(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DataPrefetcher:
    def __init__(self, loader):
        self.loader = loader
        self.dataiter = iter(loader)
        self.length = len(self.loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.__preload__()

    def __preload__(self):
        try:
            self.x, self.y, self.mask, self.disno = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            self.x, self.y, self.mask, self.disno = next(self.dataiter)

        with torch.cuda.stream(self.stream):
            self.x = self.x.cuda(non_blocking=True)
            self.y = self.y.cuda(non_blocking=True)
            self.mask = self.mask.cuda(non_blocking=True)
            self.disno = self.disno.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        self.__preload__()
        return self.x, self.y, self.mask, self.disno

    def __len__(self):
        """Return the number of images."""
        return self.length


class ERA5Dataloader:
    def __init__(
        self,
        batch_size,
        num_workers,
        pin_memory,
        horizon,
        chip_size,
        data_path,
        val_ratio=0.2,
        persistent_workers: bool = True,
        disaster="heatwave",
    ):
        super().__init__()
        self.horizon = horizon
        self.chip_size = chip_size
        self.data_path = data_path
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.disaster = disaster

    def train_dataloader(self):
        return self.get_data_loader(DataLoaderType.TRAIN, batch_size=self.batch_size)

    def val_dataloader(self):
        return self.get_data_loader(DataLoaderType.VAL, batch_size=self.batch_size)

    def test_dataloader(self):
        return self.get_data_loader(DataLoaderType.TEST, batch_size=1)

    def get_data_loader(
        self,
        data_loader_type: DataLoaderType,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        from utils.dataset.dataset_components.era5_extreme_temperature_dataset import (
            Era5ColdWave,
            Era5HeatWave,
        )
        from utils.dataset.dataset_components.era5_cyclone_dataset import TCDataset

        disasters = {
            "heatwave": Era5HeatWave,
            "coldwave": Era5ColdWave,
            "tropical cyclone": TCDataset,
        }
        if self.disaster not in disasters:
            raise ValueError(f"{self.disaster} is not a valid disaster")
        dataset = disasters[self.disaster](
            horizon=self.horizon,
            chip_size=self.chip_size,
            data_path=self.data_path,
            split=data_loader_type.value,
            val_ratio=self.val_ratio,
        )
        return (
            DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=shuffle,
                persistent_workers=self.persistent_workers,
                drop_last=drop_last,
            ),
            dataset.records,
        )


if __name__ == "__main__":
    # dataset_path ='/home/code/data_storage_home/data/pangu'
    # means, std = LoadStatic(os.path.join(dataset_path, 'aux_data'))
    # print(means.shape) #(1, 21, 1, 1)
    heatwave = ERA5Dataloader(
        2,
        0,
        False,
        28,
        128,
        val_ratio=0.5,
        data_path="/home/EarthExtreme-Bench/data/weather",
        persistent_workers=False,
    )
    loader, _ = heatwave.train_dataloader()
    print(len(loader))
    # x = next(iter(test_loader))
    for id, train_data in enumerate(loader):
        print(id, train_data["y"].shape)
    # import matplotlib.pyplot as plt

    # plt.figure()
    # plt.imshow(x['y'][0], vmin=torch.min(x['y']), vmax=torch.max(x['y']))
    # plt.colorbar()
    # title_time = x["meta_info"]['target_time']
    # plt.title(f't2m_{title_time}')
    # plt.savefig('test_label_loader.png')
    #
    # plt.figure()
    # plt.imshow(x['mask'][0,1])
    # plt.title('mask')
    # plt.savefig('test_mask_loader.png')
