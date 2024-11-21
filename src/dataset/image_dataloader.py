import sys

sys.path.insert(0, "/home/EarthExtreme-Bench")

import torch
from config.settings import settings
from schema.data_loader import DataLoaderType
from torch.utils.data import DataLoader


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


class IMGDataloader:
    def __init__(
        self,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        disaster: str = "heatwave",
    ):
        super().__init__()
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.disaster = disaster

    def train_dataloader(self):
        return self.get_data_loader(
            DataLoaderType.TRAIN,
            batch_size=settings[self.disaster]["dataloader"]["batch_size"],
        )

    def val_dataloader(self):
        return self.get_data_loader(
            DataLoaderType.VAL,
            batch_size=settings[self.disaster]["dataloader"]["batch_size"],
        )

    def test_dataloader(self):
        return self.get_data_loader(DataLoaderType.TEST, batch_size=1)

    def get_data_loader(
        self,
        data_loader_type: DataLoaderType,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = True,
    ):
        from .dataset_components.era5_cyclone_dataset import TCDataset
        from .dataset_components.era5_extreme_temperature_dataset import (
            Era5ColdWave,
            Era5HeatWave,
        )
        from .dataset_components.multispectral_dataset import HlsFire, Sentinel1Flood

        disasters = {
            "heatwave": Era5HeatWave,
            "coldwave": Era5ColdWave,
            "tropicalCyclone": TCDataset,
            "flood": Sentinel1Flood,
            "fire": HlsFire,
        }
        if self.disaster not in disasters:
            raise ValueError(f"{self.disaster} is not a valid disaster")
        dataset = disasters[self.disaster](
            split=data_loader_type.value,
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
            dataset.MetaInfo,
        )


if __name__ == "__main__":

    exevent = IMGDataloader(disaster="tropicalCyclone")
    loader, META = exevent.train_dataloader()
    print(len(loader))
    # x = next(iter(test_loader))
    for id, train_data in enumerate(loader):
        for key, val in train_data.items():
            print(key, val.shape if isinstance(val, torch.Tensor) else val)
        break
    """
    tropicalCyclone: Test loader 2422
    x torch.Size([1, 3, 2, 96, 96])
    x_upper torch.Size([1, 3, 2, 5, 96, 96])
    y torch.Size([1, 3, 1, 96, 96])
    y_upper torch.Size([1, 3, 1, 5, 96, 96]) -> [B,N,T,Z,H,W]
    mask torch.Size([1, 3, 96, 96])
    disno ['TC_2019300N41309']
    meta_info OrderedDict([('input_time', '2019-12-28 10:00'), 
                            ('target_time', '2019-12-29 00:00'), 
                            ('latitude', 140.9022606382979), 
                            ('longitude', 14.75), 
                            ('resolution', 0.10704787234042554)], 
    META  OrderedDict([('disaster', 'tropicalCyclone'), 
                            ('sur_variables', ['msl', 'u10', 'v10']), 
                            ('atm_variables', ['z', 'u', 'v']), 
                            ('pressures', [1000, 850, 700, 500, 200])])


    heatwave: Test loader 338
    x torch.Size([1, 1, 128, 128])
    y torch.Size([1, 1, 128, 128])
    mask torch.Size([1, 3, 128, 128])
    disno ['2023-0328-IND']
    meta_info OrderedDict([('input_time', ['2023-04-01']),
                           ('target_time', ['2023-04-15']), 
                           ('disaster', ['heatwave']), 
                           ('variable', ['t2m'])])
                           
    coldwave: Test loader 165
    x torch.Size([1, 1, 100, 100])
    y torch.Size([1, 1, 100, 100])
    mask torch.Size([1, 3, 100, 100])
    disno ['2023-0111-LBN']
    meta_info OrderedDict([('input_time', ['2022-12-01']), 
                           ('target_time', ['2022-12-15']), 
                           ('disaster', ['coldwave']), 
                           ('variable', ['t2m'])])

    flood: Test loader 284
    image torch.Size([1, 8, 512, 512])
    label torch.Size([1, 512, 512])
    meta_info ['20161011_Lumberton_ID_4_3']
    x torch.Size([1, 8, 512, 512])
    y torch.Size([1, 1, 512, 512])
    
    fire: Test loader: 263
    image torch.Size([1, 6, 512, 512])
    label torch.Size([1, 512, 512]) 
    meta_info ['subsetted_512x512_HLS.S30.T14RNV.2018215.v1.4']
    x torch.Size([1, 6, 512, 512])
    y torch.Size([1, 1, 512, 512]) #type: long

    """
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
