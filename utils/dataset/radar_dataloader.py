import sys
sys.path.insert(0, '/home/EarthExtreme-Bench')
import h5py
import pandas as pd
from utils.dataset.dataset_components.radar_storm_dataset import HDFIterator, infinite_batcher
import os
import cv2

class RADARDataloader():
    def __init__(
            self,
            data_path: str,
            in_seq_length: int = 5,
            out_seq_length: int = 20,
            batch_size: int = 4,
            val_ratio:float = 0.2,
            num_workers: int = 8,
            pin_memory: bool = True,
            persistent_workers: bool = True,
            disaster = "storm",
            stride: int = 1,
            filter_threshold = 0,
            return_mask: bool = True,
            run_size: int = 25,
            scan_max_value: float = 52.5
    ):
        super().__init__()

        self.data_path = data_path
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.disaster = disaster
        self.stride = stride
        self.filter_threshold = filter_threshold
        self.return_mask = return_mask
        self.run_size = run_size
        self.scan_max_value = scan_max_value

        self.all_data = h5py.File(os.path.join(data_path, 'all_data_storm.hdf5'), 'r', libver='latest')
        self.outlier_mask = cv2.imread(os.path.join(data_path, 'taasrad_mask.png'), 0)

    def train_dataloader(self):
        metadata = pd.read_csv(os.path.join(self.data_path, 'storm_2010to2019.csv'), index_col='id')
        metadata['start_datetime'] = pd.to_datetime(metadata['start_datetime'])
        metadata['end_datetime'] = pd.to_datetime(metadata['end_datetime'])

        metadata = metadata.loc[metadata['start_datetime'] >= '2010-01-01']
        metadata = metadata.loc[metadata['start_datetime'] < '2018-12-31']
        split_idx = int(len(metadata) * (1- self.val_ratio))
        train_meta = metadata.iloc[:split_idx]

        train_loader = infinite_batcher(data = self.all_data,
                                     metadata = train_meta,
                                     mask = self.outlier_mask,
                                     in_seq_length = self.in_seq_length,
                                     out_seq_length = self.out_seq_length,
                                     batch_size = self.batch_size,
                                     stride = self.stride,
                                     shuffle = True,
                                     filter_threshold = self.filter_threshold,
                                     return_mask = self.return_mask,
                                     run_size = self.run_size)
        return train_loader

    def val_dataloader(self):
        metadata = pd.read_csv(os.path.join(self.data_path, 'storm_2010to2019.csv'), index_col='id')
        metadata['start_datetime'] = pd.to_datetime(metadata['start_datetime'])
        metadata['end_datetime'] = pd.to_datetime(metadata['end_datetime'])

        metadata = metadata.loc[metadata['start_datetime'] >= '2010-01-01']
        metadata = metadata.loc[metadata['start_datetime'] < '2018-12-31']
        split_idx = int(len(metadata) * (1 - self.val_ratio))
        val_meta = metadata.iloc[split_idx:]
        val_loader = HDFIterator(data = self.all_data,
                                 metadata = val_meta,
                                 mask = self.outlier_mask,
                                 in_seq_length=self.in_seq_length,
                                 out_seq_length=self.out_seq_length,
                                 batch_size = self.batch_size,
                                 stride = self.stride,
                                 shuffle = False,
                                 filter_threshold = self.filter_threshold,
                                 return_mask = self.return_mask,
                                 run_size = self.run_size)
        return val_loader

    def test_dataloader(self):
        metadata = pd.read_csv(os.path.join(self.data_path, 'storm_2010to2019.csv'), index_col='id')
        metadata['start_datetime'] = pd.to_datetime(metadata['start_datetime'])
        metadata['end_datetime'] = pd.to_datetime(metadata['end_datetime'])

        metadata = metadata.loc[metadata['start_datetime'] >= '2019-01-01']
        metadata = metadata.loc[metadata['start_datetime'] < '2019-12-31']

        test_loader = HDFIterator(data = self.all_data,
                                 metadata = metadata,
                                 mask = self.outlier_mask,
                                 in_seq_length=self.in_seq_length,
                                 out_seq_length=self.out_seq_length,
                                 batch_size = self.batch_size,
                                 stride = self.stride,
                                 shuffle = False,
                                 filter_threshold = self.filter_threshold,
                                 return_mask = self.return_mask,
                                 run_size = self.run_size)
        return test_loader

if __name__ == "__main__":
    import torch
    from datetime import timedelta
    import numpy as np
    dataset_path ='/home/EarthExtreme-Bench/data/weather/storm-minutes'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = RADARDataloader(data_path=dataset_path, batch_size=1, val_ratio=0.0024)
    loader = dataloader.val_dataloader()

    sample = next(loader)

    data, label, mask, datetime_clip = sample['x'], sample['y'], sample['mask'], sample['datetime_seqs']
    print(datetime_clip)
    print(torch.amax(data))
    print(torch.amax(label))
    import matplotlib.pyplot as plt

    # for i in range(data.shape[0]):
    #     plt.figure()
    #     plt.imshow(mask[i,0,0] * data[i,0,0], vmin=0, vmax=1, cmap="magma")
    #     plt.colorbar()
    #     title_time = datetime_clip[0] + timedelta(minutes=5*i)
    #     plt.title(f'precipitation_{title_time}')
    #     plt.savefig(f'radar_img_{i}.png', dpi=200)
