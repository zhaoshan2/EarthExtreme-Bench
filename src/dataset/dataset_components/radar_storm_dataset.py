import sys
from datetime import timedelta

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/EarthExtreme-Bench")
from config.settings import settings


class HDFIterator:
    def __init__(
        self,
        data: h5py.File,
        metadata: pd.DataFrame,
        mask=None,
        disaster: str = "storm",
        in_seq_length: int = 1,
        out_seq_length: int = 1,
        batch_size: int = 4,
        model_patch: int = 4,
        stride: int = 1,
        shuffle: bool = True,
        filter_threshold: float = 0,
        sort_by: str = "id",
        ascending: bool = True,
        return_mask: bool = True,
        run_size: int = 25,
        return_type=np.float32,
    ):
        self.data = data
        self.disaster = disaster
        self.metadata = (
            metadata.sample(frac=1)
            if shuffle
            else metadata.sort_values(by=sort_by, ascending=ascending)
        )
        self.run_idx = 0
        self.run_seq_idx = 0
        self.batch_size = batch_size
        self.stride = stride
        self.model_patch = model_patch
        self.filter_threshold = filter_threshold
        self.run_size = run_size
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.return_type = return_type
        self.current_run = self._set_current_run()
        self.scan_max_value = settings[disaster]["normalization"]["max"]
        self.outlier_mask = mask
        self.MetaInfo = {
            "disaster": self.disaster,
            "variable": ["pcp"],
        }
        self.return_mask = return_mask

    def __len__(self):
        return self.metadata.run_length.sum() - len(self.metadata) * (self.run_size - 1)

    def __iter__(self):
        return self

    def __next__(self):
        seqs = []
        datetime_seqs = []
        lats = []
        lons = []
        # if self.exhausted:
        #     # self._octave_session.exit()
        #     raise StopIteration()

        while self.run_idx < len(self.metadata) and len(seqs) < self.batch_size:

            run_datetime = pd.to_datetime(
                self.metadata.iloc[self.run_idx]["start_datetime"]
            )

            while (
                self.run_seq_idx < (self.current_run.shape[0] - self.run_size + 1)
                and len(seqs) < self.batch_size
            ):
                frames = np.clip(
                    self.current_run[
                        self.run_seq_idx : self.run_seq_idx + self.run_size
                    ],
                    # / self.scan_max_value,
                    a_min=0,
                    a_max=self.scan_max_value,
                    # 1,
                )
                # normalize the data by mean and std (aurora is normalized within the model)
                if settings[self.disaster]['model']['name'] != "microsoft/aurora_pcp":
                    frames = (
                        frames - settings[self.disaster]["normalization"]["pcp_mean"]
                    ) / settings[self.disaster]["normalization"]["pcp_std"]
                seqs.append(frames)
                datetime_seqs.append(
                    run_datetime
                    + timedelta(
                        minutes=settings[self.disaster]["temporal_res"]
                        * self.run_seq_idx
                    )
                )
                if self.disaster == "expcp":
                    lats.append(
                        90
                        - self.metadata.iloc[self.run_idx]["start_lat"]
                        * settings[self.disaster]["spatial_res"]  # 90 to -90
                    )  # start_lat is the index on IMERG (upper left corner), correspond to the max latitude)
                    lons.append(
                        self.metadata.iloc[self.run_idx]["start_lon"]
                        * settings[self.disaster]["spatial_res"]  # 0 to 360
                    )  # start_lon is the index on IMERG (upper left corner), correspond to the min lon
                elif self.disaster == "storm":
                    lats = [
                        54.5877
                    ]  # start_lat is the latitude of upper left corner (max lat)
                    lons = [182.0715]  # start_lon is the upper left longitude (min lon)
                self.run_seq_idx += self.stride

            if len(seqs) < self.batch_size:
                self.run_seq_idx = 0
                self.run_idx += 1

                if self.run_idx < len(self.metadata):
                    self.current_run = self._set_current_run()
                else:
                    # self.exhausted = True
                    print("Exhausted!")
                    raise StopIteration()
        retval = np.stack(seqs).swapaxes(1, 0)
        retval = retval[:, :, np.newaxis, ...].astype(self.return_type)

        new_h, new_w = (retval.shape[-2] // self.model_patch) * self.model_patch, (
            retval.shape[-1] // self.model_patch
        ) * self.model_patch

        if not self.return_mask:
            # return retval, datetime_seqs

            sample = {
                "x": torch.from_numpy(
                    retval[: self.in_seq_length, :, :, :new_h, :new_w].astype(
                        self.return_type
                    )
                ),
                "y": torch.from_numpy(
                    retval[-self.out_seq_length :, :, :, :new_h, :new_w].astype(
                        self.return_type
                    )
                ),
                # datetime_seqs: the start times of the batch
                "meta_info": {
                    "input_time": datetime_seqs,
                    "latitude": lats,
                    "longitude": lons,
                },
            }

        else:
            masks = self._compute_mask(retval)
            # x: in_seq_length, B, 1, h, w -> 5, B, 1, 480, 480
            # datetime_seq: List of length 1, e.g., [Timestamp('2018-10-29 14:50:00')] the starting time
            # lats: the min latitude
            # lons: the min longitude
            sample = {
                "x": torch.from_numpy(
                    retval[: self.in_seq_length, :, :, :new_h, :new_w].astype(
                        self.return_type
                    )
                ),
                "y": torch.from_numpy(
                    retval[-self.out_seq_length :, :, :, :new_h, :new_w].astype(
                        self.return_type
                    )
                ),
                "mask": torch.from_numpy(
                    masks[-self.out_seq_length :, :, :, :new_h, :new_w].astype(
                        self.return_type
                    )
                ),
                "meta_info": {
                    "input_time": datetime_seqs,
                    "latitude": lats,
                    "longitude": lons,
                },
            }
        return sample

    def _compute_mask(self, img):
        if self.disaster == "expcp":
            mask = np.ones_like(img, dtype=bool)
            mask[np.logical_and(img < self.filter_threshold, img > 0)] = 0
        else:
            mask = np.zeros_like(img, dtype=bool)
            mask[:] = np.broadcast_to(self.outlier_mask.astype(bool), shape=img.shape)
            mask[np.logical_and(img < self.filter_threshold, img > 0)] = 0
        return mask

    def _set_current_run(self):
        metadata = self.metadata.iloc[self.run_idx]
        # print(metadata)
        return np.stack([img_slice for img_slice in self.data[str(metadata.name)][:]])


def infinite_batcher(data: h5py.File, metadata: pd.DataFrame, mask, **kwargs):
    while True:
        gen = HDFIterator(data, metadata, mask, **kwargs)
        for sample in gen:
            yield sample


if __name__ == "__main__":
    pass
