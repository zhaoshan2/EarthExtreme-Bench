import h5py
import numpy as np
import pandas as pd
from datetime import timedelta
import torch
class HDFIterator:
    def __init__(self, data: h5py.File, metadata: pd.DataFrame, mask, in_seq_length=1, out_seq_length=1,
                 batch_size=4, stride=1, shuffle=True,
                 filter_threshold=0, sort_by="id", ascending=True, return_mask=True, run_size=25,
                 scan_max_value=52.5, return_type=np.float32):
        self.data = data
        self.metadata = metadata.sample(frac=1) if shuffle \
                                                else metadata.sort_values(by=sort_by, ascending=ascending)
        self.run_idx = 0
        self.run_seq_idx = 0
        self.batch_size = batch_size
        self.stride = stride
        self.filter_threshold = filter_threshold
        self.run_size = run_size
        self.scan_max_value = scan_max_value
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.return_type = return_type

        self.current_run = self._set_current_run()
        self.outlier_mask = None
        if return_mask:
            self.outlier_mask = mask

    def __len__(self):
        return self.metadata.run_length.sum() - len(self.metadata) * (self.run_size - 1)

    def __iter__(self):
        return self

    def __next__(self):
        seqs = []
        datetime_seqs = []
        # if self.exhausted:
        #     # self._octave_session.exit()
        #     raise StopIteration()

        while self.run_idx < len(self.metadata) and len(seqs) < self.batch_size:

            run_datetime = pd.to_datetime(self.metadata.iloc[self.run_idx]['start_datetime'])

            while self.run_seq_idx < (self.current_run.shape[0] - self.run_size + 1) and len(seqs) < self.batch_size:
                frames = np.clip(self.current_run[self.run_seq_idx:self.run_seq_idx + self.run_size]
                                 / self.scan_max_value, 0, 1)
                seqs.append(frames)
                datetime_seqs.append(run_datetime+timedelta(minutes=5*self.run_seq_idx))

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
        if self.outlier_mask is None:
            # return retval, datetime_seqs
            sample = {
                "x": torch.from_numpy(retval[:self.in_seq_length, ...].astype(self.return_type)),
                "y": torch.from_numpy(retval[-self.out_seq_length:, ...].astype(self.return_type)),
                "datetime_seqs": datetime_seqs
            }

        else:
            # return retval, datetime_seqs, self._compute_mask(retval)
            masks = self._compute_mask(retval)
            # x: in_seq_length, B, 1, h, w -> 5, B, 1, 480, 480
            # datetime_seq: List of length 1, e.g., [Timestamp('2018-10-29 14:50:00')] the starting time
            sample = {
                "x": torch.from_numpy(retval[:self.in_seq_length, ...].astype(self.return_type)),
                "y": torch.from_numpy(retval[-self.out_seq_length:,...].astype(self.return_type)),
                "mask": torch.from_numpy(masks[-self.out_seq_length:, ...].astype(self.return_type)),
                "datetime_seqs": datetime_seqs
            }
        return sample

    def _compute_mask(self, img):
        mask = np.zeros_like(img, dtype=np.bool)
        mask[:] = np.broadcast_to(self.outlier_mask.astype(np.bool), shape=img.shape)
        mask[np.logical_and(img < self.filter_threshold, img > 0)] = 0
        return mask

    def _set_current_run(self):
        metadata = self.metadata.iloc[self.run_idx]
        print(metadata)
        return np.stack([img_slice for img_slice in self.data[str(metadata.name)][:]])

def infinite_batcher(data: h5py.File, metadata: pd.DataFrame, mask, **kwargs):
    while True:
        gen = HDFIterator(data, metadata, mask, **kwargs)
        for sample in gen:
            yield sample

if __name__ == "__main__":
    pass
