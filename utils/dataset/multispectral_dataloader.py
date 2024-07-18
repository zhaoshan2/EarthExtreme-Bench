from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '/home/EarthExtreme-Bench')
from utils.dataset.dataset_components.multispectral_dataset import MultispectralDataset
from utils import score
class MultiSpectralDataloader():
    def __init__(
            self,
            batch_size,
            num_workers,
            pin_memory,
            chip_size,
            data_path,
            val_ratio: float=0.2,
            persistent_workers: bool =True,
            transform: str = None,
            diaster: str="fire"
    ):
        super().__init__()
        self.chip_size = chip_size
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.val_ratio = val_ratio
        self.transform = transform
        self.diaster = diaster

    def train_dataloader(self):
        data_train = MultispectralDataset(data_path=self.data_path, split="train", val_ratio = self.val_ratio, chip_size=self.chip_size, bands=None, transform=self.transform, disaster=self.diaster)

        return DataLoader(
            dataset= data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=self.persistent_workers,
            drop_last=False
        )

    def val_dataloader(self):
        data_val = MultispectralDataset(data_path=self.data_path, split="val", val_ratio = self.val_ratio, chip_size=self.chip_size, bands=None, transform=self.transform, disaster=self.diaster)

        return DataLoader(
            dataset=data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=self.persistent_workers,
            drop_last=False
        )

    def test_dataloader(self):
        data_test = MultispectralDataset(data_path=self.data_path, split="test", val_ratio = self.val_ratio, chip_size=self.chip_size, bands=None, transform=self.transform, disaster=self.diaster)

        return DataLoader(
            dataset=data_test,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=self.persistent_workers,
            drop_last=False
        )
    # sample['x']: (B, 6, 512, 512)
    # sample['y']: (B, 1, 512, 512)

if __name__ == "__main__":
    from pathlib import Path
    import numpy as np
    # Get the absolute path of the current file
    current_file_path = Path(__file__).resolve()
    burned = MultiSpectralDataloader(2,
            0,
            False,
            chip_size=512,
            data_path=current_file_path.parent.parent.parent/'data/eo/flood',
            val_ratio=0.2,
            persistent_workers=False,
            transform='resize',
            diaster="flood")

    loader = burned.test_dataloader()

    print(len(loader))
    x = next(iter(loader))
    # for id, test_data in enumerate(loader):
    #     print(id, test_data['y'].shape, test_data['x'].shape)
    print("shape", x['y'].shape, x['x'].shape)
    mask = x['y'][0,0]
    img = (score.tensor2uint( x['x'][0]))
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(mask)
    plt.colorbar()
    plt.title('mask')
    plt.savefig('test_label_loader.png',dpi=200)

    # CHW ->  HWC
    vis = np.moveaxis(img, 0,-1)
    """
    Visualization
    https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/composites/
    """
    # SWIR color (6,4,3)
    r = vis[...,0]
    g = vis[...,1]
    b = vis[...,2]

    rgb = np.stack((r,g,b), axis=-1)
    plt.figure()
    plt.imshow(rgb)
    plt.colorbar()
    plt.title('merged')
    plt.savefig('test_input_loader.png', dpi=200)

