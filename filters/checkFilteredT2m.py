import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
import xarray as xr
import pandas as pd
import os
from pathlib import Path
import argparse
from datetime import datetime, timedelta


# Each image is normalzied for better visualization
def main():
    DATA_PATH = Path('E:/surface')
    MASK_PATH = DATA_PATH / 'soil_type.npy'
    OUTPUT_DATA_DIR = Path(__file__).parent.parent / 'res/coldwaves'

    masks = np.load(MASK_PATH).astype(np.float32)
    plt.figure()
    plt.imshow(masks)
    plt.show()
    # for file in os.listdir(OUTPUT_DATA_DIR):
    #     filename = os.fsdecode(file)
    #     if filename.endswith(".nc"):
    #         dataset = xr.open_dataset(os.path.join(OUTPUT_DATA_DIR,filename))
    #         t2m = dataset['t2m'].values.astype(np.float32)
    #         print(filename, "{:.2f}".format(np.min(t2m)-273))
    #         for i in range(t2m.shape[0]):
    #             plt.figure()
    #             plt.imshow(t2m[i])
    #             plt.colorbar()
    #             plt.title('t2m')
    #             EVENT_PNG_FOLDER = os.path.join(OUTPUT_DATA_DIR,filename[:-3])
    #             if not os.path.exists(EVENT_PNG_FOLDER):
    #                 os.mkdir(EVENT_PNG_FOLDER)
    #             plt.savefig(os.path.join(EVENT_PNG_FOLDER,filename[:-3]+str(i)+".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', default="cfgrib")
    # if grib file, engine=cfgrib
    args = parser.parse_args()
    main()

    """
    installation error 
    python3.8.8
    
    pip install ecmwflibs
    pip install eccodes==1.3.1
    pip install cfgrib
    """