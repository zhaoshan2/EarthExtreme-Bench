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
import datetime

# Each image is normalzied for better visualization
def main():
    DATA_PATH = Path('E:/surface')
    OUTPUT_DATA_DIR = Path(__file__).parent.parent / 'res/heatwave'

    for root, subdirs, _ in os.walk(OUTPUT_DATA_DIR):
        for subdir in subdirs:
            for file in os.listdir(os.path.join(root, subdir)):
                filename = os.fsdecode(file)
                if filename.endswith(".nc"):
                    dataset = xr.open_dataset(os.path.join(OUTPUT_DATA_DIR, filename[:-3], filename))
                    t2m = dataset['t2m'].values.astype(np.float32)
                    times = dataset.time
                    print(filename, "{:.2f}".format(np.min(t2m)-273))
                    print(filename, "{:.2f}".format(np.max(t2m) - 273))
                    for i in range(t2m.shape[0]):
                        plt.figure()
                        plt.imshow(t2m[i])
                        plt.colorbar()
                        title_time = pd.to_datetime(times[i].values).strftime('%Y-%m-%d %H:%M')
                        plt.title(f't2m_{title_time}')
                        EVENT_PNG_FOLDER = os.path.join(OUTPUT_DATA_DIR, filename[:-3], 'PNG')
                        if not os.path.exists(EVENT_PNG_FOLDER):
                            os.mkdir(EVENT_PNG_FOLDER)
                        plt.savefig(os.path.join(EVENT_PNG_FOLDER, filename[:-3]+str(i)+".png"))


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