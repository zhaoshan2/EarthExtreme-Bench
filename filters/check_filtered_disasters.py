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
    DISASTER = 'tropicalCyclone'
    OUTPUT_DATA_DIR = Path(__file__).parent.parent / f'res/{DISASTER}'



    for root, subdirs, _ in os.walk(OUTPUT_DATA_DIR):
        for subdir in subdirs:
            for file in os.listdir(os.path.join(root, subdir)):
                filename = os.fsdecode(file)
                if filename.endswith(".nc"):

                    if DISASTER == 'tropicalCyclone':
                        dataset = xr.open_dataset(os.path.join(OUTPUT_DATA_DIR, filename[:-11], filename)) # multi vars
                    else:
                        dataset = xr.open_dataset(os.path.join(OUTPUT_DATA_DIR, filename[:-3], filename)) # single vars
                    for var in dataset.data_vars:
                        data = dataset[var].values.astype(np.float32)
                        times = dataset.time
                        # print(filename, "{:.2f}".format(np.min(t2m)-273))
                        # print(filename, "{:.2f}".format(np.max(t2m) - 273))
                        for i in range(data.shape[0]):
                            plt.figure()
                            plt.imshow(data[i])
                            plt.colorbar()
                            title_time = pd.to_datetime(times[i].values).strftime('%Y-%m-%d %H:%M')
                            plt.title(f'{var}_{title_time}')
                            if DISASTER == 'tropicalCyclone':
                                EVENT_PNG_FOLDER = os.path.join(OUTPUT_DATA_DIR, filename[:-11], 'PNG') # multi-vars
                            else:
                                EVENT_PNG_FOLDER = os.path.join(OUTPUT_DATA_DIR, filename[:-3], 'PNG') # single var
                            if not os.path.exists(EVENT_PNG_FOLDER):
                                os.mkdir(EVENT_PNG_FOLDER)
                            plt.savefig(os.path.join(EVENT_PNG_FOLDER, f"{filename[:-3]}_{var}_{str(i)}.png"))


if __name__ == "__main__":

    main()

    """
    installation error 
    python3.8.8
    
    pip install ecmwflibs
    pip install eccodes==1.3.1
    pip install cfgrib
    """