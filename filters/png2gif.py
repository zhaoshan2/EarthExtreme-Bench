import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from scipy.io import netcdf


def create_gif(root, image_list, gif_name, duration=0.1):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(os.path.join(root, image_name)))
    imageio.mimsave(gif_name, frames, "GIF", duration=duration)
    return


def main():
    DISASTER = "pcp"
    CURR_FOLDER_PATH = Path(__file__).parent
    # OUTPUT_DATA_DIR = (
    #    CURR_FOLDER_PATH.parent
    #    / "data"
    #    / "weather"
    #    / f"{DISASTER}-daily"
    #    / "2022-0800-MNG"
    #    / "PNG"
    # )
    OUTPUT_DATA_DIR = CURR_FOLDER_PATH / "imgs"

    filenamelist = []
    for file in os.listdir(OUTPUT_DATA_DIR):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            if DISASTER == "tropicalCyclone":
                if "_msl_" in filename:
                    filenamelist.append(filename)
            else:
                filenamelist.append(filename)
    if DISASTER == "tropicalCyclone":
        filenamelist = sorted(filenamelist, key=lambda x: int(x[29:-4]))
    else:
        filenamelist = sorted(filenamelist, key=lambda x: int(x[-7:-4]))
    print(filenamelist)
    create_gif(
        OUTPUT_DATA_DIR,
        filenamelist,
        os.path.join(OUTPUT_DATA_DIR, f"animation_{DISASTER}.gif"),
    )


if __name__ == "__main__":

    # sns.scatterplot(data=file, x="LON", y="LAT", hue="ISO_TIME",legend=False)

    main()
    # file = "/home/code/EarthExtreme-Bench/data/weather/coldwave-daily/2019-0044-DZA/topography_2019-0044-DZA.npy"
    # data = np.load(file)
    # plt.figure()
    # plt.imshow(data)
    # plt.savefig("/home/code/EarthExtreme-Bench/data/weather/coldwave-daily/2019-0044-DZA/topography_2019-0044-DZA.png")
    """
    installation error 
    python3.8.8
    
    pip install ecmwflibs
    pip install eccodes==1.3.1
    pip install cfgrib
    """
