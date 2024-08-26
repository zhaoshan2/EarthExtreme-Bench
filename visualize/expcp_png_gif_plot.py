import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
import h5py
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


def get_order(filename):
    """
    Return: the i index of the file
    """
    return int(filename[10:-4])


def main():
    DISASTER = "pcp"
    CURR_FOLDER_PATH = Path(__file__).parent

    OUTPUT_DATA_DIR = CURR_FOLDER_PATH / "figures/expcp_pngs"

    filenamelist = []
    for file in os.listdir(OUTPUT_DATA_DIR):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            filenamelist.append(filename)
    filenamelist = sorted(filenamelist, key=get_order)
    # print(filenamelist)
    create_gif(
        OUTPUT_DATA_DIR,
        filenamelist,
        os.path.join(OUTPUT_DATA_DIR, f"animation_{DISASTER}.gif"),
    )


if __name__ == "__main__":
    # sns.scatterplot(data=file, x="LON", y="LAT", hue="ISO_TIME",legend=False)

    filename = "../filters/hdf_crops_daily/20200720_20200722_0900_1000.hdf5"
    seq = h5py.File(filename, "r", libver="latest")
    seq = seq["precipitation"]
    max_value = np.amax(seq)
    print("The max value of the sequence is ", max_value)
    # seq = np.clip(seq / max_value, 0, 1)

    print("seq has the shape of", seq.shape)
    for i in range(seq.shape[0]):
        plt.figure()
        plt.imshow(seq[i, :, :], vmin=0, vmax=47.65, cmap="magma")
        cbar = plt.colorbar()
        cbar.set_label("mm/h", rotation=270, labelpad=15)
        start_date_str = datetime.strptime("20200720", "%Y%m%d")
        title_time = start_date_str + timedelta(minutes=30 * i)
        plt.title(f"precipitation_{title_time}")
        plt.savefig(f"figures/expcp_pngs/IMERG_img_{i}.png", dpi=200)
        plt.close()
    main()

    """
    installation error 
    python3.8.8

    pip install ecmwflibs
    pip install eccodes==1.3.1
    pip install cfgrib
    """
