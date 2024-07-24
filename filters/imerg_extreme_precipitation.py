"""
IMERG extreme precipitation extraction: selection of frames >= threshold
"""

import argparse
import gc
import os
from datetime import datetime
from pathlib import Path

import dask
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

CURR_FOLDER_PATH = Path(__file__).parent
DATA_FOLDER_PATH = CURR_FOLDER_PATH.parent.parent / "data_storage_home"
DISASTER = "precipitation"


def crop_patch(image, center, patch_size):
    """
    Crop a patch from the image given the center and patch size.

    Parameters:
    - image: 2D NumPy array representing the image.
    - center: Tuple (a, b) representing the center of the patch.
    - patch_size: size of the patch.

    Returns:
    - Cropped patch as a 2D NumPy array.
    """
    a, b = center
    # Compute half sizes
    half_size = patch_size // 2

    # Calculate the boundaries of the patch
    start_row = max(a - half_size, 0)
    end_row = min(a + half_size, image.shape[0])
    start_col = max(b - half_size, 0)
    end_col = min(b + half_size, image.shape[1])

    # Crop the patch
    patch = image[start_row:end_row, start_col:end_col]

    assert patch.shape == (
        patch_size,
        patch_size,
    ), f"Expected patch size ({patch_size},{patch_size}), but find the patch {patch.shape}. Please check the dimensions!"

    return patch


def open_dataset(filepath):
    try:
        return xr.open_dataset(filepath)
    except Exception as e:
        print(f"Error opening {filepath}: {e}")
        return None


if __name__ == "__main__":

    filenames_path = (
        DATA_FOLDER_PATH / "imerg" / "subset_GPM_3IMERGHH_07_20240723_145315_.txt"
    )
    with open(filenames_path, "r") as file:
        filenames = [line[-61:].strip() for line in file.readlines()[2:]]
    season0 = np.load(
        "trmm7_global_wd_score_cor_seasonal_rain_perc95_season2.npy"
    )  # (lat, lon)
    # If the threshold is less than 2, then convert to 2 mm/h
    season0 = np.where(season0 == 0, 2, season0)

    extreme_pcp = []
    for filename in filenames:
        # filename = "3B-HHR.MS.MRG.3IMERG.20200531-S160000-E162959.0960.V07B.HDF5"
        if "202008" in filename:
            file_path = DATA_FOLDER_PATH / "imerg" / filename
            date = filename[-39:-24]

            scale_factor = 50
            with h5py.File(file_path, "r") as file:
                # Get the dataset
                precipitation = file["Grid"]["precipitation"][
                    :
                ]  # shape [1, 3600, 1800] dim(time, lon, lat)
                precipitation = np.where(precipitation < 0, 0, precipitation).squeeze(
                    0
                )  # fill the missing value as 0 and disgard the first dim

                precipitation = np.transpose(
                    precipitation
                )  # conver the dim (lon, lat) to (lat, lon)
                lon = file["Grid"]["lon"][:]  # 3600: -180, 180
                lat = file["Grid"]["lat"][:]  # 1800: -90, 90
                time = file["Grid"]["time"][:]

            assert precipitation.shape[0] == len(lat)
            assert precipitation.shape[1] == len(lon)

            lat_indices = np.where((lat >= -50) & (lat <= 50))[0]
            pcp = precipitation[lat_indices, :]

            # plt.figure()
            # plt.imshow(pcp)
            # plt.colorbar()
            # plt.savefig("pcp.png")
            coarsened_pcp = pcp[::50, ::50]
            # reshaped_pcp = pcp.reshape(
            #     pcp.shape[0] // scale_factor, scale_factor,
            #     pcp.shape[1] // scale_factor, scale_factor
            # )
            #
            # # Compute the mean over the blocks
            # coarsened_pcp = reshaped_pcp.mean(axis=(1, 3)) # coarsened_pcp (lat, lon) (20,72)
            # plt.figure()
            # plt.imshow(coarsened_pcp)
            # plt.colorbar()
            # plt.savefig("corasen.png")
            indices = np.where(coarsened_pcp > season0)

            if len(indices[0]) == 0:
                # print("No frames are found")
                continue
            else:
                # print( indices[0][0], indices[1][0])
                # To do: current only return the 1st element, wrong!
                print("date", date)
                extreme_pcp.append(
                    (date, indices[0][0] * scale_factor, indices[1][0] * scale_factor)
                )
    #             print("Coarsen Indices (Lat, Lon) of values larger than thresholds:\n", list(zip(indices[0], indices[1])))
    #             # i = 0
    #             # # Crop the data from fine-resolution img, the center indices is 50*coarsen indices
    #             # for idx in list(zip(indices[0]*scale_factor, indices[1]*scale_factor)):
    #             #     print(idx)
    #             #     cropped_extreme_precipitation = crop_patch(pcp, idx, scale_factor)
    #             #     np.save('imerg_rain_perc95_season0_time%s_patch%d.npy'%(date, i), cropped_extreme_precipitation)
    #             #     i += 1
    print(extreme_pcp)

    with open("imerg_rain_perc95_season2_aug_2020.txt", "w") as file:
        for item in extreme_pcp:
            file.write(f"{item}\n")
