"""
IMERG extreme precipitation extraction: selection of frames >= threshold
"""

import argparse
import gc
import os
from datetime import datetime, timedelta
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


def open_dataset(filepath):
    try:
        return xr.open_dataset(filepath)
    except Exception as e:
        print(f"Error opening {filepath}: {e}")
        return None


def extract_extreme_indices(filenames, thresholds):
    """
    Inputs:
    filenames: the imerg filenames
    thresholds: the monthly 95 percentiles values
    Returns:
        date,
        lat index on the imerg starting from 50 deg. (When crop the data from Imerg, please add 400 on this index!)
        lon index on the imerg
        latitude,
        longitude
        )
    """
    thresholds = thresholds / 24
    extreme_pcp = []
    scale_factor = 50
    # Step 1: Group filenames by date
    files_by_date = {}
    for filename in filenames:
        # Extract date from the filename
        date = filename.split(".")[4][0:8]

        if date not in files_by_date:
            files_by_date[date] = []
        files_by_date[date].append(filename)

    # Step 2: Concatenate files by date and store in a new variable
    for date, files in files_by_date.items():
        concatenated_data = None

        for file in files:
            file_path = DATA_FOLDER_PATH / "imerg" / file
            with h5py.File(file_path, "r") as hdf_in:
                hourly_pcp = hdf_in["Grid"]["precipitation"][
                    :
                ]  # shape [1, 3600, 1800] dim(time, lon, lat)
                lon = hdf_in["Grid"]["lon"][:]  # 3600: -180, 180
                lat = hdf_in["Grid"]["lat"][
                    :
                ]  # 1800: -90, 90 [-89.95, -89.85, ... 89.95]
                lat_indices = np.where((lat >= -50) & (lat <= 50))[0]
                hourly_pcp = hourly_pcp[:, :, lat_indices]
                hourly_pcp = np.where(hourly_pcp < 0, 0, hourly_pcp)

                if concatenated_data is None:
                    concatenated_data = hourly_pcp
                else:
                    concatenated_data = np.concatenate(
                        (concatenated_data, hourly_pcp), axis=0
                    )

        daily_pcp = np.mean(concatenated_data, axis=0)  # after mean: shape (3600, 1000)
        daily_pcp = np.transpose(daily_pcp)
        # assert daily_pcp.shape[0] == len(lat)
        # assert daily_pcp.shape[1] == len(lon)

        reshaped_pcp = daily_pcp.reshape(
            daily_pcp.shape[0] // scale_factor,
            scale_factor,
            daily_pcp.shape[1] // scale_factor,
            scale_factor,
        )
        # Compute the mean over the blocks
        coarsened_pcp = reshaped_pcp.mean(
            axis=(1, 3)
        )  # coarsened_pcp (lat, lon) (20,72)

        indices = np.where(coarsened_pcp > thresholds)

        if len(indices[0]) == 0:
            # print("No frames are found")
            continue
        else:
            print(
                "Coarsen indices (Lat, Lon) of values larger than thresholds:",
                list(zip(indices[0], indices[1])),
            )

            for id in range(len(indices[0])):
                extreme_pcp.append(
                    (
                        date,
                        # lat index on the imerg starting from 50 deg
                        indices[0][id] * scale_factor,
                        # lon index on the imerg
                        indices[1][id] * scale_factor,
                        # lat in ascending order -90 to 90, but a smaller index corresponds to a larger latitude
                        lat[-indices[0][id] * scale_factor - 400],
                        lon[indices[1][id] * scale_factor],
                    )
                )
    return extreme_pcp


if __name__ == "__main__":
    # 1. Compare the coarsen IMERG and monthly thresholds, extract the indices where the precipitation exceed the thresholds.
    # filenames_path = (
    #     DATA_FOLDER_PATH / "imerg" / "subset_GPM_3IMERGHH_07_20240723_145315_.txt"
    # )
    # with open(filenames_path, "r") as file:
    #     filenames = [line[-61:].strip() for line in file.readlines()[2:]]

    years = ["2020", "2021", "2022", "2023"]
    season = "spring_030405"
    month = 5
    perc = 95
    tau = np.load(
        f"thresholds_months/trmm7_global_wd_score_cor_seasonal_rain_perc{perc}_month{month}.npy"
    )  # (lat, lon)

    for year in years:
        file_paths = []
        root_folder = DATA_FOLDER_PATH / "imerg" / year / season / f"m{month:02}"
        for root, dirs, files in os.walk(root_folder):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                # print("file_path", file_path)
        filenames = sorted(file_paths)
        print("Checking...", filenames[:3])
        extreme_pcp_indices = extract_extreme_indices(filenames, thresholds=tau)

        with open(
            f"crops_daily_txt/imerg_rain_perc{perc}_{year}_month{month:02}.txt",
            "w",
        ) as file:
            for item in extreme_pcp_indices:
                file.write(f"{item}\n")

    # 2. Integrate the crops txt files as a single txt file and remove the repeated records.
    """
    Filter the frames to remove some redundant frames 
    """
    """
    files = []
    for year in range(2020, 2024):
        for month in range(1, 13):
            file_path = f"crops_txt/imerg_rain_perc{perc}_{year}_month{month:02}.txt"
            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    for line in file.readlines():
                        files.append(line.strip())
    # Set to keep track of unique entries
    unique_entries = set()

    # List to store the final results
    filtered_files = []

    # Process each file entry
    for file in files:
        # Extract parts of the file entry
        parts = file.strip("()").split(", ")
        date = parts[0].split("-")[0].strip("'")  # Get the date part
        size = parts[1]  # Get the size part
        count = parts[2]  # Get the count part

        # Create a tuple of (date, size, count)
        entry = (date, size, count)

        # Check if this entry is unique
        if entry not in unique_entries:
            unique_entries.add(entry)
            filtered_files.append(f"('{date}', {size}, {count})")

    # Print the result
    print(filtered_files)
    # Define the output file path
    output_file_path = (
        f"crops_txt_set/imerg_rain_perc{perc}_{year}_month{month:02}_unique.txt"
    )

    # Write the results to the file
    with open(output_file_path, "w") as file:
        for line in filtered_files:
            file.write(line + "\n")

    print(f"Filtered results have been saved to {output_file_path}")
    """
