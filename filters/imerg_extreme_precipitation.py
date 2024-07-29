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


def crop_patch(image, upperleft, patch_size):
    """
    Crop a patch from the image given the center and patch size.

    Parameters:
    - image: 2D NumPy array representing the image.
    - center: Tuple (a, b) representing the center of the patch.
    - patch_size: size of the patch.

    Returns:
    - Cropped patch as a 2D NumPy array.
    """
    a, b = upperleft

    # Calculate the boundaries of the patch
    end_row = min(a + patch_size, image.shape[0])
    end_col = min(b + patch_size, image.shape[1])

    # Crop the patch
    patch = image[a:end_row, b:end_col]

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
    extreme_pcp = []
    # 1. Compare the coarsen IMERG and monthly thresholds, extract the indices where the precipitation exceed the thresholds.
    # filenames_path = (
    #     DATA_FOLDER_PATH / "imerg" / "subset_GPM_3IMERGHH_07_20240723_145315_.txt"
    # )
    # with open(filenames_path, "r") as file:
    #     filenames = [line[-61:].strip() for line in file.readlines()[2:]]

    file_paths = []
    year = "2020"
    season = "winter_120102"
    month = 2
    season0 = np.load(
        f"thresholds_months/trmm7_global_wd_score_cor_seasonal_rain_perc95_month{month}.npy"
    )  # (lat, lon)
    root_folder = DATA_FOLDER_PATH / "imerg" / year / season / f"m{month:02}"
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
            # print("file_path", file_path)
    filenames = sorted(file_paths)
    print("Checking...", filenames[:3])

    for filename in tqdm(filenames):
        # filename = "3B-HHR.MS.MRG.3IMERG.20200531-S160000-E162959.0960.V07B.HDF5"
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
        # coarsened_pcp = pcp[::50, ::50]
        # or
        reshaped_pcp = pcp.reshape(
            pcp.shape[0] // scale_factor,
            scale_factor,
            pcp.shape[1] // scale_factor,
            scale_factor,
        )

        # Compute the mean over the blocks
        coarsened_pcp = reshaped_pcp.mean(
            axis=(1, 3)
        )  # coarsened_pcp (lat, lon) (20,72)

        indices = np.where(coarsened_pcp >= season0)

        if len(indices[0]) == 0:
            # print("No frames are found")
            continue
        else:
            print(indices)
            # print(
            #     f"target mean {np.mean(coarsened_pcp[indices])} is large than the threshold 2.03"
            # )
            for id in range(len(indices[0])):
                extreme_pcp.append(
                    (
                        date,
                        indices[0][id] * scale_factor,
                        indices[1][id] * scale_factor,
                    )
                )
                print(
                    "Coarsen indices (Lat, Lon) of values larger than thresholds:\n",
                    list(zip(indices[0], indices[1])),
                )
                i = 0
                # Crop the data from fine-resolution img, the center indices is 50*coarsen indices
                for idx in list(
                    zip(indices[0] * scale_factor, indices[1] * scale_factor)
                ):
                    cropped_extreme_precipitation = crop_patch(pcp, idx, scale_factor)
                #     np.save('imerg_rain_perc95_season0_time%s_patch%d.npy'%(date, i), cropped_extreme_precipitation)
                #     i += 1

    # with open(
    #    f"crops_txt/imerg_rain_perc95_{year}_month{month:02}.txt",
    #    "w",
    # ) as file:
    #    for item in extreme_pcp:
    #        file.write(f"{item}\n")

    # 2. Integrate the crops txt files as a single txt file and remove the repeated records.

    # 3. Crop the extreme pcp events from the IMERG data and save to hdf5 files
    import re

    with open(
        f"../data/weather/extrempcp-30minutes/pcp_crops.txt",
        "r",
    ) as file:
        daily_crop_filenames = [line.strip() for line in file.readlines()]
    import ast

    daily_crop_filenames = [ast.literal_eval(item) for item in daily_crop_filenames]

    scale_factor = 50
    month_to_season = {
        "01": "winter_120102",
        "02": "winter_120102",
        "03": "spring_030405",
        "04": "spring_030405",
        "05": "spring_030405",
        "06": "summer_060708",
        "07": "summer_060708",
        "08": "summer_060708",
        "09": "fall_091011",
        "10": "fall_091011",
        "11": "fall_091011",
        "12": "winter_120102",
    }
    seq_start_dates = []
    frames = []
    means = []
    start_lats, start_lons = [], []
    max_pcp = 0
    for daily_crop_filename in daily_crop_filenames:

        date_str, la_ind, lo_ind = daily_crop_filename
        # Crop on the original IMERG data, therefore the starting indices should +400
        # IMERG data has -90 to 90 deg lat data
        la_ind = la_ind + 400
        start_lats.append(la_ind)
        start_lons.append(lo_ind)

        year = date_str[:4]
        month = date_str[4:6]
        day = date_str[6:8]
        print(year, month, day)

        file_path = (
            DATA_FOLDER_PATH / "imerg" / year / month_to_season[month] / f"m{month}"
        )

        # Regular expression to extract the date from filenames
        file_pattern = re.compile(
            r"3B-HHR\.MS\.MRG\.3IMERG\.(\d{8})-S\d{6}-E\d{6}\.\d{4}\.V07B\.HDF5"
        )

        # Dictionary to store files by date
        files_by_date = {}

        # Iterate over files in the directory
        for filename in os.listdir(file_path):
            match = file_pattern.match(filename)
            if match:
                file_date = match.group(1)
                if file_date not in files_by_date:
                    files_by_date[file_date] = []
                files_by_date[file_date].append(filename)

        # Helper function to get date strings for current, previous, and next day
        def get_surrounding_dates(date_str):
            date = datetime.strptime(date_str, "%Y%m%d")
            prev_date = (date - timedelta(days=1)).strftime("%Y%m%d")
            next_date = (date + timedelta(days=1)).strftime("%Y%m%d")
            return prev_date, date_str, next_date

        # Prepare the sequence based on your data list
        sequence = []

        prev_date, curr_date, next_date = get_surrounding_dates(date_str)

        for date_key in [prev_date, curr_date, next_date]:
            if date_key in files_by_date:
                for filename in files_by_date[date_key]:
                    sequence.append(filename)
        sequence = sorted(sequence)
        seq_start_date = sequence[0][21:29]
        seq_start_dates.append(
            f"{seq_start_date[:4]}-{seq_start_date[4:6]}-{seq_start_date[6:8]}"
        )
        # Print the sequence
        sequence_precipitation = []
        for item in sequence:
            pcp_path = file_path / item
            with h5py.File(pcp_path, "r") as file:
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

            cropped_extreme_precipitation = crop_patch(
                precipitation, (la_ind, lo_ind), scale_factor
            )

            sequence_precipitation.append(
                np.expand_dims(cropped_extreme_precipitation, axis=0)
            )
        sequence_precipitation = np.concatenate(sequence_precipitation, axis=0)
        frames.append(sequence_precipitation.shape[0])
        means.append(np.mean(sequence_precipitation))
        if np.amax(sequence_precipitation) > max_pcp:
            max_pcp = np.amax(sequence_precipitation)

        # Save the file according to starting date, upperleft corder position in IMERG.
        with h5py.File(
            f"hdf_crops_n/{seq_start_date}_{la_ind:04}_{lo_ind:04}.hdf5", "w"
        ) as output_file:
            output_file.create_dataset(f"precipitation", data=sequence_precipitation)

    # 4. Write the metadata of the extreme pcps to csv file.
    import csv

    if len(seq_start_dates) == len(frames) == len(means):
        # Define the file path
        csv_file_path = "pcp_metadata.csv"

        # Write the lists to the CSV file
        with open(csv_file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["start_datetime", "run_length", "avg_pcp", "start_lat", "start_lon"]
            )  # Writing the header
            for i in range(len(seq_start_dates)):
                writer.writerow(
                    [
                        seq_start_dates[i],
                        frames[i],
                        means[i],
                        start_lats[i],
                        start_lons[i],
                    ]
                )
    else:
        print("Lists are not of the same length.")

    print(f"Max value is {max_pcp}")
