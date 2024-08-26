# 3. Crop the extreme pcp events from the IMERG data and save to hdf5 files
import re
import os
from datetime import datetime, timedelta
import h5py
import numpy as np
from pathlib import Path


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


def main(filepath):
    with open(
        filepath,
        "r",
    ) as file:
        lines = file.readlines()
    import ast

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
    seq_end_dates = []
    frames = []
    means = []
    start_lats, start_lons = [], []
    max_pcp = 0
    for line in lines:
        la_ind = eval(line.split("Lat: ")[1].split(",")[0].strip())  # Int

        # Extract the longitude
        lo_ind = eval(line.split("Lon: ")[1].split(",")[0].strip())  # Int

        # Extract the dates list
        dates_str = eval(line.split("Dates: ")[1].strip())  # List

        # Crop on the original IMERG data, therefore the starting indices should +400
        # IMERG data has -90 to 90 deg lat data
        la_ind = la_ind + 400
        start_lats.append(la_ind)
        start_lons.append(lo_ind)
        # Find the year, month, day of the starting date
        year = dates_str[0][:4]
        month = dates_str[0][4:6]
        day = dates_str[0][6:8]
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
        def get_periods(date_str):
            start_date_str = date_str[0]
            end_date_str = date_str[-1]
            start_date = datetime.strptime(start_date_str, "%Y%m%d")
            end_date = datetime.strptime(end_date_str, "%Y%m%d")
            dates = [
                (start_date + timedelta(days=i)).strftime("%Y%m%d")
                for i in range((end_date - start_date).days + 1)
            ]
            return dates

        # Prepare the sequence based on your data list
        sequence = []

        dates = get_periods(dates_str)
        for date_key in dates:
            if date_key in files_by_date:
                for filename in files_by_date[date_key]:
                    sequence.append(filename)
        sequence = sorted(sequence)
        seq_start_date = sequence[0][21:29]
        print("Start date: ", seq_start_date)

        seq_end_date = sequence[-1][21:29]
        print("End date: ", seq_end_date)
        seq_start_dates.append(
            f"{seq_start_date[:4]}-{seq_start_date[4:6]}-{seq_start_date[6:8]}"
        )
        seq_end_dates.append(
            f"{seq_end_date[:4]}-{seq_end_date[4:6]}-{seq_end_date[6:8]}"
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
            f"hdf_crops_daily/{seq_start_date}_{seq_end_date}_{la_ind:04}_{lo_ind:04}.hdf5",
            "w",
        ) as output_file:
            output_file.create_dataset(f"precipitation", data=sequence_precipitation)

    # 4. Write the metadata of the extreme pcps to csv file.
    import csv

    if len(seq_start_dates) == len(frames) == len(means):
        # Define the file path
        csv_file_path = "pcp_daily_metadata.csv"

        # Write the lists to the CSV file
        with open(csv_file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "start_datetime",
                    "end_datetime",
                    "run_length",
                    "avg_pcp",
                    "start_lat",
                    "start_lon",
                ]
            )  # Writing the header
            for i in range(len(seq_start_dates)):
                writer.writerow(
                    [
                        seq_start_dates[i],
                        seq_end_dates[i],
                        frames[i],
                        means[i],
                        start_lats[i],
                        start_lons[i],
                    ]
                )
    else:
        print("Lists are not of the same length.")

    print(f"Max value is {max_pcp}")


if __name__ == "__main__":
    CURR_FOLDER_PATH = Path(__file__).parent
    DATA_FOLDER_PATH = CURR_FOLDER_PATH.parent.parent / "data_storage_home"
    filepath = "../filters/crops_daily_txt/imerg_rain_perc95_2020_3days.txt"
    main(filepath)
