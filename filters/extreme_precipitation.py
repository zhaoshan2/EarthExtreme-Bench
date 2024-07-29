"""
Filter rainfall sequence with tag "downpour", "storm", or "hail"
and generate the external link file for the new hdf5 files
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import h5py
import pandas as pd

CURR_FOLDER_PATH = Path(__file__).parent
DATA_FOLDER_PATH = (
    CURR_FOLDER_PATH.parent.parent / "data_storage_home" / "data" / "disaster"
)
INPUT_CSV_PATH = DATA_FOLDER_PATH / "input_csv" / "tassrad19_metadata.csv"
OUTPUT_FOLDER = DATA_FOLDER_PATH / "output_csv"
DISASTER = "pcp"
if __name__ == "__main__":
    """
    # Italy Radar
    rainfalls = pd.read_csv(INPUT_CSV_PATH, encoding="unicode_escape")

    print(f"In total {rainfalls.shape[0]} rainfalls")

    # Select events with tag storms etc
    storm = rainfalls[
        rainfalls["tags"].str.contains(
            "hail|storm|downpour", na=False, case=False, regex=True
        )
    ]
    # Resetting index to start from 0
    storm.reset_index(drop=True, inplace=True)

    # Renaming 'id' column to match the new index
    storm["id"] = storm.index

    storm_path = OUTPUT_FOLDER / f"{DISASTER}" / f"{DISASTER}_2010to2019.csv"
    storm.to_csv(storm_path, index=False)
    print(storm[:5])
    print(f"{storm.shape[0]} storms.")

    # According to the new meta_info, generate the external links of files.
    # Data path of the storm sequences
    OUTPUT_DATA_DIR = (
        CURR_FOLDER_PATH.parent / "data" / "weather" / f"{DISASTER}-minutes"
    )
    # Data path of the storm sequences

    metadata = pd.read_csv(storm_path, index_col="id")

    run_n = len(metadata)

    num = 0
    with h5py.File(
        os.path.join(OUTPUT_DATA_DIR, f"all_data_{DISASTER}.hdf5"), "w", libver="latest"
    ) as hdf_archive:
        # The first run
        record = metadata.loc[0]
        date_string = record["start_datetime"]
        date_object = datetime.strptime(date_string, "%m/%d/%Y %H:%M")
        hdf_archive[str(0)] = h5py.ExternalLink(
            os.path.join(OUTPUT_DATA_DIR, date_object.strftime("%Y%m%d.hdf5")), str(0)
        )
        hdf_archive.flush()

        for idx in range(run_n - 1):
            record = metadata.loc[idx]
            record_next = metadata.loc[idx + 1]

            date_string = record["start_datetime"]
            date_object = datetime.strptime(date_string, "%m/%d/%Y %H:%M")
            date_str = date_object.strftime("%Y%m%d.hdf5")

            date_string_next = record_next["start_datetime"]
            date_string_next = datetime.strptime(date_string_next, "%m/%d/%Y %H:%M")
            date_str_next = date_string_next.strftime("%Y%m%d.hdf5")

            # Check if two runs are from the same date
            if date_str_next == date_str:
                num += 1
            else:
                num = 0
            hdf_archive[str(idx + 1)] = h5py.ExternalLink(
                os.path.join(OUTPUT_DATA_DIR, date_str_next), str(num)
            )
            hdf_archive.flush()

    # Check the file links
    # all_data = h5py.File(os.path.join(OUTPUT_DATA_DIR, "all_data_storm.hdf5"), 'r', libver='latest')
    # for i, v in all_data.items():
    #     print(i, v)
    """
    # IMERG Satellites
    storm_path = CURR_FOLDER_PATH / "hdf_crops_n" / "pcp_metadata_2020to2023.csv"

    OUTPUT_DATA_DIR = CURR_FOLDER_PATH / "hdf_crops_n"
    # Data path of the storm sequences

    metadata = pd.read_csv(storm_path, index_col="id")

    run_n = len(metadata)

    num = 0
    with h5py.File(
        os.path.join(OUTPUT_DATA_DIR, f"all_imerg_{DISASTER}.hdf5"),
        "w",
        libver="latest",
    ) as hdf_archive:

        for idx in range(run_n):
            record = metadata.loc[idx]

            date_string = record["start_datetime"]
            start_lat = int(record["start_lat"])
            start_lon = int(record["start_lon"])
            date_object = datetime.strptime(date_string, "%m/%d/%Y")
            date_str = date_object.strftime(
                f"%Y%m%d_{start_lat:04}_{start_lon:04}.hdf5"
            )
            assert os.path.exists(
                os.path.join(OUTPUT_DATA_DIR, date_str)
            ), f"The linked file {os.path.join(OUTPUT_DATA_DIR, date_str)} doesn't exsit!"

            hdf_archive[str(idx)] = h5py.ExternalLink(
                os.path.join(OUTPUT_DATA_DIR, date_str), "precipitation"
            )
            assert os.path.exists(os.path.join(OUTPUT_DATA_DIR, date_str))

            hdf_archive.flush()

    # Check the file links
    all_data = h5py.File(
        os.path.join(OUTPUT_DATA_DIR, f"all_imerg_{DISASTER}.hdf5"),
        "r",
        libver="latest",
    )
    for i, v in all_data.items():
        print(i, v)
