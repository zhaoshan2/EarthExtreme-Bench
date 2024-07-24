import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def regionalDailyExtremeTemperature():
    DISASTER = "coldwave"
    CURR_FOLDER_PATH = Path(__file__).parent
    OUTPUT_DATA_DIR = CURR_FOLDER_PATH.parent / "data" / "weather" / f"{DISASTER}-daily"
    for root, subdirs, _ in os.walk(OUTPUT_DATA_DIR):
        for subdir in subdirs:
            for file in os.listdir(os.path.join(root, subdir)):
                filename = os.fsdecode(file)
                if filename.endswith(".nc"):
                    dataset = xr.open_dataset(
                        os.path.join(OUTPUT_DATA_DIR, filename[:-3], filename)
                    )  # single vars
                    var = "t2m"
                    data = dataset[var].values.astype(np.float32)
                    times = dataset.time
                    if DISASTER == "coldwave":
                        extreme_data = np.min(data, axis=(-1, -2))
                    elif DISASTER == "heatwave":
                        extreme_data = np.max(data, axis=(-1, -2))
                    current_times = []
                    for i in range(len(times)):
                        current_time = pd.to_datetime(times[i].values).strftime(
                            "%Y-%m-%d"
                        )
                        current_times.append(current_time)
                    with open(
                        os.path.join(
                            OUTPUT_DATA_DIR,
                            filename[:-3],
                            f"{filename[:-3]}_sequence.csv",
                        ),
                        "w",
                        newline="",
                    ) as file:
                        writer = csv.writer(file)
                        # Write the header
                        writer.writerow(["date", "temperature"])
                        # Write the data
                        for item1, item2 in zip(current_times, extreme_data):
                            writer.writerow([item1, item2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="cfgrib")
    # if grib file, engine=cfgrib
    args = parser.parse_args()
    regionalDailyExtremeTemperature()
