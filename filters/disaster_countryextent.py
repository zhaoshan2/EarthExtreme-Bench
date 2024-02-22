import numpy as np
import xarray as xr
import csv
import json
import pandas as pd
import os
from pathlib import Path
import argparse

CURR_FOLDER_PATH = Path(__file__).parent
DATA_FOLDER_PATH = CURR_FOLDER_PATH.parent.parent / 'data_storage_home' / 'data' / 'disaster'
DISASTER = "coldwave" #heatwave, coldwave, tropicalCyclone
INPUT_CSV_PATH = DATA_FOLDER_PATH / "output_csv" / f"{DISASTER}_2019to2022.csv"
OUTPUT_CSV_PATH = DATA_FOLDER_PATH / "output_csv" / f"{DISASTER}_2019to2022_pos.csv"
OUTPUT_JSON_PATH = DATA_FOLDER_PATH / "output_csv" / "country_extent.json"

if __name__ == "__main__":

    disaster = pd.read_csv(INPUT_CSV_PATH, encoding='unicode_escape')

    print(f"{disaster.shape[0]} {DISASTER}")

    countries = disaster['ISO'].unique()
    print(countries)
    #lon_min lon_max lat_min lat_max

    with open(OUTPUT_JSON_PATH) as f:
        countries2lonlat_DIC = json.load(f)
        print(countries2lonlat_DIC)

    disaster['min_lon'] = ""
    disaster['max_lon'] = ""
    disaster['min_lat'] = ""
    disaster['max_lat'] = ""

    for country in countries:
        position = countries2lonlat_DIC[country]
        assert position[0] <= position[1]
        assert position[2] <= position[3]
        disaster.min_lon[disaster.ISO == country] = position[0]
        disaster.max_lon[disaster.ISO == country] = position[1]
        disaster.min_lat[disaster.ISO == country] = position[2]
        disaster.max_lat[disaster.ISO == country] = position[3]

    disaster.to_csv(OUTPUT_CSV_PATH, index=False)


