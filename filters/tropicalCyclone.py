"""
Filter disasters
1. Disaster group: Natural
2. Disaster subgroup: Hydrological, Meteorological, Climatological
3. Disaster Type: compute their temporal duration
"""
import numpy as np
import xarray as xr
import csv
import pandas as pd
import os
from pathlib import Path
import argparse
# CURR_FOLDER_PATH = Path(__file__).parent
DATA_FOLDER_PATH = Path(__file__).parent.parent.parent.parent / "E:/datasets/disasters"
INPUT_CSV_PATH = DATA_FOLDER_PATH / "output/coldwave2015_2019.csv"
OUTPUT_CSV_PATH = DATA_FOLDER_PATH / "output/coldwave2015_2019_pos.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="E:/datasets/disasters")

    args = parser.parse_args()

    disaster = pd.read_csv(INPUT_CSV_PATH, encoding='unicode_escape')

    print(f"{disaster.shape[0]} coldwaves")

    countries = disaster['Country'].unique()
    print(countries)
    #lon_min lon_max lat_min lat_max

    countries2lonlat_DIC = {
        'Peru': [-81, -68.5, -18.3, 0],
        'Poland': [14.1, 24.2, 49.0, 54.9],
        'Ukraine': [22.1, 40.3, 44.3, 52.4],
        'China': [75.9, 134.3, 18.3, 52.3],
        'Japan': [124.1, 145.6, 24.3, 45.5],
        'Republic of Korea': [126.1, 129.4, 33.2, 38.4],
        'Thailand': [97.3, 105.7, 5.6, 20.5],
        'Taiwan (Province of China)': [118.3, 124.5, 20.5, 25.3],
        'Morocco': [-13.3, -1, 27.7, 35.9],
        'Algeria': [-8.1, 8.5, 22.7, 37.0],
        'Bangladesh': [88, 92.5, 20.5, 26.5],
        'India': [68, 97.25, 8, 37.1],
        'Nepal': [80.0, 88.3, 26.3, 30.5],
        'Czechia': [12.1, 18.8, 48.7, 51.1],
        'Estonia': [22.5, 28.2, 57.7, 59.5],
        'France': [-4.6, 9.5, 41.5, 51.1],
        'Italy': [7.0, 18.4, 36.7, 47.0],
        'Lithuania': [21.0, 26.5, 54.0, 56.4],
        'Romania': [20.4, 28.9, 43.6, 48.2],
        'Hungary': [16.2, 22.7,  45.8, 48.4]}

    disaster['min_lon'] = ""
    disaster['max_lon'] = ""
    disaster['min_lat'] = ""
    disaster['max_lat'] = ""

    for country, _ in countries2lonlat_DIC.items():

        position = countries2lonlat_DIC[country]
        disaster.min_lon[disaster.Country == country] = position[0]
        disaster.max_lon[disaster.Country == country] = position[1]
        disaster.min_lat[disaster.Country == country] = position[2]
        disaster.max_lat[disaster.Country == country] = position[3]

    disaster.to_csv(OUTPUT_CSV_PATH)


