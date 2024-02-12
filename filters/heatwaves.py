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
DATA_FOLDER_PATH = Path("E:/datasets/disasters")
INPUT_CSV_PATH = DATA_FOLDER_PATH / "output/heatwave2015_2019.csv"
OUTPUT_CSV_PATH = DATA_FOLDER_PATH / "output/heatwave2015_2019_pos.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="E:/datasets/disasters")

    args = parser.parse_args()

    heatwaves = pd.read_csv(INPUT_CSV_PATH, encoding='unicode_escape')

    print(f"{heatwaves.shape[0]} heatwaves")

    countries = heatwaves['Country'].unique()
    #lon_min lon_max lat_max lat_min

    countries2lonlat_DIC = {
        'India': [68, 97.25, 8, 37.1],
        'Germany': [5.0, 16.0, 47.0, 55.0],
        'Pakistan': [11.0, 69.3, 21.8, 30.4],
        'Egypt': [25.5, 34.9, 24.0, 31.6],
        'Sudan': [22.4, 37.8, 10.5, 21.1],
        'Belgium': [2.5, 6.3, 49.5, 51.5],
        'France': [-4.6, 9.5, 41.5, 51.1],
        'Japan': [124.1, 145.6, 24.3, 45.5],
        'South Africa': [17.8, 32.1, -34.6, -22.3],
        'Canada': [-135.1, -52.8, 42.1, 63.8],
        'Republic of Korea': [126.1, 129.4, 33.2, 38.4],
        'Spain': [-17.9, 4.3, 27.7, 43.7],
        'United Kingdom of Great Britain and Northern Ireland': [-9.7, 1.8, 50.1, 60.2],
        'Italy': [7.0, 18.4, 36.7, 47.0],
        'Netherlands (Kingdom of the)': [3.5, 7.2, 50.7, 53.4],
        'Portugal': [-28.7, -6.8, 32.6, 42.1],
        "Democratic People's Republic of Korea": [124.3, 130.5, 37.9, 43.0],
        'Australia': [113.6, 153.7, -43.00, -12.4],
        'Austria': [9.6, 17.0, 46.5, 48.9]
    # below use - for w and + for e, needs to be converted to 0 to 360 (but not here)
        # 'Russian Federation': [19.90, 177.55, 71.70, 41.25],
        # 'United States of America' : [66.95, 124.77, 49.38, 24.52],
        # 'New Zealand': [113.6594, 153.61194, -12.46113, -43.00311], #Austrilia
        # 'Nigeria': [3, 15, 14, 4],
        # 'China': [75.98, 134.29, 52.33, 18.25],
        # 'Bangladesh': [88, 92.5, 26.5, 20.5],
        # 'Belgium':[2.59368 to 6.25749, 49.56652 to 51.46791],
        # 'Switzerland':[6.07544 to 9.83723, 45.83203 to 47.69732],
        # 'Czechia':[12.19499 to 18.76458, 48.73881 to 51.00369 ],
        # 'France':[-4.65 to 9.45,  41.59101 to 51.03457],
        # 'United Kingdom of Great Britain and Northern Ireland':[-9.70264 to 1.75159, 50.10319 to 60.15456],
        # 'Croatia':[13.52389 to 19.37694,42.64807 to 46.38444 ],
        # 'Italy':[7.05809 to 18.3781,36.71703 to 46.99623],
        # 'Luxembourg':[5.88056 to 6.44194, 49.48056 to 49.86778],
        # 'Slovakia':[17.02188 to 22.18136, 47.76356 to 49.43503],
        # 'Slovenia':[13.52639 to 16.4509,45.50526 to 46.83694],
        # 'Algeria':[ -8.14743 to 8.46667,22.785 to 36.92917],
        # 'Albania':[19.44139 to 20.99, 39.87556 to 42.07694],
        # 'North Macedonia':[20.52421 to 22.89056, 41.03143 to 42.20194],
        # 'Romania':[20.48333 to 28.86667, 43.66667 to 48.18333],
        # #'Canary Islands':[],
        # 'Bulgaria':[22.68361 to 28.33333, 41.38333 to 44.11667],
        # 'Bosnia and Herzegovina':[15.77806 to 19.29256, 42.71197 to 45.18497],
        # 'Cyprus':[32.42451 to 34.37916,34.68406 to 35.59719],
        # 'Greece':[19.91975 to 28.2225, 35.01186 to 41.50306],
        # 'Hungary':[ 16.27358 to 22.68096,  45.85499 to 48.39492],
        # 'Serbia':[18.98472 to 22.58611 ,42.55139 to 46.102792],
        # 'Turkey':[ 25.90902 to 44.5742,35.9025 to 42.02683  ],
        # 'Brazil':[-72.89583 to -34.80861, -33.69111 to 2.8197],
        # 'Bolivia (Plurinational State of)':[ -68.85063 to -57.76667, -22.08659 to -10.83676],
        # 'Denmark':[8.24402 to 14.70664,54.76906 to 57.72093 ],
        # 'Estonia':[ 22.50389 to 28.19028, 57.77781 to 59.47667],
        # 'Finland':[ 21.37596 to 30.93276,59.83333 to 68.90596],
        # 'Ireland':[-9.70264 to -6.04944,51.58666 to 55.13333],
        # 'Lithuania':[ 21.06861 to 26.41667,54.01667 to 56.31667],
        # 'Latvia':[21.01667 to 28.12165, 55.88333 to 57.89752 ],
        # 'Malta':[14.20361 to 14.56701,35.82583 to 36.07222],
        # 'Montenegro' :[18.5375 to 20.16652,41.92936 to 43.3567],
        # 'Norway':[5.0328 to 29.74943,58.0274 to 70.66336],
        # 'Poland':[ 14.24712 to 23.89251,49.29899 to 54.79086],
        # 'Sweden':[11.1712 to 23.15645, 55.37514 to 67.85572],
        # 'Kyrgyzstan':[69.5276 to 78.39362, 39.83895 to 42.89106
    }
    print(countries)
    heatwaves['min_lon'] = ""
    heatwaves['max_lon'] = ""
    heatwaves['min_lat'] = ""
    heatwaves['max_lat'] = ""

    for country in countries:

        position = countries2lonlat_DIC[country]
        heatwaves.min_lon[heatwaves.Country == country] = position[0]
        heatwaves.max_lon[heatwaves.Country == country] = position[1]
        heatwaves.min_lat[heatwaves.Country == country] = position[2]
        heatwaves.max_lat[heatwaves.Country == country] = position[3]


    heatwaves.to_csv(OUTPUT_CSV_PATH)


