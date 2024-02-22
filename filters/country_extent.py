import json
import pandas as pd
from pathlib import Path
"""
Look up table between country and longitude, latitude extend
Data source: https://latitudelongitude.org/
(Some extents are larger than the recorded cities' extent.)
"""
CURR_FOLDER_PATH = Path(__file__).parent
DATA_FOLDER_PATH = CURR_FOLDER_PATH.parent.parent / 'data_storage_home' / 'data' / 'disaster'
OUTPUT_JSON_PATH = DATA_FOLDER_PATH / "output_csv" / "country_extent.json"

if __name__ == "__main__":

    #lon_min lon_max lat_min lat_max

    countries2lonlat_DIC = {
        'MHL': [162, 172, 4, 14],
        'THA': [97.3, 105.7, 5.6, 20.5],
        'PHL': [116.9, 126.7, 4.6, 21.2],
        'MDG': [43.2, 50.5,-25.6, -12.0],
        'MOZ': [30.2, 40.9, -26.9, -10.4],
        'ZWE': [25.2, 33.1, -22.5, -16.5],
        'FSM': [137.3, 163.1, 1.0, 9.6],
        'BGD': [88, 92.5, 20.5, 26.5],
        'IND': [68, 97.25, 8, 35],
        'COM': [43.2, 44.6, -12.4, -11.3],
        'TZA': [29.3, 40.4, -11.8, -1],
        'USA': [-124.8, -66.9, 24.6, 49.3],
        'CHN': [75.9, 134.3, 18.3, 52.3],
        'TWN': [118.3, 124.5, 20.5, 25.3],
        'BHS': [-79, -77.0, 20.9, 27.1],
        'MEX': [-117.2, -86.7, 14.5, 32.8],
        'JPN': [124.1, 145.6, 24.3, 45.5],
        'KOR': [126.1, 129.4, 33.2, 38.4],
        "PRK": [124.3, 130.5, 37.9, 43.0],
        'VNM': [102.1, 109.5, 8.1, 23.4],
        'SOM': [40.9, 51.5, -1.7, 12],
        'FJI': [177.0, 179.4, -18.2, -12.5],
        'PER': [-81, -68.5, -18.3, 0],
        'POL': [14.1, 24.2, 49.0, 54.9],
        'UKR': [22.1, 40.3, 44.3, 52.4],
        'MAR': [-16, -1, 23.6, 35.8],
        'DZA': [-8.1, 8.5, 22.7, 37.0],
        'NPL': [80.0, 88.3, 26.3, 30.5],
        'CZE': [12.1, 18.8, 48.7, 51.1],
        'EST': [22.5, 28.2, 57.7, 59.5],
        'FRA': [-4.6, 9.5, 41.5, 51.1],
        'ITA': [7.0, 18.4, 36.7, 47.0],
        'LTU': [21.0, 26.5, 54.0, 56.4],
        'ROU': [20.4, 28.9, 43.6, 48.2],
        'HUN': [16.2, 22.7, 45.8, 48.4],
        'DEU': [5.0, 16.0, 47.0, 55.0],
        'PAK': [61.6, 75.2, 24.3, 36.0],
        'EGY': [25.5, 34.9, 24.0, 31.6],
        'SDN': [22.4, 37.8, 10.5, 21.1],
        'BEL': [2.5, 6.3, 49.5, 51.5],
        'ZAF': [17.8, 32.1, -34.6, -22.3],
        'CAN': [-135.1, -52.8, 42.1, 63.8],
        'ESP': [-17.9, 4.3, 27.7, 43.7],
        'GBR': [-9.7, 1.8, 50.1, 60.2],
        'NLD': [3.5, 7.2, 50.7, 53.4],
        'PRT': [-28.7, -6.8, 32.6, 42.1],
        'AUS': [113.6, 153.7, -43.00, -12.4],
        'AUT': [9.6, 17.0, 46.5, 48.9],
        'PSE':[34.2, 35.6, 31.2, 32.6],
        'MNG':[89.9, 114.6, 43.5, 50.3]
        }
    # Convert and write JSON object to file
    with open(OUTPUT_JSON_PATH, "w") as outfile:
        json.dump(countries2lonlat_DIC, outfile)



