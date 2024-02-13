import json
import pandas as pd
from pathlib import Path
"""
Look up table between country and longitude, latitude extend
Data source: https://latitudelongitude.org/
(Some extents are larger than the recorded cities' extent.)
"""
# CURR_FOLDER_PATH = Path(__file__).parent
DATA_FOLDER_PATH = Path(__file__).parent.parent.parent.parent / "E:/datasets/disasters"
INPUT_CSV_PATH = DATA_FOLDER_PATH / "output/tropicalCyclone2019.csv"
OUTPUT_JSON_PATH = DATA_FOLDER_PATH / "output/country_extent.json"

if __name__ == "__main__":

    disaster = pd.read_csv(INPUT_CSV_PATH, encoding='unicode_escape')
    print(f"{disaster.shape[0]} tropical cyclones")

    countries = disaster['Country'].unique()
    print(countries)
    #lon_min lon_max lat_min lat_max

    countries2lonlat_DIC = {
        'Marshall Islands': [162, 172, 4, 14],
        'Thailand': [97.3, 105.7, 5.6, 20.5],
        'Philippines': [116.9, 126.7, 4.6, 21.2],
        'Madagascar': [43.2, 50.5,-25.6, -12.0],
        'Mozambique': [30.2, 40.9, -26.9, -10.4],
        'Zimbabwe': [25.2, 33.1, -22.5, -16.5],
        'Micronesia (Federated States of)': [137.3, 163.1, 1.0, 9.6],
        'Bangladesh': [88, 92.5, 20.5, 26.5],
        'India': [68, 97.25, 8, 35],
        'Comoros': [43.2, 44.6, -12.4, -11.3],
        'United Republic of Tanzania': [29.3, 40.4, -11.8, -1],
        'United States of America': [-124.8, -66.9, 24.6, 49.3],
        'China': [75.9, 134.3, 18.3, 52.3],
        'Taiwan (Province of China)': [118.3, 124.5, 20.5, 25.3],
        'Bahamas': [-79, -77.0, 20.9, 27.1],
        'Mexico': [-117.2, -86.7, 14.5, 32.8],
        'Japan': [124.1, 145.6, 24.3, 45.5],
        'Republic of Korea': [126.1, 129.4, 33.2, 38.4],
        "Democratic People's Republic of Korea": [124.3, 130.5, 37.9, 43.0],
        'Viet Nam': [102.1, 109.5, 8.1, 23.4],
        'Somalia': [40.9, 51.5, -1.7, 12],
        'Fiji': [177.0, 179.4, -18.2, -12.5],
        'Peru': [-81, -68.5, -18.3, 0],
        'Poland': [14.1, 24.2, 49.0, 54.9],
        'Ukraine': [22.1, 40.3, 44.3, 52.4],
        'Morocco': [-16, -1, 23.6, 35.8],
        'Algeria': [-8.1, 8.5, 22.7, 37.0],
        'Nepal': [80.0, 88.3, 26.3, 30.5],
        'Czechia': [12.1, 18.8, 48.7, 51.1],
        'Estonia': [22.5, 28.2, 57.7, 59.5],
        'France': [-4.6, 9.5, 41.5, 51.1],
        'Italy': [7.0, 18.4, 36.7, 47.0],
        'Lithuania': [21.0, 26.5, 54.0, 56.4],
        'Romania': [20.4, 28.9, 43.6, 48.2],
        'Hungary': [16.2, 22.7, 45.8, 48.4],
        'Germany': [5.0, 16.0, 47.0, 55.0],
        'Pakistan': [61.6, 75.2, 24.3, 36.0],
        'Egypt': [25.5, 34.9, 24.0, 31.6],
        'Sudan': [22.4, 37.8, 10.5, 21.1],
        'Belgium': [2.5, 6.3, 49.5, 51.5],
        'South Africa': [17.8, 32.1, -34.6, -22.3],
        'Canada': [-135.1, -52.8, 42.1, 63.8],
        'Spain': [-17.9, 4.3, 27.7, 43.7],
        'United Kingdom of Great Britain and Northern Ireland': [-9.7, 1.8, 50.1, 60.2],
        'Netherlands (Kingdom of the)': [3.5, 7.2, 50.7, 53.4],
        'Portugal': [-28.7, -6.8, 32.6, 42.1],
        'Australia': [113.6, 153.7, -43.00, -12.4],
        'Austria': [9.6, 17.0, 46.5, 48.9]
        }
    # Convert and write JSON object to file
    with open(OUTPUT_JSON_PATH, "w") as outfile:
        json.dump(countries2lonlat_DIC, outfile)



