import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from pathlib import Path
import os
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
def plot_aoi(path, color, m):
    records = pd.read_csv(path, encoding='unicode_escape')
    lon_mins, lon_maxs = records['min_lon'], records['max_lon']
    lat_mins, lat_maxs = records['min_lat'], records['max_lat']
    for lon_min, lon_max, lat_min, lat_max in zip(lon_mins, lon_maxs, lat_mins, lat_maxs):
        # Plot the box
        x = [lon_min, lon_max, lon_max, lon_min, lon_min]
        y = [lat_min, lat_min, lat_max, lat_max, lat_min]
        m.plot(x, y, marker=None, color=color, linewidth=2)

if __name__ == '__main__':

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a Basemap instance for a global map
    m = Basemap(projection='cyl', resolution='c')

    # Draw coastlines and countries
    m.drawcoastlines()
    m.drawcountries()

    # Add a shaded relief image
    m.shadedrelief()
    # m.etopo()
    # Draw parallels (latitude lines) and meridians (longitude lines)
    parallels = range(-90, 91, 30)  # from -90 to 90 degrees, every 30 degrees
    meridians = range(0, 361, 60)  # from -180 to 180 degrees, every 60 degrees

    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10)

    CURR_FOLDER_PATH = Path(__file__).parent.parent.parent / 'data_storage_home/data/disaster/output_csv'

    DISASTER = 'coldwave'
    record_path_coldwave_train = CURR_FOLDER_PATH / DISASTER / f'{DISASTER}_2019to2022_pos.csv'
    plot_aoi(record_path_coldwave_train, color='b', m=m)

    record_path_coldwave_test = CURR_FOLDER_PATH / DISASTER / f'{DISASTER}_2023to2023_pos.csv'
    plot_aoi(record_path_coldwave_test, color='b', m=m)

    DISASTER = 'heatwave'
    record_path_heatwave_train = CURR_FOLDER_PATH / DISASTER / f'{DISASTER}_2019to2022_pos.csv'
    plot_aoi(record_path_heatwave_train, color='r', m=m)
    # Test set US
    lon_min, lon_max, lat_min, lat_max = -124.0, -68.25, 9.25, 65.0
    x = [lon_min, lon_max, lon_max, lon_min, lon_min]
    y = [lat_min, lat_min, lat_max, lat_max, lat_min]
    m.plot(x, y, marker=None, color='r', linewidth=2)
    # Extreme precipitation, Italy
    latitu, lontitu = 46.4883, 11.2106
    # lon_min, lon_max, lat_min, lat_max = 6.4, 16.1, 45.4, 47.6
    # x = [lon_min, lon_max, lon_max, lon_min, lon_min]
    # y = [lat_min, lat_min, lat_max, lat_max, lat_min]
    x, y = m(lontitu, latitu)
    m.plot(x, y, marker='o', color='b')
    # Tropical cyclones, tropics
    lon_min, lon_max, lat_min, lat_max = -179, 179, -40, 60
    x = [lon_min, lon_max, lon_max, lon_min, lon_min]
    y = [lat_min, lat_min, lat_max, lat_max, lat_min]
    m.plot(x, y, linestyle='--', color='orange', linewidth=2)
    # # Add title and show the plot
    # plt.title('Colored Basemap with Box')
    # contiguous United States
    lon_min, lon_max, lat_min, lat_max = -124.5, -66, 24, 49.5
    x = [lon_min, lon_max, lon_max, lon_min, lon_min]
    y = [lat_min, lat_min, lat_max, lat_max, lat_min]
    m.plot(x, y, marker=None, color='green', linewidth=3)

    # Flood 17 sites
    locations = [
        {"name": "Houston, US", "latitude": 29.7604, "longitude": -95.3698},
        {"name": "Lumberton, US", "latitude": 34.6182, "longitude": -79.0086},
        {"name": "Sainte-Marthe-sur-le-Lac, Canada", "latitude": 45.5302, "longitude": -73.9269},
        {"name": "Beledweyne, Somalia", "latitude": 4.7368, "longitude": 45.2033},
        {"name": "Beira, Mozambique", "latitude": -19.8173, "longitude": 34.8156},
        {"name": "Jubba, Somalia", "latitude": 4.8517, "longitude": 31.5825},  # Assuming Juba, South Sudan
        {"name": "Lokoja, Nigeria", "latitude": 7.8023, "longitude": 6.7333},  # Corrected country to Nigeria
        {"name": "Iwaki, Japan", "latitude": 37.0507, "longitude": 140.8928},
        {"name": "Koriyama, Japan", "latitude": 37.4000, "longitude": 140.3833},
        {"name": "Weihui, China", "latitude": 35.4030, "longitude": 114.0658},
        {"name": "Aqqala, Iran", "latitude": 37.0133, "longitude": 54.4556},
        {"name": "Zhuozhou, China", "latitude": 39.4855, "longitude": 115.9730},
        {"name": "Langfang, China", "latitude": 39.5097, "longitude": 116.6945},
        {"name": "Coraki, Australia", "latitude": -28.8085, "longitude": 153.2030},
        {"name": "Sydney, Australia", "latitude": -33.8688, "longitude": 151.2093},
        {"name": "Nova Kakhovka, Ukraine", "latitude": 46.7545, "longitude": 33.3792}
    ]

    for location in locations:
        print(f"Location: {location['name']}, Latitude: {location['latitude']}, Longitude: {location['longitude']}")
        x, y = m(location['longitude'], location['latitude'])
        m.plot(x, y, marker="^", color='lime')

    plt.savefig("test.png", dpi=300)
