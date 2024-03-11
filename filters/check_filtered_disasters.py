import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
import xarray as xr
import pandas as pd
import os
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import datetime

# Each image is normalzied for better visualization
def main():
    DISASTER = 'heatwave'
    CURR_FOLDER_PATH = Path(__file__).parent
    OUTPUT_DATA_DIR = CURR_FOLDER_PATH.parent / 'data' / 'weather' / f'{DISASTER}-daily'
    for root, subdirs, _ in os.walk(OUTPUT_DATA_DIR):
        for subdir in subdirs:
            for file in os.listdir(os.path.join(root, subdir)):
                filename = os.fsdecode(file)
                if filename.endswith("2019-0650-GBR.nc"):
                    dataset = xr.open_dataset(os.path.join(OUTPUT_DATA_DIR, filename[:-3], filename)) # single vars
                    var = "t2m"
                    data = dataset[var].values.astype(np.float32)
                    times = dataset.time
                    min_v = np.percentile(data, 1)
                    max_v = np.percentile(data, 99)
                    # print(filename, "{:.2f}".format(np.max(t2m) - 273))
                    for i in range(data.shape[0]):
                        plt.figure()
                        plt.imshow(data[i], vmin=min_v, vmax=max_v)
                        plt.colorbar()
                        title_time = pd.to_datetime(times[i].values).strftime('%Y-%m-%d')
                        plt.title(f'{var}_{title_time}')
                        EVENT_PNG_FOLDER = os.path.join(OUTPUT_DATA_DIR, filename[:-3], 'PNG') # single var
                        if not os.path.exists(EVENT_PNG_FOLDER):
                            os.mkdir(EVENT_PNG_FOLDER)
                        plt.savefig(os.path.join(EVENT_PNG_FOLDER, f"{filename[:-3]}_{var}_{str(i)}.png"))
def cyclone2Png():
    DISASTER = 'tropicalCyclone'
    CURR_FOLDER_PATH = Path(__file__).parent
    OUTPUT_DATA_DIR = CURR_FOLDER_PATH.parent / 'data' / f'{DISASTER}'
    for root, subdirs, _ in os.walk(OUTPUT_DATA_DIR):
        for subdir in subdirs:
            for file in os.listdir(os.path.join(root, subdir)):
                filename = os.fsdecode(file)
                # example visualize z variable at last levels
                if filename.endswith("TC_2019233N15255_upper.nc"):
                    dataset = xr.open_dataset(os.path.join(OUTPUT_DATA_DIR, filename[:-9], filename)) # multi vars
                    for var in dataset.data_vars:
                        data = dataset[var].values.astype(np.float32)
                        times = dataset.time
                        min_v = np.percentile(data[:,-1,...], 1)
                        max_v = np.percentile(data[:,-1,...], 99)
                        # print(filename, "{:.2f}".format(np.max(t2m) - 273))
                        for i in range(25):
                            plt.figure()
                            plt.imshow(data[i,-1,:,:], vmin=min_v, vmax=max_v)
                            plt.colorbar()
                            title_time = pd.to_datetime(times[i].values).strftime('%Y-%m-%d %H:%M')
                            plt.title(f'{var}_{title_time}')
                            EVENT_PNG_FOLDER = os.path.join(OUTPUT_DATA_DIR, filename[:-9], 'PNG') # multi-vars
                            if not os.path.exists(EVENT_PNG_FOLDER):
                                os.mkdir(EVENT_PNG_FOLDER)
                            plt.savefig(os.path.join(EVENT_PNG_FOLDER, f"{filename[:-9]}_{var}_{str(i)}.png"))

def extremeTemperature_attributes():
    DISASTER = 'heatwave'
    CURR_FOLDER_PATH = Path(__file__).parent
    # OUTPUT_DATA_DIR = CURR_FOLDER_PATH.parent / 'data' / 'weather' / f'{DISASTER}-daily'
    OUTPUT_DATA_DIR = Path(
        __file__).parent.parent.parent / 'data_storage_home' / 'data' / 'disaster' / 'data' / 'weather' / f'{DISASTER}2022-daily'

    disaster = pd.DataFrame()

    for root, subdirs, _ in os.walk(OUTPUT_DATA_DIR):
        for subdir in subdirs:
            for file in os.listdir(os.path.join(root, subdir)):
                filename = os.fsdecode(file)
                if filename.endswith(".nc"):
                    dataset = xr.open_dataset(os.path.join(OUTPUT_DATA_DIR, filename[:-3], filename)) # single vars
                    for var in dataset.data_vars:
                        data = dataset[var].values.astype(np.float32)
                        times = dataset.time

                        aoi_longitude = dataset["longitude"][:]
                        aoi_latitude = dataset["latitude"][:]

                        disaster = pd.concat([disaster, pd.DataFrame([{
                            'Disno.':filename[:-3],
                            'disaster_type':DISASTER,
                            'start':pd.to_datetime(times[0].values).strftime('%Y-%m-%d %H:%M'),
                            'end':pd.to_datetime(times[-1].values).strftime('%Y-%m-%d %H:%M'),
                            'num_frames':data.shape[0],
                            'W':data.shape[1],
                            'H':data.shape[2],
                            'spatial_resolution': 0.25,
                            'spatial_resolution_unit': 'degree',
                            'temporal_resolution': 1 ,
                            'temporal_resolution_unit': 'day',
                            'min_val':np.min(data),
                            'max_val':np.max(data),
                            'min1_val':np.percentile(data, 1),
                            'max99_val':np.percentile(data, 99),
                            'mean_val':np.mean(data),
                            'val_unit':'K',
                            'min_lon': aoi_longitude.values.astype(np.float32)[0],
                            'max_lon': aoi_longitude.values.astype(np.float32)[-1],
                            'min_lat': aoi_latitude.values.astype(np.float32)[-1],
                            'max_lat': aoi_latitude.values.astype(np.float32)[0],
                            'variables': var
                        }])], ignore_index=True)

    # Convert the 'start' column to datetime
    disaster['start'] = pd.to_datetime(disaster['start'])

    # Sort the DataFrame by the 'start' column
    disaster = disaster.sort_values(by='start')
    disaster.to_csv(os.path.join(OUTPUT_DATA_DIR, f'{DISASTER}2022-daily_records.csv'), index=False)

def cyclone_attributes():
    DISASTER = 'tropicalCyclone'
    CURR_FOLDER_PATH = Path(__file__).parent
    OUTPUT_DATA_DIR = CURR_FOLDER_PATH.parent / 'data' / 'weather' / f'{DISASTER}'
    disaster = pd.DataFrame()
    disaster_surface = pd.DataFrame()

    for root, subdirs, _ in os.walk(OUTPUT_DATA_DIR):
        for subdir in subdirs:
            for file in os.listdir(os.path.join(root, subdir)):
                filename = os.fsdecode(file)
                # meta information about upper variables
                if filename.endswith("_upper.nc"):
                    dataset = xr.open_dataset(os.path.join(OUTPUT_DATA_DIR, filename[:-9], filename)) # single vars
                    var = list(dataset.data_vars)
                    var = var[0]

                    data = dataset[var].values.astype(np.float32)
                    times = dataset.time
                    disaster = pd.concat([disaster, pd.DataFrame([{
                        'Disno.':filename[:-9],
                        'disaster_type':DISASTER,
                        'start':pd.to_datetime(times[0].values).strftime('%Y-%m-%d %H:%M'),
                        'end':pd.to_datetime(times[-1].values).strftime('%Y-%m-%d %H:%M'),
                        'num_frames':data.shape[0],
                        'W':data.shape[2],
                        'H':data.shape[3],
                        'Z':data.shape[1],
                        'spatial_resolution': 0.25,
                        'spatial_resolution_unit': 'degree',
                        'temporal_resolution': 1 ,
                        'temporal_resolution_unit': 'hour',
                        'variables': list(dataset.data_vars)
                    }])], ignore_index=True)

                # meta information about surface variables
                elif filename.endswith("_surface.nc"):
                    dataset_surface = xr.open_dataset(os.path.join(OUTPUT_DATA_DIR, filename[:-11], filename)) # single vars
                    var = list(dataset_surface.data_vars)
                    var = var[0]
                    data_surface = dataset_surface[var].values.astype(np.float32)
                    times = dataset_surface.time
                    disaster_surface = pd.concat([disaster_surface, pd.DataFrame([{
                        'Disno.':filename[:-11],
                        'disaster_type':DISASTER,
                        'start':pd.to_datetime(times[0].values).strftime('%Y-%m-%d %H:%M'),
                        'end':pd.to_datetime(times[-1].values).strftime('%Y-%m-%d %H:%M'),
                        'num_frames':data_surface.shape[0],
                        'W':data_surface.shape[1],
                        'H':data_surface.shape[2],
                        'spatial_resolution': 0.25,
                        'spatial_resolution_unit': 'degree',
                        'temporal_resolution': 1 ,
                        'temporal_resolution_unit': 'hour',
                        'variables': list(dataset_surface.data_vars)
                    }])], ignore_index=True)
    # Convert the 'start' column to datetime
    disaster['start'] = pd.to_datetime(disaster['start'])

    # Sort the DataFrame by the 'start' column
    disaster = disaster.sort_values(by='start')
    disaster_surface['start'] = pd.to_datetime(disaster_surface['start'])

    # Sort the DataFrame by the 'start' column
    disaster_surface = disaster_surface.sort_values(by='start')

    disaster.to_csv(os.path.join(OUTPUT_DATA_DIR, f'{DISASTER}_upper_records.csv'), index=False)
    disaster_surface.to_csv(os.path.join(OUTPUT_DATA_DIR, f'{DISASTER}_surface_records.csv'), index=False)
if __name__ == "__main__":

    extremeTemperature_attributes()

    """
    installation error 
    python3.8.8
    
    pip install ecmwflibs
    pip install eccodes==1.3.1
    pip install cfgrib
    """