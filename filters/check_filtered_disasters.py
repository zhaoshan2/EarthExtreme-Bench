import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
import xarray as xr
import pandas as pd
import os
from pathlib import Path
import json
import argparse
from datetime import datetime, timedelta
import datetime

# Each image is normalzied for better visualization
def temperature2Png():
    DISASTER = 'coldwave'
    CURR_FOLDER_PATH = Path(__file__).parent
    OUTPUT_DATA_DIR = CURR_FOLDER_PATH.parent / 'data' / 'weather' / f'{DISASTER}-daily'
    for root, subdirs, _ in os.walk(OUTPUT_DATA_DIR):
        for subdir in subdirs:
            for file in os.listdir(os.path.join(root, subdir)):
                filename = os.fsdecode(file)
                if filename.endswith("2022-0800-MNG.nc"):
                    dataset = xr.open_dataset(os.path.join(OUTPUT_DATA_DIR, filename[:-3], filename)) # single vars
                    var = "t2m"
                    data = dataset[var].values.astype(np.float32)
                    times = dataset.time
                    # min_v = np.percentile(data, 1)
                    # max_v = np.percentile(data, 99)
                    min_v = 225
                    max_v = 281
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
    DISASTER = 'coldwave'
    CURR_FOLDER_PATH = Path(__file__).parent
    OUTPUT_DATA_DIR = CURR_FOLDER_PATH.parent / 'data' / 'weather' / f'{DISASTER}-daily'
    # OUTPUT_DATA_DIR = Path(
    #     __file__).parent.parent.parent / 'data_storage_home' / 'data' / 'disaster' / 'data' / 'weather' / f'{DISASTER}-daily'
    disaster = pd.DataFrame()

    for root, subdirs, _ in os.walk(OUTPUT_DATA_DIR):
        for subdir in subdirs:
            for file in os.listdir(os.path.join(root, subdir)):
                filename = os.fsdecode(file)
                if filename.startswith("2023") and filename.endswith(".nc"):
                # if filename.endswith(".nc"):
                    dataset = xr.open_dataset(os.path.join(OUTPUT_DATA_DIR, filename[:-3], filename)) # single vars
                    for var in dataset.data_vars:
                        data = dataset[var].values.astype(np.float32)
                        times = dataset.time
                        print("times", times)

                        aoi_longitude = dataset["longitude"][:]
                        aoi_latitude = dataset["latitude"][:]

                        disaster = pd.concat([disaster, pd.DataFrame([{
                            'Disno.':filename[:-3],
                            'disaster_type':DISASTER,
                            'start':pd.to_datetime(times[0].values).strftime('%Y-%m-%d %H:%M'),
                            'end':pd.to_datetime(times[-1].values).strftime('%Y-%m-%d %H:%M'),
                            'num_frames':data.shape[0],
                            'H':data.shape[1],
                            'W':data.shape[2],
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
    disaster.to_csv(os.path.join(OUTPUT_DATA_DIR, f'{DISASTER}-daily_records_test.csv'), index=False)



def extremeTemperature_statistics():
    DISASTER = 'coldwave'
    CURR_FOLDER_PATH = Path(__file__).parent
    OUTPUT_DATA_DIR = CURR_FOLDER_PATH.parent / 'data' / 'weather' / f'{DISASTER}-daily'
    mean_std_dic = {}
    full_data = []
    for root, subdirs, _ in os.walk(OUTPUT_DATA_DIR):
        for subdir in subdirs:
            for file in os.listdir(os.path.join(root, subdir)):
                filename = os.fsdecode(file)
                if filename.endswith(".nc"):
                    # data in and after 2023 will be not used to compute the statistics
                    if not filename.startswith("2023"):
                        dataset = xr.open_dataset(os.path.join(OUTPUT_DATA_DIR, filename[:-3], filename)) # single vars
                        for var in dataset.data_vars:
                            data = dataset[var].values.astype(np.float32)
                            full_data.append(data.flatten())
    full_data = np.concatenate(full_data)
    # print("shape", full_data.shape)
    mean_std_dic[f"{DISASTER}_mean"] = np.mean(full_data.flatten())
    mean_std_dic[f"{DISASTER}_std"] = np.std(full_data.flatten())

    # masks
    MASK_DIR = CURR_FOLDER_PATH.parent / 'data' / 'masks'
    for root, _, files in os.walk(MASK_DIR):
        for file in files:
            mask = np.load(os.path.join(root, file))
            mean_std_dic[f"{file[:-4]}_mean"] = np.mean(mask.flatten())
            mean_std_dic[f"{file[:-4]}_std"] = np.std(mask.flatten())

    for key, value in mean_std_dic.items():
        if isinstance(value, np.float32):
            mean_std_dic[key] = float(value)
    with open(os.path.join(OUTPUT_DATA_DIR, f'{DISASTER}-daily_records_stats.json'), 'w') as f:
        f.write(json.dumps(mean_std_dic))

def cyclone_attributes():
    DISASTER = 'tropicalCyclone'
    CURR_FOLDER_PATH = Path(__file__).parent
    OUTPUT_DATA_DIR = CURR_FOLDER_PATH.parent / 'data' / 'weather' / f'{DISASTER}-hourly'
    disaster = pd.DataFrame()
    disaster_surface = pd.DataFrame()

    for root, subdirs, _ in os.walk(OUTPUT_DATA_DIR):
        for subdir in subdirs:
            if subdir.startswith("TC_20193"):
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
                            'H':data.shape[2],
                            'W':data.shape[3],
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
                            # To do: an image is H x W, here is wrong with the naming
                            'H':data_surface.shape[1],
                            'W':data_surface.shape[2],
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

    disaster.to_csv(os.path.join(OUTPUT_DATA_DIR, f'{DISASTER}_upper_records_test.csv'), index=False)
    disaster_surface.to_csv(os.path.join(OUTPUT_DATA_DIR, f'{DISASTER}_surface_records_test.csv'), index=False)
def cyclone_statistics():
    DISASTER = 'tropicalCyclone'
    CURR_FOLDER_PATH = Path(__file__).parent
    OUTPUT_DATA_DIR = CURR_FOLDER_PATH.parent / 'data' / 'weather' / f'{DISASTER}-hourly'
    mean_std_dic = {}
    full_data_msl, full_data_u10, full_data_v10 = [], [], []
    for root, subdirs, _ in os.walk(OUTPUT_DATA_DIR):
        for subdir in subdirs:
            # Compute the statistics of the training set
            if not subdir.startswith("TC_20193"):
                for file in os.listdir(os.path.join(root, subdir)):
                    filename = os.fsdecode(file)
                    if filename.endswith("_surface.nc"):
                        dataset = xr.open_dataset(os.path.join(OUTPUT_DATA_DIR, filename[:-11], filename)) # single vars

                        msl = dataset['msl'].values.astype(np.float32) #(T,H,W)
                        u10 = dataset['u10'].values.astype(np.float32)  # (T,H,W)
                        v10 = dataset['v10'].values.astype(np.float32)  # (T,H,W)
                        full_data_msl.append(msl.flatten())
                        full_data_u10.append(u10.flatten())
                        full_data_v10.append(v10.flatten())
    full_data_msl = np.concatenate(full_data_msl)
    full_data_u10 = np.concatenate(full_data_u10)
    full_data_v10 = np.concatenate(full_data_v10)

    mean_std_dic[f"{DISASTER}_msl_mean"] = np.mean(full_data_msl.flatten())
    mean_std_dic[f"{DISASTER}_msl_std"] = np.std(full_data_msl.flatten())
    mean_std_dic[f"{DISASTER}_u10_mean"] = np.mean(full_data_u10.flatten())
    mean_std_dic[f"{DISASTER}_u10_std"] = np.std(full_data_u10.flatten())
    mean_std_dic[f"{DISASTER}_v10_mean"] = np.mean(full_data_v10.flatten())
    mean_std_dic[f"{DISASTER}_v10_std"] = np.std(full_data_v10.flatten())
    # masks
    MASK_DIR = CURR_FOLDER_PATH.parent / 'data' / 'masks'
    for root, _, files in os.walk(MASK_DIR):
        for file in files:
            mask = np.load(os.path.join(root, file))
            mean_std_dic[f"{file[:-4]}_mean"] = np.mean(mask.flatten())
            mean_std_dic[f"{file[:-4]}_std"] = np.std(mask.flatten())

    for key, value in mean_std_dic.items():
        if isinstance(value, np.float32):
            mean_std_dic[key] = float(value)
    with open(os.path.join(OUTPUT_DATA_DIR, f'{DISASTER}-hourly_surface_records_stats.json'), 'w') as f:
        f.write(json.dumps(mean_std_dic))

def cyclone_upper_statistics():
    DISASTER = 'tropicalCyclone'
    CURR_FOLDER_PATH = Path(__file__).parent
    OUTPUT_DATA_DIR = CURR_FOLDER_PATH.parent / 'data' / 'weather' / f'{DISASTER}-hourly'
    mean_std_dic = {}
    full_data_z, full_data_u, full_data_v = [], [], []
    for root, subdirs, _ in os.walk(OUTPUT_DATA_DIR):
        for subdir in subdirs:
            # Compute the statistics of the training set
            if not subdir.startswith("TC_20193"):
                for file in os.listdir(os.path.join(root, subdir)):
                    filename = os.fsdecode(file)
                    if filename.endswith("_upper.nc"):
                        dataset = xr.open_dataset(os.path.join(OUTPUT_DATA_DIR, filename[:-9], filename)) # single vars

                        z = dataset['z'].values.astype(np.float32)#(T,Z,H,W)
                        # The datacube shall be transformed to (Z, T, H, W) for reshaping to (Z, T*H*W) (keep the spatial information intact)
                        z = z.transpose(1,0,2,3)
                        Z, T, H, W = z.shape
                        z = z.reshape(Z, T*H*W)
                        u = dataset['u'].values.astype(np.float32)  # (T,H,W)
                        u = u.transpose(1,0,2,3)
                        u = u.reshape(Z, T*H*W)
                        v = dataset['v'].values.astype(np.float32) # (T,H,W)
                        v = v.transpose(1,0,2,3)
                        v= v.reshape(Z, T*H*W)
                        full_data_z.append(z)
                        full_data_u.append(u)
                        full_data_v.append(v)
    full_data_z = np.concatenate(full_data_z, axis=-1)
    full_data_u = np.concatenate(full_data_u, axis=-1)
    full_data_v = np.concatenate(full_data_v, axis=-1)

    l = 0
    for pressure_level in dataset.level.values:
        print(pressure_level)
        mean_std_dic[f"{DISASTER}_z_{pressure_level}_mean"] = np.mean(full_data_z, axis=-1)[l]
        mean_std_dic[f"{DISASTER}_z_{pressure_level}_std"] = np.std(full_data_z, axis=-1)[l]
        mean_std_dic[f"{DISASTER}_u_{pressure_level}_mean"] = np.mean(full_data_u, axis=-1)[l]
        mean_std_dic[f"{DISASTER}_u_{pressure_level}_std"] = np.std(full_data_u, axis=-1)[l]
        mean_std_dic[f"{DISASTER}_v_{pressure_level}_mean"] = np.mean(full_data_v, axis=-1)[l]
        mean_std_dic[f"{DISASTER}_v_{pressure_level}_std"] = np.std(full_data_v, axis=-1)[l]
        l += 1
    # masks
    MASK_DIR = CURR_FOLDER_PATH.parent / 'data' / 'masks'
    for root, _, files in os.walk(MASK_DIR):
        for file in files:
            mask = np.load(os.path.join(root, file))
            mean_std_dic[f"{file[:-4]}_mean"] = np.mean(mask.flatten())
            mean_std_dic[f"{file[:-4]}_std"] = np.std(mask.flatten())

    for key, value in mean_std_dic.items():
        if isinstance(value, np.float32):
            mean_std_dic[key] = float(value)
    with open(os.path.join(OUTPUT_DATA_DIR, f'{DISASTER}-hourly_upper_records_stats.json'), 'w') as f:
        f.write(json.dumps(mean_std_dic))


if __name__ == "__main__":

    regionalDailyExtremeTemperature()

    """
    installation error 
    python3.8.8
    
    pip install ecmwflibs
    pip install eccodes==1.3.1
    pip install cfgrib
    """
    # z means from pressure levels 1000, ... , 200 during tropical cyclones:
    # 1038, 14768, 30742, 57232, 121268
    # z,q,t,u,v means from pressure levels for 39 years:
    # p200 [[[1.15558266e+05  1.93777287e-05  2.18535049e+02  1.41896858e+01, -4.50169668e-02]]]
    # p500: [[[5.41329375e+04  8.51599325e-04  2.53472382e+02  6.55437517e+00, -2.38431394e-02]]]
    # p700: [[[2.88882793e+04  2.42784317e-03  2.67694305e+02  3.34081912e+00, 2.15189029e-02]]]
    # p850: [[[1.37797188e+04,  4.56286687e-03,  2.74664062e+02,  1.41666412e+00, 1.42662525e-01]]]
    # p1000: [[[7.37141235e+02  7.01335957e-03  2.81473816e+02 - 3.30022648e-02, 1.86560124e-01]]]]
