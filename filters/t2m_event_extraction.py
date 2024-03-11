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
import data_utils
from dateutil.relativedelta import relativedelta
def main():
    CURR_FOLDER_PATH = Path(__file__).parent
    DISASTER_PATH = CURR_FOLDER_PATH.parent.parent / 'data_storage_home' / 'data' / 'disaster'
    CSV_PATH = DISASTER_PATH / 'output_csv'
    DATA_PATH = DISASTER_PATH.parent / 'pangu' / 'surface'
    MASK_PATH = DISASTER_PATH / 'masks'
    DISASTER = "heatwave" #heatwave, coldwave

    VARIABLE = 't2m'

    extremeTemperature = pd.read_csv(os.path.join(CSV_PATH, f'{DISASTER}_2019to2022_pos.csv'))

    # Load constant masks
    soil_type = np.load(os.path.join(MASK_PATH, 'soil_type.npy')).astype(np.float32) #(721,1440)
    topography = np.load(os.path.join(MASK_PATH, 'topography.npy')).astype(np.float32)
    land_mask = np.load(os.path.join(MASK_PATH, 'land_mask.npy')).astype(np.float32)

    for i in range(extremeTemperature.shape[0]):
    # for i in range(1):
        # ith disaster
        OUTPUT_DATA_DIR = CURR_FOLDER_PATH.parent / 'data' / f'{DISASTER}-daily'
        record = extremeTemperature.iloc[i] # iloc: ith row in the new array, loc: index i (old array)
        start_year = str(int(record['Start Year']))
        start_month = str(int(record['Start Month']))
        # Fill missing values in 'Day'
        start_day = 1 if np.isnan(record['Start Day']) else record['Start Day']
        end_day = 28 if np.isnan(record['End Day']) else record['End Day']
        start_day = str(int(start_day))
        end_year = str(int(record['End Year']))
        end_month = str(int(record['End Month']))
        end_day = str(int(end_day))

        start_time_object = datetime.strptime(start_year+start_month, "%Y%m")
        start_time_str = start_time_object.strftime('%Y%m')
        end_time_object = datetime.strptime(end_year+end_month, "%Y%m")
        end_time_str = end_time_object.strftime('%Y%m')

        ## Temporal Processing
        # disasterDuration = (end_time_object - start_time_object).days
        disasterDuration = data_utils.months_within_date_range(start_time_object, end_time_object)
        print(disasterDuration)

        # If the disaster spanned within 2 months
        dataset_list = []
        if len(disasterDuration) <= 1:
            for disasterMonth in disasterDuration:
                dataset_month = xr.open_dataset(os.path.join(DATA_PATH, f'surface_{disasterMonth}.nc'))
                t2m_month = dataset_month[VARIABLE]
                dataset_list.append(t2m_month)
            t2m = xr.concat(dataset_list, dim='time')
        # if disasterDuration == 0:
        #     dataset = xr.open_dataset(os.path.join(DATA_PATH, f'surface_{start_time_str}.nc'))
        #     t2m = dataset[VARIABLE]
        # # If the event spans two months
        # elif disasterDuration >0 and disasterDuration <=31:
        #     dataset_0 =
        #     t2m_0 = dataset_0[VARIABLE]
        #     dataset_1 = xr.open_dataset(os.path.join(DATA_PATH, f'surface_{end_time_str}.nc'))
        #     t2m_1 = dataset_1[VARIABLE]
        #     # Concatenate along the time axis
        #     t2m = xr.concat([t2m_0, t2m_1], dim='time')
        else:
            print("The event lasted for too long, skip")
            continue

        # Temporal coverage of this disaster
        start_day_obj = datetime.strptime(f'{start_year}-{start_month}-{start_day}', '%Y-%m-%d')
        end_day_obj = datetime.strptime(f'{end_year}-{end_month}-{end_day}', '%Y-%m-%d')
        # Select the affected temporal ranges
        t2m = t2m[t2m['time'].isin(pd.date_range(start_day_obj, end_day_obj, freq='H'))] #(721,1440)
        t2m = t2m.resample(time='1D').min()

        ## Spatial Processing
        # Spatial coverage of this disaster (AOI)
        lon0 = 0.5*(record['min_lon'] + record['max_lat']) - 16
        lon1 = lon0 + 32
        lat0 = 0.5*(record['min_lat'] + record['max_lat']) - 16
        lat1 = lat0 + 32
        # If extends both western and eastern Earth
        if lon0 * lon1 < 0:
            # Western part
            part1 = t2m.sel(longitude=slice(360+lon0, 359.75), latitude=slice(lat0, lat1))
            part1_aoi_longitude = part1["longitude"][:]
            aoi_latitude = part1["latitude"][:]

            part1_land_mask = data_utils.crop_mask(land_mask, aoi_latitude, part1_aoi_longitude)
            part1_topography = data_utils.crop_mask(topography, aoi_latitude, part1_aoi_longitude)
            part1_soil_type = data_utils.crop_mask(soil_type, aoi_latitude, part1_aoi_longitude)

            # Eastern part
            part2 = t2m.sel(longitude=slice(0, lon1), latitude=slice(lat0, lat1))
            part2_aoi_longitude = part2["longitude"][:]

            part2_land_mask = data_utils.crop_mask(land_mask, aoi_latitude, part2_aoi_longitude)
            part2_topography = data_utils.crop_mask(topography, aoi_latitude, part2_aoi_longitude)
            part2_soil_type = data_utils.crop_mask(soil_type, aoi_latitude, part2_aoi_longitude)

            # Merge two parts of t2m
            t2m = xr.concat([part1, part2], dim='longitude')
            # Merge two parts of mask
            land_mask_cropped = np.concatenate((part1_land_mask, part2_land_mask), axis=1)
            topography_cropped = np.concatenate((part1_topography, part2_topography), axis=1)
            soil_type_cropped = np.concatenate((part1_soil_type, part2_soil_type), axis=1)
        # If within Western part or within Eastern part
        else:
            lon0, lon1 = data_utils.west2numbers(lon0, lon1)
            region = {'longitude': slice(lon0, lon1), 'latitude': slice(lat0, lat1)}
            # Select the affected area
            t2m = t2m.sel(**region)

            # Longitudinal and latitudinal extent of AOI
            aoi_longitude = t2m["longitude"][:]
            aoi_latitude = t2m["latitude"][:]

            # Crop masks for AOI
            land_mask_cropped = data_utils.crop_mask(land_mask, aoi_latitude, aoi_longitude)
            topography_cropped = data_utils.crop_mask(topography, aoi_latitude, aoi_longitude)
            soil_type_cropped = data_utils.crop_mask(soil_type, aoi_latitude, aoi_longitude)

        # Identifier of the disaster event
        disno = record['DisNo.']
        OUTPUT_DATA_DIR = OUTPUT_DATA_DIR / disno
        if not os.path.exists(OUTPUT_DATA_DIR):
            os.mkdir(OUTPUT_DATA_DIR)
        # Plot cropped masks
        # plt.figure()
        # plt.imshow(land_mask_cropped)
        # plt.savefig(os.path.join(OUTPUT_DATA_DIR, f'land_{disno}.png'))
        # plt.figure()
        # plt.imshow(topography_cropped)
        # plt.savefig(os.path.join(OUTPUT_DATA_DIR, f'topography_{disno}.png'))
        # plt.figure()
        # plt.imshow(soil_type_cropped)
        # plt.savefig(os.path.join(OUTPUT_DATA_DIR, f'soil_type_{disno}.png'))

        ## Save disaster data to nc file
        t2m.to_netcdf(os.path.join(OUTPUT_DATA_DIR, f'{disno}.nc'))

        ## Save relevent masks to npy file
        np.save(os.path.join(OUTPUT_DATA_DIR, f'land_{disno}.npy'), land_mask_cropped)
        np.save(os.path.join(OUTPUT_DATA_DIR, f'topography_{disno}.npy'), topography_cropped)
        np.save(os.path.join(OUTPUT_DATA_DIR, f'soil_type_{disno}.npy'), soil_type_cropped)
        print("shape", t2m.shape)
        print(disno, np.amin(t2m) - 273, np.amax(t2m) - 273)
        # plt.figure()
        # plt.imshow(t2m[14])
        # plt.colorbar()
        # plt.title('t2m')
        # plt.savefig(fname=os.path.join(OUTPUT_DATA_DIR, 't2m.png'))
def extract_surface():

    CURR_FOLDER_PATH = Path(__file__).parent
    DISASTER_PATH = CURR_FOLDER_PATH.parent.parent / 'data_storage_home' / 'data' / 'disaster'
    CSV_PATH = DISASTER_PATH / 'output_csv'
    DATA_PATH = DISASTER_PATH.parent / 'pangu' / 'surface'
    MASK_PATH = DISASTER_PATH / 'masks'
    DISASTER = "heatwave" #heatwave, coldwave

    SURFACE_VARIABLES = 't2m'

    extremeTemperature = pd.read_csv(os.path.join(CSV_PATH, f'{DISASTER}_2019to2022_pos.csv'))
    # Load constant masks
    soil_type = np.load(os.path.join(MASK_PATH, 'soil_type.npy')).astype(np.float32)  # (721,1440)
    topography = np.load(os.path.join(MASK_PATH, 'topography.npy')).astype(np.float32)
    land_mask = np.load(os.path.join(MASK_PATH, 'land_mask.npy')).astype(np.float32)

    for i in range(extremeTemperature.shape[0]):
        OUTPUT_DATA_DIR = Path(__file__).parent.parent.parent / 'data_storage_home' / 'data' / 'disaster' / 'data' / 'weather' /f'{DISASTER}2022-daily'
        # for i in range(1):
        # ith disaster
        record = extremeTemperature.iloc[i]


        start_year = f"{int(record['Start Year']):04d}"
        start_month = f"{int(record['Start Month']):02d}"
        # Fill missing values in 'Day'
        start_day = 1 if np.isnan(record['Start Day']) else int(record['Start Day'])
        end_day = 28 if np.isnan(record['End Day']) else int(record['End Day'])
        start_day = f"{int(start_day):02d}"
        end_year = f"{int(record['End Year']):04d}"
        end_month = f"{int(record['End Month']):02d}"
        end_day = f"{int(end_day):02d}"

        start_time_object = datetime.strptime(start_year+start_month, "%Y%m")
        start_time_wday_object = datetime.strptime(start_year + start_month + start_day, "%Y%m%d")
        predict_start_time_obj = start_time_wday_object - relativedelta(months=1)
        predict_start_year_str = f"{int(predict_start_time_obj.year):04d}"
        predict_start_month_str = f"{int(predict_start_time_obj.month):02d}"
        predict_start_day_str = f"{int(predict_start_time_obj.day):02d}"
        start_time_str = start_time_object.strftime('%Y%m')
        end_time_object = datetime.strptime(end_year+end_month, "%Y%m")
        end_time_str = end_time_object.strftime('%Y%m')

        ## Temporal Processing
        # disasterDuration = (end_time_object - start_time_object).days
        disasterDuration = data_utils.months_within_date_range(predict_start_time_obj, end_time_object)
        print(disasterDuration)

        # print(disasterDuration_inDays)

        # Spatial coverage of this disaster (AOI)
        lon0 = record['min_lon']
        lon1 = record['max_lon']
        lat0 = record['max_lat']
        lat1 = record['min_lat']
        # lon0 = 0.5*(record['min_lon'] + record['max_lat']) - 16
        # lon1 = lon0 + 32
        # lat0 = 0.5*(record['min_lat'] + record['max_lat']) - 16
        # lat1 = lat0 + 32

        # If the disaster spanned within 6 months
        dataset_list = []
        if len(disasterDuration) <= 6:
            mask_flag = True
            for disasterMonth in disasterDuration:
                dataset_month = xr.open_dataset(os.path.join(DATA_PATH, f'surface_{disasterMonth}.nc'))
                surface_month = dataset_month[SURFACE_VARIABLES]

                ## Spatial Processing
                # If extends both western and eastern Earth
                if lon0 * lon1 < 0:
                    # Western part
                    part1 = surface_month.sel(longitude=slice(360 + lon0, 359.75), latitude=slice(lat0, lat1))
                    # Eastern part
                    part2 = surface_month.sel(longitude=slice(0, lon1), latitude=slice(lat0, lat1))
                    # Merge two parts of variables
                    surface = xr.concat([part1, part2], dim='longitude')

                    if mask_flag:
                        part1_aoi_longitude = part1["longitude"][:]
                        aoi_latitude = part1["latitude"][:]
                        part1_land_mask = data_utils.crop_mask(land_mask, aoi_latitude, part1_aoi_longitude)
                        part1_topography = data_utils.crop_mask(topography, aoi_latitude, part1_aoi_longitude)
                        part1_soil_type = data_utils.crop_mask(soil_type, aoi_latitude, part1_aoi_longitude)

                        part2_aoi_longitude = part2["longitude"][:]
                        part2_land_mask = data_utils.crop_mask(land_mask, aoi_latitude, part2_aoi_longitude)
                        part2_topography = data_utils.crop_mask(topography, aoi_latitude, part2_aoi_longitude)
                        part2_soil_type = data_utils.crop_mask(soil_type, aoi_latitude, part2_aoi_longitude)

                        # Merge two parts of mask
                        land_mask_cropped = np.concatenate((part1_land_mask, part2_land_mask), axis=1)
                        topography_cropped = np.concatenate((part1_topography, part2_topography), axis=1)
                        soil_type_cropped = np.concatenate((part1_soil_type, part2_soil_type), axis=1)
                        mask_flag = False
                # If within Western part or within Eastern part
                else:
                    lon0, lon1 = data_utils.west2numbers(lon0, lon1)
                    region = {'longitude': slice(lon0, lon1), 'latitude': slice(lat0, lat1)}
                    # Select the affected area
                    surface = surface_month.sel(**region)
                    if mask_flag:
                        # Longitudinal and latitudinal extent of AOI
                        aoi_longitude = surface["longitude"][:]
                        aoi_latitude = surface["latitude"][:]
                        # Crop masks for AOI
                        land_mask_cropped = data_utils.crop_mask(land_mask, aoi_latitude, aoi_longitude)
                        topography_cropped = data_utils.crop_mask(topography, aoi_latitude, aoi_longitude)
                        soil_type_cropped = data_utils.crop_mask(soil_type, aoi_latitude, aoi_longitude)
                        mask_flag = False

                dataset_list.append(surface)
            surface_total = xr.concat(dataset_list, dim='time')
        else:
            print("The event lasted for too long, skip")
            continue

        # Temporal coverage of this disaster
        start_day_obj = datetime.strptime(f'{predict_start_year_str}-{predict_start_month_str}-{predict_start_day_str}', '%Y-%m-%d')
        end_day_obj = datetime.strptime(f'{end_year}-{end_month}-{end_day}', '%Y-%m-%d')
        # Select the affected temporal ranges
        surface_vars = surface_total.where(
            surface_total['time'].isin(pd.date_range(start_day_obj, end_day_obj, freq='H')), drop=True)  # (721,1440)
        # Aggregated to daily extreme value (max for heatwave and min for coldwave)
        if DISASTER == "heatwave":
            surface_vars = surface_vars.resample(time='1D').max()
        elif DISASTER == "coldwave":
            surface_vars = surface_vars.resample(time='1D').min()
        else:
            print("The Value is not is not aggregated!")
        # Identifier of the disaster event
        disno = record['DisNo.']
        OUTPUT_DATA_DIR = OUTPUT_DATA_DIR / disno
        if not os.path.exists(OUTPUT_DATA_DIR):
            os.mkdir(OUTPUT_DATA_DIR)

        # # Plot cropped masks
        # plt.figure()
        # plt.imshow(land_mask_cropped)
        # plt.savefig(os.path.join(OUTPUT_DATA_DIR, f'land_{disno}.png'))
        # plt.figure()
        # plt.imshow(topography_cropped)
        # plt.savefig(os.path.join(OUTPUT_DATA_DIR, f'topography_{disno}.png'))
        # plt.figure()
        # plt.imshow(soil_type_cropped)
        # plt.savefig(os.path.join(OUTPUT_DATA_DIR, f'soil_type_{disno}.png'))

        # Save disaster data to nc file
        surface_vars.to_netcdf(os.path.join(OUTPUT_DATA_DIR, f'{disno}.nc'))
        print("shape", surface_vars.shape)
        ## Save relevent masks to npy file
        np.save(os.path.join(OUTPUT_DATA_DIR, f'land_{disno}.npy'), land_mask_cropped)
        np.save(os.path.join(OUTPUT_DATA_DIR, f'topography_{disno}.npy'), topography_cropped)
        np.save(os.path.join(OUTPUT_DATA_DIR, f'soil_type_{disno}.npy'), soil_type_cropped)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', default="cfgrib")
    # if grib file, engine=cfgrib
    args = parser.parse_args()
    extract_surface()

    """
    installation error 
    python3.8.8
    
    pip install ecmwflibs
    pip install eccodes==1.3.1
    pip install cfgrib
    """