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

def latlon2xy(lat:xr.DataArray, lon:xr.DataArray) -> tuple[np.array, np.array]:
     """
     Latitude, longitude to pixel index
     """
     x = (90 - lat.values.astype(np.float32)) * 4
     y = (lon.values.astype(np.float32)) * 4
     return x, y
def crop_mask(mask: np.array, lat: xr.DataArray , lon:xr.DataArray) -> np.array:
    x, y = latlon2xy(lat, lon)
    mask_cropped = mask[int(np.min(x)):int(np.max(x)), int(np.min(y)):int(np.max(y))]
    return mask_cropped.astype(np.float32)

def months_within_date_range(start_date, end_date):
    """
    Returns a list of months within the given date range.
    """
    months_list = []
    current_date = start_date

    while current_date <= end_date:
        months_list.append(current_date.strftime('%Y%m'))
        # Move to the next month
        month = current_date.month
        year = current_date.year
        if month == 12:
            month = 1
            year += 1
        else:
            month += 1
        current_date = datetime(year, month, 1)
    return list(months_list)
def main():
    DISASTER_PATH = Path('E:/datasets/disasters')
    CSV_PATH = DISASTER_PATH / 'output'
    DATA_PATH = DISASTER_PATH / 'surface'
    MASK_PATH = DATA_PATH / 'masks'
    OUTPUT_DATA_DIR = Path(__file__).parent.parent / 'res/heatwaves'
    VARIABLE = 't2m'

    extremeTemperature = pd.read_csv(os.path.join(CSV_PATH, 'heatwave2015_2019_pos.csv'))
    # Sample dataset, only study disaster in 2019
    extremeTemperature = extremeTemperature[extremeTemperature['Start Year'] >= 2019]

    # Load constant masks
    soil_type = np.load(os.path.join(MASK_PATH, 'soil_type.npy')).astype(np.float32) #(721,1440)
    topography = np.load(os.path.join(MASK_PATH, 'topography.npy')).astype(np.float32)
    land_mask = np.load(os.path.join(MASK_PATH, 'land_mask.npy')).astype(np.float32)

    # for i in range(extremeTemperature.shape[0]):
    for i in range(1):
        # ith disaster
        record = extremeTemperature.iloc[i]
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

        # disasterDuration = (end_time_object - start_time_object).days
        disasterDuration = months_within_date_range(start_time_object, end_time_object)
        print(disasterDuration)

        # If the disaster spanned within 2 months
        dataset_list = []
        if len(disasterDuration) <= 2:
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

        # Spatial coverage of this disaster (AOI)
        # To do: if longitude * latitude < 0
        # part1 = ds.sel(longitude=slice(355, 360))
        # part2 = ds.sel(longitude=slice(0, 9))
        # merged_data = xr.concat([part1, part2], dim='longitude')
        lon0 = record['min_lon']
        lon1 = record['max_lon']
        lat0 = record['max_lat']
        lat1 = record['min_lat']
        region = {'longitude': slice(lon0, lon1), 'latitude': slice(lat0, lat1)}
        # Select the affected area
        t2m = t2m.sel(**region)

        # Longitudinal and latitudinal extent of AOI
        aoi_longitude = t2m["longitude"][:]
        aoi_latitude = t2m["latitude"][:]
        # Crop masks for AOI
        land_mask_cropped = crop_mask(land_mask, aoi_latitude, aoi_longitude)
        topography_cropped = crop_mask(topography, aoi_latitude, aoi_longitude)
        soil_type_cropped = crop_mask(soil_type, aoi_latitude, aoi_longitude)

        # Identifier of the disaster event
        disno = record['DisNo.']

        # Plot cropped masks
        plt.figure()
        plt.imshow(land_mask_cropped)
        plt.savefig(os.path.join(OUTPUT_DATA_DIR, f'land_{disno}.png'))
        plt.figure()
        plt.imshow(topography_cropped)
        plt.savefig(os.path.join(OUTPUT_DATA_DIR, f'topography_{disno}.png'))
        plt.figure()
        plt.imshow(soil_type_cropped)
        plt.savefig(os.path.join(OUTPUT_DATA_DIR, f'soil_type_{disno}.png'))

        ## Save disaster data to nc file
        # t2m.to_netcdf(os.path.join(OUTPUT_DATA_DIR, f'{disno}.nc'))

        ## Save relevent masks to npy file
        # np.save(os.path.join(OUTPUT_DATA_DIR, f'land_{disno}.npy'), land_mask_cropped)
        # np.save(os.path.join(OUTPUT_DATA_DIR, f'topography_{disno}.npy'), topography_cropped)
        # np.save(os.path.join(OUTPUT_DATA_DIR, f'soil_type_{disno}.npy'), soil_type_cropped)
        print("shape", t2m.shape)
        print(disno, np.amin(t2m) - 273)
        # plt.figure()
        # plt.imshow(t2m[14])
        # plt.colorbar()
        # plt.title('t2m')
        # plt.savefig(fname=os.path.join(OUTPUT_DATA_DIR, 't2m.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', default="cfgrib")
    # if grib file, engine=cfgrib
    args = parser.parse_args()
    main()

    """
    installation error 
    python3.8.8
    
    pip install ecmwflibs
    pip install eccodes==1.3.1
    pip install cfgrib
    """