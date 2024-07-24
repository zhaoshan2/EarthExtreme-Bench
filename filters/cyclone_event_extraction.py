import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import data_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import netcdf


def extract_surface():

    OUTPUT_DATA_DIR = Path(__file__).parent.parent / "data" / f"{DISASTER}"
    if not os.path.exists(OUTPUT_DATA_DIR):
        os.mkdir(OUTPUT_DATA_DIR)

    SURFACE_VARIABLES = ["msl", "u10", "v10"]

    storm = pd.read_csv(os.path.join(CSV_PATH, f"{DISASTER}_2019_ibtracs_emdat.csv"))

    # Load constant masks
    soil_type = np.load(os.path.join(MASK_PATH, "soil_type.npy")).astype(
        np.float32
    )  # (721,1440)
    topography = np.load(os.path.join(MASK_PATH, "topography.npy")).astype(np.float32)
    land_mask = np.load(os.path.join(MASK_PATH, "land_mask.npy")).astype(np.float32)

    for i in range(storm.shape[0]):
        OUTPUT_DATA_DIR = Path(__file__).parent.parent / "data" / f"{DISASTER}"
        # for i in range(1):
        # ith disaster
        record = storm.iloc[i]
        start_year = f"{record['Start Year']:04d}"
        start_month = f"{record['Start Month']:02d}"
        start_day = f"{record['Start Day']:02d}"
        end_year = f"{record['End Year']:04d}"
        end_month = f"{record['End Month']:02d}"
        end_day = f"{record['End Day']:02d}"

        start_time_object = datetime.strptime(start_year + start_month, "%Y%m")
        start_time_day_object = datetime.strptime(
            start_year + start_month + start_day, "%Y%m%d"
        )
        start_time_str = start_time_object.strftime("%Y%m")
        end_time_object = datetime.strptime(end_year + end_month, "%Y%m")
        end_time_day_object = datetime.strptime(
            end_year + end_month + end_day, "%Y%m%d"
        )
        end_time_str = end_time_object.strftime("%Y%m")

        disasterDuration = data_utils.months_within_date_range(
            start_time_object, end_time_object
        )
        disasterDuration_inDays = data_utils.days_within_date_range(
            start_time_day_object, end_time_day_object
        )
        print(disasterDuration)
        # print(disasterDuration_inDays)

        # Spatial coverage of this disaster (AOI)
        lon0 = record["min_lon"]
        lon1 = record["max_lon"]
        lat0 = record["max_lat"]
        lat1 = record["min_lat"]

        # If the disaster spanned within 14 days
        dataset_list = []
        if len(disasterDuration_inDays) <= 14:
            mask_flag = True
            for disasterMonth in disasterDuration:
                dataset_month = xr.open_dataset(
                    os.path.join(SURFACE_DATA_PATH, f"surface_{disasterMonth}.nc")
                )
                surface_month = dataset_month[SURFACE_VARIABLES]

                ## Spatial Processing
                # If extends both western and eastern Earth
                if lon0 * lon1 < 0:
                    # Western part
                    part1 = surface_month.sel(
                        longitude=slice(360 + lon0, 359.75), latitude=slice(lat0, lat1)
                    )
                    # Eastern part
                    part2 = surface_month.sel(
                        longitude=slice(0, lon1), latitude=slice(lat0, lat1)
                    )
                    # Merge two parts of variables
                    surface = xr.concat([part1, part2], dim="longitude")

                    if mask_flag:
                        part1_aoi_longitude = part1["longitude"][:]
                        aoi_latitude = part1["latitude"][:]
                        part1_land_mask = data_utils.crop_mask(
                            land_mask, aoi_latitude, part1_aoi_longitude
                        )
                        part1_topography = data_utils.crop_mask(
                            topography, aoi_latitude, part1_aoi_longitude
                        )
                        part1_soil_type = data_utils.crop_mask(
                            soil_type, aoi_latitude, part1_aoi_longitude
                        )

                        part2_aoi_longitude = part2["longitude"][:]
                        part2_land_mask = data_utils.crop_mask(
                            land_mask, aoi_latitude, part2_aoi_longitude
                        )
                        part2_topography = data_utils.crop_mask(
                            topography, aoi_latitude, part2_aoi_longitude
                        )
                        part2_soil_type = data_utils.crop_mask(
                            soil_type, aoi_latitude, part2_aoi_longitude
                        )

                        # Merge two parts of mask
                        land_mask_cropped = np.concatenate(
                            (part1_land_mask, part2_land_mask), axis=1
                        )
                        topography_cropped = np.concatenate(
                            (part1_topography, part2_topography), axis=1
                        )
                        soil_type_cropped = np.concatenate(
                            (part1_soil_type, part2_soil_type), axis=1
                        )
                        mask_flag = False
                # If within Western part or within Eastern part
                else:
                    lon0, lon1 = data_utils.west2numbers(lon0, lon1)
                    region = {
                        "longitude": slice(lon0, lon1),
                        "latitude": slice(lat0, lat1),
                    }
                    # Select the affected area
                    surface = surface_month.sel(**region)
                    if mask_flag:
                        # Longitudinal and latitudinal extent of AOI
                        aoi_longitude = surface["longitude"][:]
                        aoi_latitude = surface["latitude"][:]
                        # Crop masks for AOI
                        land_mask_cropped = data_utils.crop_mask(
                            land_mask, aoi_latitude, aoi_longitude
                        )
                        topography_cropped = data_utils.crop_mask(
                            topography, aoi_latitude, aoi_longitude
                        )
                        soil_type_cropped = data_utils.crop_mask(
                            soil_type, aoi_latitude, aoi_longitude
                        )
                        mask_flag = False

                dataset_list.append(surface)
            surface_total = xr.concat(dataset_list, dim="time")
        else:
            print("The event lasted for too long, skip")
            continue

        # Temporal coverage of this disaster
        start_day_obj = datetime.strptime(
            f"{start_year}-{start_month}-{start_day}", "%Y-%m-%d"
        )
        end_day_obj = datetime.strptime(f"{end_year}-{end_month}-{end_day}", "%Y-%m-%d")
        # Select the affected temporal ranges
        surface_vars = surface_total.where(
            surface_total["time"].isin(
                pd.date_range(start_day_obj, end_day_obj, freq="H")
            ),
            drop=True,
        )  # (721,1440)

        # Identifier of the disaster event
        disno = record["DisNo."]
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
        surface_vars.to_netcdf(os.path.join(OUTPUT_DATA_DIR, f"{disno}_surface.nc"))
        for var in surface_vars.data_vars:
            print("shape", surface_vars[var].shape)
        ## Save relevent masks to npy file
        np.save(os.path.join(OUTPUT_DATA_DIR, f"land_{disno}.npy"), land_mask_cropped)
        np.save(
            os.path.join(OUTPUT_DATA_DIR, f"topography_{disno}.npy"), topography_cropped
        )
        np.save(
            os.path.join(OUTPUT_DATA_DIR, f"soil_type_{disno}.npy"), soil_type_cropped
        )


def extract_upper():
    OUTPUT_DATA_DIR = Path(__file__).parent.parent / "data" / f"{DISASTER}"
    if not os.path.exists(OUTPUT_DATA_DIR):
        os.mkdir(OUTPUT_DATA_DIR)

    UPPER_VARIABLES = ["z", "u", "v"]
    PRESSURE_LEVELS = [1000, 850, 700, 500, 200]
    storm = pd.read_csv(os.path.join(CSV_PATH, f"{DISASTER}_2019_ibtracs_emdat.csv"))
    # for i in range(1):
    for i in range(storm.shape[0]):
        OUTPUT_DATA_DIR = Path(__file__).parent.parent / "data" / f"{DISASTER}"
        # for i in range(1):
        # ith disaster
        record = storm.iloc[i]
        start_year = f"{record['Start Year']:04d}"
        start_month = f"{record['Start Month']:02d}"
        start_day = f"{record['Start Day']:02d}"
        end_year = f"{record['End Year']:04d}"
        end_month = f"{record['End Month']:02d}"
        end_day = f"{record['End Day']:02d}"

        start_time_object = datetime.strptime(
            f"{start_year}{start_month}{start_day}", "%Y%m%d"
        )
        # start_time_str = start_time_object.strftime('%Y%m')
        end_time_object = datetime.strptime(f"{end_year}{end_month}{end_day}", "%Y%m%d")
        # end_time_str = end_time_object.strftime('%Y%m')

        # disasterDuration = (end_time_object - start_time_object).days
        disasterDuration = data_utils.days_within_date_range(
            start_time_object, end_time_object
        )
        print(disasterDuration)

        # Spatial coverage of this disaster (AOI)
        lon0 = record["min_lon"]
        lon1 = record["max_lon"]
        lat0 = record["max_lat"]
        lat1 = record["min_lat"]

        # If the disaster spanned within 2 months
        dataset_list = []
        if len(disasterDuration) <= 14:
            for disasterDays in disasterDuration:
                dataset_month = xr.open_dataset(
                    os.path.join(UPPER_DATA_PATH, f"upper_{disasterDays}.nc")
                )
                upper_month = dataset_month[UPPER_VARIABLES]
                # print(upper_month.dims)
                # To do: select pressure levels
                # print("level",upper_month.level)
                upper_month = upper_month.sel(level=PRESSURE_LEVELS, drop=True)
                ## Spatial Processing
                # If extends both western and eastern Earth
                if lon0 * lon1 < 0:
                    # Western part
                    part1 = upper_month.sel(
                        longitude=slice(360 + lon0, 359.75), latitude=slice(lat0, lat1)
                    )
                    # Eastern part
                    part2 = upper_month.sel(
                        longitude=slice(0, lon1), latitude=slice(lat0, lat1)
                    )
                    # Merge two parts of variables
                    upper = xr.concat([part1, part2], dim="longitude")

                # If within Western part or within Eastern part
                else:
                    lon0, lon1 = data_utils.west2numbers(lon0, lon1)
                    region = {
                        "longitude": slice(lon0, lon1),
                        "latitude": slice(lat0, lat1),
                    }
                    # Select the affected area
                    upper = upper_month.sel(**region)

                dataset_list.append(upper)
            upper_total = xr.concat(dataset_list, dim="time")
        else:
            print("The event lasted for too long, skip")
            continue

        # Temporal coverage of this disaster
        start_day_obj = datetime.strptime(
            f"{start_year}-{start_month}-{start_day}", "%Y-%m-%d"
        )
        end_day_obj = datetime.strptime(f"{end_year}-{end_month}-{end_day}", "%Y-%m-%d")
        # Select the affected temporal ranges
        upper_vars = upper_total.where(
            upper_total["time"].isin(
                pd.date_range(start_day_obj, end_day_obj, freq="H")
            ),
            drop=True,
        )  # (721,1440)

        # Identifier of the disaster event
        disno = record["DisNo."]
        OUTPUT_DATA_DIR = OUTPUT_DATA_DIR / disno

        # Save disaster data to nc file
        upper_vars.to_netcdf(os.path.join(OUTPUT_DATA_DIR, f"{disno}_upper.nc"))
        for var in upper_vars.data_vars:
            print("shape", upper_vars[var].shape)

        # plt.figure()
        # plt.imshow(t2m[14])
        # plt.colorbar()
        # plt.title('t2m')
        # plt.savefig(fname=os.path.join(OUTPUT_DATA_DIR, 't2m.png'))


if __name__ == "__main__":
    CURR_FOLDER_PATH = Path(__file__).parent
    DISASTER_PATH = (
        CURR_FOLDER_PATH.parent.parent / "data_storage_home" / "data" / "disaster"
    )
    SURFACE_DATA_PATH = DISASTER_PATH.parent / "pangu" / "surface"
    MASK_PATH = DISASTER_PATH / "masks"
    CSV_PATH = DISASTER_PATH / "output_csv"
    DISASTER = "tropicalCyclone"
    UPPER_DATA_PATH = DISASTER_PATH.parent / "pangu" / "upper"
    # extract_surface()
    extract_upper()
    """
    installation error 
    python3.8.8

    pip install ecmwflibs
    pip install eccodes==1.3.1
    pip install cfgrib
    """
