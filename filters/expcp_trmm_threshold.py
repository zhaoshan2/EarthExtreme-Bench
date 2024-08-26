"""
Find the 95 percentile threshold at each grid for four seasons using the 22 year historical data (1998-2019)
"""

import gc
import os
from pathlib import Path

import dask
import xarray as xr
from tqdm import tqdm

CURR_FOLDER_PATH = Path(__file__).parent
DATA_FOLDER_PATH = CURR_FOLDER_PATH.parent.parent / "data_storage_home"
DISASTER = "precipitation"


def open_dataset(filepath):
    try:
        return xr.open_dataset(filepath)  # Use dask to handle large files
    except Exception as e:
        print(f"Error opening {filepath}: {e}")
        return None


def plot_thresholds(data, i):
    latitudes = np.linspace(-50, 50, data.shape[0])
    longitudes = np.linspace(-180, 180, data.shape[1])

    m = Basemap(
        projection="merc",
        llcrnrlat=-50,
        urcrnrlat=50,
        llcrnrlon=-180,
        urcrnrlon=180,
        resolution="c",
    )

    # Create a meshgrid for latitude and longitude
    lon, lat = np.meshgrid(longitudes, latitudes)
    x, y = m(lon, lat)

    # Plot the data
    plt.figure(figsize=(10, 6))
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary()

    # Plot the data using pcolormesh
    c_scheme = m.pcolormesh(x, y, data, vmin=0, vmax=55, shading="auto", cmap="RdBu")

    # Add x and y axis labels and ticks
    # Create integer ticks for longitude
    long_ticks = np.arange(-180, 181, 30)
    # Create integer ticks for latitude
    lat_ticks = np.arange(-50, 51, 20)

    # Set the x ticks with integer longitude values
    plt.xticks([m(lon, 0)[0] for lon in long_ticks], long_ticks)
    # Set the y ticks with integer latitude values
    plt.yticks([m(0, lat)[1] for lat in lat_ticks], lat_ticks)

    # Add a colorbar
    plt.colorbar(c_scheme, orientation="vertical", fraction=0.016, pad=0.05)
    plt.title(f"Season {i}")
    plt.savefig(f"Season {i}.png", dpi=300, bbox_inches="tight", pad_inches=0.01)


if __name__ == "__main__":
    """
    To do:
    1. Coarsen the file at the scale of 20 (0.25 to 5 degree)
    2. Combine the corasen file into a single file
    3. Check the threshold for each pixel at 95 percentile and save.
    """
    """
    Stage 1
    """
    # # Corsen each file
    # file_path = DATA_FOLDER_PATH / "trmm/subset_TRMM_3B42_Daily_7_20240722_114424_.txt"
    # with open(file_path, 'r') as file:
    #     file_paths = [line[-26:].strip() for line in file.readlines()[1:]]
    # ds = safe_open_dataset(fp)
    # with dask.config.set(**{'array.slicing.split_large_chunks': False}):
    #     ds_coarse = ds.coarsen(lat=20, lon=20).mean()
    #     for var in ['precipitation_cnt','IRprecipitation_cnt','HQprecipitation_cnt','randomError_cnt']:
    #         ds_coarse[var] = ds_coarse[var].astype('int32')
    #         ds_coarse[var].attrs['_FillValue'] = -9999
    #
    # chunk_dict = {'lat': -1, 'lon': -1}
    #
    # for var in ds_coarse:
    #     if ds_coarse[var].encoding.get('chunks'):
    #         del ds_coarse[var].encoding['chunks']
    #     if ds_coarse[var].encoding.get('preferred_chunks'):
    #         del ds_coarse[var].encoding['preferred_chunks']
    # for var in ds_coarse.coords:
    #     if ds_coarse[var].encoding.get('chunks'):
    #         del ds_coarse[var].encoding['chunks']
    #     if ds_coarse[var].encoding.get('preferred_chunks'):
    #         del ds_coarse[var].encoding['preferred_chunks']
    # ds_coarse.chunk(chunk_dict).to_netcdf(ds_coarse_path)
    """
    Stage 2     ## Combine yearly data into a single file
    """
    # years = ["1998"]
    # for year in years:
    #     combined_file_path = f'trmm_3B4207_{year}_n.nc'
    #
    #     root_folder = f"data_coarsen/{year}"
    #     file_paths = []
    #     for root, dirs, files in os.walk(root_folder):
    #         for file in files:
    #             file_path = os.path.join(root, file)
    #             file_paths.append(file_path)
    #             # print("file_path", file_path)
    #     file_paths = sorted(file_paths)
    #     datasets = []
    #     for fp in tqdm(file_paths):
    #         ds_coarse = open_dataset(fp)
    #         if ds_coarse is not None:
    #             datasets.append(ds_coarse)
    #     combined_ds = xr.concat(datasets, dim='time')
    #     try:
    #         combined_ds.load()
    #         combined_ds.to_netcdf(combined_file_path)
    #         print(f"File saved successfully to: {combined_file_path}")
    #         del combined_ds
    #         gc.collect()
    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    # print("All files saved successfully")

    """
    Stage 3
    """
    import numpy as np
    import scipy.stats as st

    """

    trmm = xr.open_dataset(DATA_FOLDER_PATH / "trmm/data_coarsen/trmm_5deg_combined.nc")
    lat = trmm.variables["lat"][:]  # 纬度 20
    lon = trmm.variables["lon"][:]  # 经度 72
    time = trmm["time"][:].values

    pcp = trmm.variables["precipitation"]
    pcp = pcp.transpose("time", "lat", "lon")  # switch the dim (lon, lat) to (lat, lon)
    print("dims of precipitation variable:", pcp.dims)
    t = time.shape[0]
    la = lat.shape[0]
    lo = lon.shape[0]
    n = la * lo

    y = 22
    # Options: extract seasonal indices
    # index_seasons = np.zeros((4, 22), dtype="object")
    # index_seasons[0, 0] = np.arange(0, 59, 1)
    # index_seasons[1, 0] = np.arange(59, 151, 1)
    # index_seasons[2, 0] = np.arange(151, 243, 1)
    # index_seasons[3, 0] = np.arange(243, 334, 1)
    # for i in range(0, y - 1):
    #     index_seasons[0, i + 1] = np.arange(334 + 365 * i, 424 + 365 * i, 1)
    #     index_seasons[1, i + 1] = np.arange(424 + 365 * i, 516 + 365 * i, 1)
    #     index_seasons[2, i + 1] = np.arange(516 + 365 * i, 608 + 365 * i, 1)
    #     index_seasons[3, i + 1] = np.arange(608 + 365 * i, 699 + 365 * i, 1)
    #
    # index_season = np.zeros((4), dtype="object")
    # index_season[0] = np.arange(0, 59, 1)  # winter
    # index_season[1] = np.arange(59, 151, 1)  # spring
    # index_season[2] = np.arange(151, 243, 1)  # summer
    # index_season[3] = np.arange(243, 334, 1)  # fall
    #
    # for i in range(0, y - 1):
    #     index_season[0] = np.concatenate(
    #         (index_season[0], np.arange(334 + 365 * i, 424 + 365 * i, 1))
    #     )
    #     index_season[1] = np.concatenate(
    #         (index_season[1], np.arange(424 + 365 * i, 516 + 365 * i, 1))
    #     )
    #     index_season[2] = np.concatenate(
    #         (index_season[2], np.arange(516 + 365 * i, 608 + 365 * i, 1))
    #     )
    #     index_season[3] = np.concatenate(
    #         (index_season[3], np.arange(608 + 365 * i, 699 + 365 * i, 1))
    #     )

    # for i in range(0, y - 1):
    #     index_seasons[0, i + 1] = np.arange(334 + 365 * i, 424 + 365 * i, 1)
    #     index_seasons[1, i + 1] = np.arange(424 + 365 * i, 516 + 365 * i, 1)
    #     index_seasons[2, i + 1] = np.arange(516 + 365 * i, 608 + 365 * i, 1)
    #     index_seasons[3, i + 1] = np.arange(608 + 365 * i, 699 + 365 * i, 1)

    # index_season = np.zeros((4), dtype="object")
    # index_season[0] = np.arange(0, 59, 1)  # winter
    # index_season[1] = np.arange(59, 151, 1)  # spring
    # index_season[2] = np.arange(151, 243, 1)  # summer
    # index_season[3] = np.arange(243, 334, 1)  # fall

    # for i in range(0, y - 1):
    #     index_season[0] = np.concatenate(
    #         (index_season[0], np.arange(334 + 365 * i, 424 + 365 * i, 1))
    #     )
    #     index_season[1] = np.concatenate(
    #         (index_season[1], np.arange(424 + 365 * i, 516 + 365 * i, 1))
    #     )
    #     index_season[2] = np.concatenate(
    #         (index_season[2], np.arange(516 + 365 * i, 608 + 365 * i, 1))
    #     )
    #     index_season[3] = np.concatenate(
    #         (index_season[3], np.arange(608 + 365 * i, 699 + 365 * i, 1))
    #     )
    """
    """
    Stage 3.2: Extract monthly indices
    """
    """
    from datetime import datetime, timedelta

    # # Start date
    start_date = datetime(1998, 1, 1)

    # # Function to get the corresponding date
    def get_date(day):
        return start_date + timedelta(days=int(day))

    # Create a list to store the indices corresponding to each month
    months_list = [[] for _ in range(12)]

    for day in time:
        date = get_date(day)  # 1998-01-02
        month_index = (
            date.month - 1
        )  # month_index will be 0 for January, 1 for February, etc.
        months_list[month_index].append(day)

    def ec_wd(ts, perc):
        th = st.scoreatpercentile(ts[ts > 1], perc)
        if th > 0.1:
            return th
        else:
            return 0.1

    index_season = months_list
    # # for perc in xrange(80, 100):
    perc = 80
    for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        mnoe = len(index_season[j]) * (1 - perc / 100.0)
        print(mnoe)
        th = np.zeros((la, lo))
        for l in range(la):
            rain = np.zeros((t, lo))
            precip = pcp[index_season[j], l, :]
            if np.ma.is_masked(precip) is True:
                rain[index_season[j], :] = pcp[index_season[j], l, :].filled(0)
            else:
                rain[index_season[j], :] = pcp[index_season[j], l, :]
            for k in range(lo):
                th[l, k] = ec_wd(rain[:, k], perc)
        # np.save(
        #     "trmm7_global_wd_score_cor_seasonal_rain_perc%d_season%d" % (perc, j), th
        # )
        np.save(
            "thresholds_months/trmm7_global_wd_score_cor_seasonal_rain_perc%d_month%d"
            % (perc, j + 1),
            th,
        )
    """
    """
    Stage 4: visualize the extracted thresholds
    """

    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt

    for i in range(4):
        data = np.load(
            f"thresholds_seasons/trmm7_global_wd_score_cor_seasonal_rain_perc95_season{i}.npy"
        )
        plot_thresholds(data, i)
