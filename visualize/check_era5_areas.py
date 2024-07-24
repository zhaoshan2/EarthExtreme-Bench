import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER


def plot_aoi_tc(file_path, variable_name):
    ds = xr.open_dataset(file_path)
    # print(ds)
    ds = ds.sel(time=ds.time[0])
    lats = ds["latitude"]
    lons = ds["longitude"]
    ds = ds[variable_name]
    if lons[0] > lons[-1]:
        data_var0 = ds.sel(longitude=slice(lons[0], 359.75))
        data_var1 = ds.sel(longitude=slice(0, lons[-1]))
        # Plot the data
        data_var0.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            add_labels=False,
            alpha=0.6,
        )
        data_var1.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            add_labels=False,
            alpha=0.6,
        )
    else:
        ds.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            add_labels=False,
            alpha=0.6,
        )


def plot_aoi_heat(file_path, variable_name):
    ds = xr.open_dataset(file_path)

    # Print the dataset to understand its structure
    # print(ds)
    ds = ds.sel(time=ds.time[0])
    lats = ds["latitude"]
    lons = ds["longitude"]
    ds = ds[variable_name]
    if lons[0] > lons[-1]:
        data_var0 = ds.sel(longitude=slice(lons[0], 359.75))
        data_var1 = ds.sel(longitude=slice(0, lons[-1]))
        # Plot the data
        data_var0.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            add_labels=False,
            cmap=plt.cm.Reds,
        )
        data_var1.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            add_labels=False,
            cmap=plt.cm.Reds,
        )
    else:
        ds.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            add_labels=False,
            cmap=plt.cm.Reds,
        )


def plot_aoi_cold(file_path, variable_name):
    ds = xr.open_dataset(file_path)
    print(ds)
    ds = ds.sel(time=ds.time[0])
    lats = ds["latitude"]
    lons = ds["longitude"]
    ds = ds[variable_name]
    if lons[0] > lons[-1]:
        data_var0 = ds.sel(longitude=slice(lons[0], 359.75))
        data_var1 = ds.sel(longitude=slice(0, lons[-1]))
        # Plot the data
        data_var0.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            add_labels=False,
            cmap=plt.cm.Blues,
        )
        data_var1.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            add_labels=False,
            cmap=plt.cm.Blues,
        )
    else:
        ds.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            add_labels=False,
            cmap=plt.cm.Blues,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--disaster", default="coldwave")
    args = parser.parse_args()

    DISASTER = args.disaster
    # Create a figure and axis
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    CURR_FOLDER_PATH = Path(__file__).parent

    # # Layer - TC
    if DISASTER == "tropicalCyclone":
        OUTPUT_DATA_DIR = (
            CURR_FOLDER_PATH.parent / "data" / "weather" / f"{DISASTER}-hourly"
        )
        points_path = "/home/data_storage_home/data/disaster/input_csv/ibtracs.since1980.list.v04r00.csv"  # Replace with your actual file path
        # Load the IBTrACS data
        df = pd.read_csv(points_path)
        # Inspect the dataset
        print(df.head())
        # Extract relevant columns
        latitudes = df["LAT"]
        longitudes = df["LON"]
        # Plot ERA5 MSLP
        for root, subdirs, _ in os.walk(OUTPUT_DATA_DIR):
            for subdir in subdirs:
                for file in os.listdir(os.path.join(root, subdir)):
                    filename = os.fsdecode(file)
                    if filename.endswith("_surface.nc"):
                        filepath = os.path.join(
                            OUTPUT_DATA_DIR, filename[:-11], filename
                        )  # single vars
                        var = "msl"
                        plot_aoi_tc(filepath, var)
        # Plot Trajectories
        for root, subdirs, _ in os.walk(OUTPUT_DATA_DIR):
            for subdir in subdirs:
                for file in os.listdir(os.path.join(root, subdir)):
                    filename = os.fsdecode(file)
                    if filename.endswith("_surface.nc"):
                        filepath = os.path.join(
                            OUTPUT_DATA_DIR, filename[:-11], filename
                        )  # single vars
                        var = "msl"
                        # Plot each storm's trajectory
                        storm_name = filename[3:-11]
                        storm_data = df[df["SID"] == storm_name]
                        ax.plot(
                            storm_data["LON"],
                            storm_data["LAT"],
                            transform=ccrs.PlateCarree(),
                            linewidth=2,
                        )

    elif DISASTER == "heatwave":
        # Layer - Heatwave
        OUTPUT_DATA_DIR = (
            CURR_FOLDER_PATH.parent / "data" / "weather" / f"{DISASTER}-daily"
        )
        for root, subdirs, _ in os.walk(OUTPUT_DATA_DIR):
            for subdir in subdirs:
                for file in os.listdir(os.path.join(root, subdir)):
                    filename = os.fsdecode(file)
                    if filename.endswith(".nc"):
                        filepath = os.path.join(
                            OUTPUT_DATA_DIR, filename[:-3], filename
                        )  # single vars
                        var = "t2m"
                        plot_aoi_heat(filepath, var)

    elif DISASTER == "coldwave":
        # Layer - Coldwave
        OUTPUT_DATA_DIR = (
            CURR_FOLDER_PATH.parent / "data" / "weather" / f"{DISASTER}-daily"
        )
        for root, subdirs, _ in os.walk(OUTPUT_DATA_DIR):
            for subdir in subdirs:
                for file in os.listdir(os.path.join(root, subdir)):
                    filename = os.fsdecode(file)
                    if filename.endswith(".nc"):
                        filepath = os.path.join(
                            OUTPUT_DATA_DIR, filename[:-3], filename
                        )  # single vars
                        var = "t2m"
                        plot_aoi_cold(filepath, var)
    # Add coastlines and other features
    ax.add_feature(cfeature.BORDERS, edgecolor="lightgrey")
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.add_feature(cfeature.OCEAN, facecolor="lightgrey")
    # Add gridlines and labels
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    plt.savefig(f"figures/vis_{DISASTER}.png", dpi=300)
