import matplotlib.pyplot as plt

if __name__ == "__main__":
    import xarray as xr
    import numpy as np
    import h5py
    from datetime import timedelta
    import pandas as pd

    month = 2
    tau = np.load(
        f"thresholds_months/trmm7_global_wd_score_cor_seasonal_rain_perc95_month{month}.npy"
    )
    # pcp_path = "hdf_crops_n/20200206_0750_1600.hdf5"
    pcp_path = "/home/data_storage_home/trmm/3B42_Daily.20190207.7.nc4"
    scale_factor = 20
    trmm = xr.open_dataset(pcp_path)
    pcp = trmm.variables["precipitation"].values
    print(pcp.shape)
    # with h5py.File(pcp_path, "r") as file:
    #     # Get the dataset
    #     precipitation = file["Grid"]["precipitation"][
    #         :
    #     ]  # shape [1, 3600, 1800] dim(time, lon, lat)
    #     precipitation = np.where(precipitation < 0, 0, precipitation).squeeze(
    #         0
    #     )  # fill the missing value as 0 and disgard the first dim
    #
    #     precipitation = np.transpose(
    #         precipitation
    #     )  # conver the dim (lon, lat) to (lat, lon)
    #     lon = file["Grid"]["lon"][:]  # 3600: -180, 180
    #     lat = file["Grid"]["lat"][:]  # 1800: -90, 90
    #     time = file["Grid"]["time"][:]
    #
    # assert precipitation.shape[0] == len(lat)
    # assert precipitation.shape[1] == len(lon)
    #
    # lat_indices = np.where((lat >= -50) & (lat <= 50))[0]
    # pcp = precipitation[lat_indices, :]
    plt.figure()
    plt.imshow(pcp.T, vmax=150)
    plt.colorbar()
    title_time = pd.to_datetime("2019-02-07")
    plt.title(f"trmm_{title_time}")
    plt.savefig("pp.png")

    # plt.figure()
    # plt.imshow(pcp, vmax=20)
    # plt.colorbar()
    # title_time = pd.to_datetime("2020-02-07") + timedelta(minutes=30 * 21)
    # plt.title(f"cropped_imerg_{title_time}")
    # plt.savefig("cropped.png")

    # or
    reshaped_pcp = pcp.reshape(
        pcp.shape[0] // scale_factor,
        scale_factor,
        pcp.shape[1] // scale_factor,
        scale_factor,
    )

    # Compute the mean over the blocks
    coarsened_pcp = reshaped_pcp.mean(axis=(1, 3))  # coarsened_pcp (lat, lon) (20,72)
    # print(tau[7, 32])
    # print(coarsened_pcp[7, 32])
    plt.figure()
    plt.imshow(coarsened_pcp.T)
    title_time = pd.to_datetime("2019-02-07")  # + timedelta(minutes=30 * 21)
    plt.title(f"coarsen_trmm_{title_time}")
    # plt.title("Feb. 95 percentile value")

    plt.colorbar()
    plt.savefig(f"tau-2.png")

    # s = np.mean(precipitation, axis=(1, 2))
    # print("mean of croped patch", s[48:96])
    # for i in range(precipitation.shape[0]):
    #     plt.figure()
    #     plt.imshow(precipitation[i], vmax=20)
    #     title_time = pd.to_datetime("2023-06-04") + timedelta(minutes=30 * i)
    #     plt.title(f"precipitation_{title_time}")
    #     plt.savefig(f"imgs/fra20230604_1150_1500_cropped_{i:03}.png")
    #     # print("tau", tau[(1150 - 400) // 50, 1550 // 50])
    #     plt.close()
