import matplotlib.pyplot as plt

if __name__ == "__main__":
    import xarray as xr
    import numpy as np
    import h5py
    from datetime import timedelta
    import pandas as pd

    # month = 7
    # tau = np.load(
    #    f"thresholds_months/trmm7_global_wd_score_cor_seasonal_rain_perc95_month{month}.npy"
    # )
    pcp_path = "hdf_crops_n/20230604_1150_1500.hdf5"
    with h5py.File(pcp_path, "r") as file:
        precipitation = file["precipitation"][:]

    # s = np.mean(precipitation, axis=(1, 2))
    # print("mean of croped patch", s[48:96])
    for i in range(precipitation.shape[0]):
        plt.figure()
        plt.imshow(precipitation[i], vmax=20)
        title_time = pd.to_datetime("2023-06-04") + timedelta(minutes=30 * i)
        plt.title(f"precipitation_{title_time}")
        plt.savefig(f"imgs/fra20230604_1150_1500_cropped_{i:03}.png")
        # print("tau", tau[(1150 - 400) // 50, 1550 // 50])
        plt.close()
