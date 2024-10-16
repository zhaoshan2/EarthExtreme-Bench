import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

if __name__ == "__main__":
    land_mask_pangu = np.load("/home/EarthExtreme-Bench/data/masks/land_mask.npy")
    soil_type_pangu = np.load("/home/EarthExtreme-Bench/data/masks/soil_type.npy")
    topography_pangu = np.load("/home/EarthExtreme-Bench/data/masks/topography.npy")

    plt.figure(0)
    plt.imshow(land_mask_pangu)
    plt.colorbar()
    plt.savefig(f"figures/land_mask_pangu.png", dpi=100)

    plt.figure(1)
    plt.imshow(soil_type_pangu)
    plt.colorbar()
    plt.savefig(f"figures/soil_type_pangu.png", dpi=100)

    plt.figure(2)
    plt.imshow(topography_pangu)
    plt.colorbar()
    plt.savefig(f"figures/topography_pangu.png", dpi=100)

    aurora = xr.open_dataset(
        "/home/EarthExtreme-Bench/data/masks/af66b1a0629735dceb6f7ccd70b4b1d6.nc"
    )
    print(list(aurora.keys()))
    land_mask_aurora = aurora.lsm.values.astype(np.float32)[0]
    soil_type_aurora = aurora.slt.values.astype(np.float32)[0]
    geopotential_aurora = aurora.z.values.astype(np.float32)[0]

    plt.figure(3)
    plt.imshow(land_mask_aurora)
    plt.colorbar()
    plt.savefig(f"figures/land_mask_aurora.png", dpi=100)

    plt.figure(4)
    plt.imshow(soil_type_aurora)
    plt.colorbar()
    plt.savefig(f"figures/soil_type_aurora.png", dpi=100)

    plt.figure(5)
    plt.imshow(geopotential_aurora)
    plt.colorbar()
    plt.savefig(f"figures/geopotential_aurora.png", dpi=100)

    plt.figure(6)
    plt.imshow(topography_pangu - geopotential_aurora)
    plt.colorbar()
    plt.savefig(f"figures/geopotential_diff.png", dpi=100)
