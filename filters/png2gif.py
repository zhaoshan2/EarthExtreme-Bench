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
import imageio
def create_gif(root, image_list, gif_name, duration=0.1):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(os.path.join(root, image_name)))
    imageio.mimsave(gif_name,frames,'GIF', duration=duration)
    return

def main():
    OUTPUT_DATA_DIR = Path(__file__).parent.parent / 'res/heatwave' / '2019-0650-GBR' / 'PNG'

    filenamelist = []
    for file in os.listdir(OUTPUT_DATA_DIR):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            filenamelist.append(filename)
    create_gif(OUTPUT_DATA_DIR, filenamelist, os.path.join(OUTPUT_DATA_DIR, 'animation.gif'))

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