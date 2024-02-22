import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
import xarray as xr
import pandas as pd
import os
from pathlib import Path
import seaborn as sns
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
    DISASTER = 'coldwave'
    CURR_FOLDER_PATH = Path(__file__).parent
    OUTPUT_DATA_DIR = CURR_FOLDER_PATH.parent / 'data' / f'{DISASTER}' / '2022-0800-MNG' / 'PNG'

    filenamelist = []
    for file in os.listdir(OUTPUT_DATA_DIR):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            if DISASTER == "tropicalCyclone":
                if '_msl_' in filename:
                    filenamelist.append(filename)
            else:
                filenamelist.append(filename)
    if DISASTER == "tropicalCyclone":
        filenamelist = sorted(filenamelist, key=lambda x: int(x[29:-4]))
    else:
        filenamelist = sorted(filenamelist, key=lambda x: int(x[18:-4]))
    print(filenamelist)
    create_gif(OUTPUT_DATA_DIR, filenamelist, os.path.join(OUTPUT_DATA_DIR, 'animation_t2m.gif'))

if __name__ == "__main__":

    # sns.scatterplot(data=file, x="LON", y="LAT", hue="ISO_TIME",legend=False)

    main()

    """
    installation error 
    python3.8.8
    
    pip install ecmwflibs
    pip install eccodes==1.3.1
    pip install cfgrib
    """