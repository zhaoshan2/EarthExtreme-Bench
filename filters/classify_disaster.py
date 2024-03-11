"""
Filter disasters
1. Disaster group: Natural
2. Disaster subgroup: Hydrological, Meteorological, Climatological
3. Disaster Subtype: heatwave, coldwave, tropical cyclone
"""
import pandas as pd
import os
from pathlib import Path
import argparse
CURR_FOLDER_PATH = Path(__file__).parent
DATA_FOLDER_PATH = CURR_FOLDER_PATH.parent.parent / 'data_storage_home' / 'data' / 'disaster'
INPUT_CSV_PATH = DATA_FOLDER_PATH / 'input_csv' / "public_emdat_20240207.csv"
OUTPUT_FOLDER = DATA_FOLDER_PATH / "output_csv"
START_YEAR = 2019
END_YEAR = 2022
if __name__ == "__main__":

    totalDisasters = pd.read_csv(INPUT_CSV_PATH, encoding='unicode_escape')

    totalDisasters = totalDisasters[totalDisasters['Start Year'] >= START_YEAR]
    totalDisasters = totalDisasters[totalDisasters['End Year'] <= END_YEAR]
    print(f"{totalDisasters.shape[0]} disasters")


    natureDisasters = totalDisasters[totalDisasters['Disaster Group'] == 'Natural']

    hydroDisasters = natureDisasters[natureDisasters['Disaster Subgroup'] == 'Hydrological']
    meteoroDisasters = natureDisasters[natureDisasters['Disaster Subgroup'] == 'Meteorological']
    climatoDisasters = natureDisasters[natureDisasters['Disaster Subgroup'] == 'Climatological']

    print(f"{hydroDisasters.shape[0]} hydrological, {meteoroDisasters.shape[0]} meteorological, and {climatoDisasters.shape[0]} climatological disasters.")

    # # Forest fires
    # wildfire = climatoDisasters[climatoDisasters['Disaster Type'] == 'Wildfire']
    # forestfire = wildfire[wildfire['Disaster Subtype'] == 'Forest fire']
    #
    #
    # print(f"{forestfire.shape[0]} forest fires.")
    #
    # if not os.path.exists(OUTPUT_FOLDER):
    #     os.mkdir(OUTPUT_FOLDER)
    #
    # forestfire.to_csv(os.path.join(OUTPUT_FOLDER, "forestfire.csv"), index=False)
    #
    # # Floods
    # floods = hydroDisasters[hydroDisasters['Disaster Type'] == 'Flood']
    # floods.to_csv(os.path.join(OUTPUT_FOLDER, "flood.csv"), index=False)
    # print(f"{floods.shape[0]} floods.")

    ## heatwave, coldwave
    extremeTemperatures = meteoroDisasters[meteoroDisasters['Disaster Type'] == 'Extreme temperature']
    heatwave = extremeTemperatures[extremeTemperatures['Disaster Subtype'] == 'Heat wave']
    coldwave = extremeTemperatures[extremeTemperatures['Disaster Subtype'] == 'Cold wave']

    heatwave.to_csv(os.path.join(OUTPUT_FOLDER, f"heatwave_{str(START_YEAR)}to{str(END_YEAR)}.csv"), index=False)
    # coldwave.to_csv(os.path.join(OUTPUT_FOLDER, f"coldwave_{str(START_YEAR)}to{str(END_YEAR)}.csv"), index=False)
    print(f"{heatwave.shape[0]} heatwaves.")
    print(f"{coldwave.shape[0]} coldwaves.")

    # tropical cyclones
    tropicalCyclone = meteoroDisasters[meteoroDisasters['Disaster Subtype'] == 'Tropical cyclone']
    # tropicalCyclone.to_csv(os.path.join(OUTPUT_FOLDER, f"tropicalCyclone_{str(YEAR)}.csv"), index=False)
    # print(f"{tropicalCyclone.shape[0]} tropical cyclones.")