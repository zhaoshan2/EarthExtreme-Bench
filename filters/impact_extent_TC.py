import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import re
import numpy as np
"""
Look up table between country and longitude, latitude extend
Data source: https://latitudelongitude.org/
(Some extents are larger than the recorded cities' extent.)
"""
CURR_FOLDER_PATH = Path(__file__).parent
DISASTER_PATH = CURR_FOLDER_PATH.parent.parent / 'data_storage_home' / 'data' / 'disaster'
INPUT_POINTS_PATH = DISASTER_PATH / "input_csv" / "ibtracs.since1980.list.v04r00.csv"
INPUT_EVENTS_PATH = DISASTER_PATH / "output_csv" / "tropicalCyclone_2019.csv"
OUTPUT_TC_PATH = DISASTER_PATH / "output_csv" / 'tropicalCyclone'/ "tropicalCyclone_2019_ibtracs.csv"
OUTPUT_CSV_PATH = DISASTER_PATH / "output_csv" / 'tropicalCyclone' / "tropicalCyclone_2019_ibtracs_emdat.csv"
if __name__ == "__main__":

    """
    Tropical cyclone from ibtracs
    """

    tropical_events = pd.read_csv(INPUT_EVENTS_PATH,
                                  encoding='unicode_escape')
    #
    #
    cyclone_points = pd.read_csv(INPUT_POINTS_PATH, skiprows=[1])
    cyclone_points = cyclone_points[cyclone_points['SEASON']==2019]

    tc_sids = cyclone_points.SID.unique()

    i = -1
    empty_df = pd.DataFrame(columns=tropical_events.columns)
    for sid in tc_sids:
        i += 1
        records = cyclone_points[cyclone_points['SID']==sid]
        min_lon_impact = min(records['LON'])
        max_lon_impact = max(records['LON'])
        min_lat_impact = min(records['LAT'])
        max_lat_impact = max(records['LAT'])

        times = records['ISO_TIME']
        # Convert string times to datetime objects 2018-09-13 06:00:00
        datetime_objects = [datetime.strptime(time, '%Y-%m-%d %H:%M:%S') for time in times]

        # Find the earliest time
        earliest_time = min(datetime_objects)
        latest_time = max(datetime_objects)
        start_year = earliest_time.year
        start_month = earliest_time.month
        start_day = earliest_time.day

        end_year = latest_time.year
        end_month = latest_time.month
        end_day = latest_time.day

        empty_df.loc[i, 'DisNo.'] = f"TC_{sid}"
        empty_df.loc[i, 'Classification Key'] = 'nat-met-sto-tro'
        empty_df.loc[i, 'Disaster Group'] = 'Natural'
        empty_df.loc[i, 'Disaster Subgroup'] = 'Meteorological'
        empty_df.loc[i, 'Disaster Type'] = 'Storm'
        empty_df.loc[i, 'Disaster Subtype'] = 'Tropical cyclone'
        empty_df.loc[i, 'Event Name'] = records.iloc[0].NAME
        empty_df.loc[i, 'Start Year'] = start_year
        empty_df.loc[i, 'Start Month'] = start_month
        empty_df.loc[i, 'Start Day'] = start_day
        empty_df.loc[i, 'End Year'] = end_year
        empty_df.loc[i, 'End Month'] = end_month
        empty_df.loc[i, 'End Day'] = end_day
        empty_df.loc[i, 'min_lon'] = min_lon_impact
        empty_df.loc[i, 'max_lon'] = max_lon_impact
        empty_df.loc[i, 'min_lat'] = min_lat_impact
        empty_df.loc[i, 'max_lat'] = max_lat_impact

    empty_df.to_csv(OUTPUT_TC_PATH, index=False)

    
    cyclone_trajectories = pd.read_csv(OUTPUT_TC_PATH)
    # cyclone_trajectories = cyclone_trajectories.astype({'Start Year': 'int'})
    cyclone_trajectories = cyclone_trajectories[cyclone_trajectories['Start Year'] >= 2019] # index keeps the same before and after selection

    # print(cyclone_trajectories.index)
    for i in cyclone_trajectories.index:
        cyclone_trajectory = cyclone_trajectories.loc[i]
        event_name = cyclone_trajectory['Event Name'].capitalize() #"Penny"
        tc_name = tropical_events['Event Name'].values #"Cyclone 'Penny'"
        if event_name in tc_name:
            rows = tropical_events[tropical_events['Event Name']==event_name]

            isos = ','.join(map(str, list(set(rows['ISO'].values))))
            cyclone_trajectories.loc[i, 'ISO'] = isos

            countries = ','.join(map(str, list(set(rows['Country'].values))))
            cyclone_trajectories.loc[i, 'Country'] = countries
            subregions = ','.join(map(str, list(set(rows['Subregion'].values))))
            cyclone_trajectories.loc[i, 'Subregion'] = subregions
            regions = ','.join(map(str, list(set(rows['Region'].values))))
            cyclone_trajectories.loc[i,'Region'] = regions
            locations = ','.join(map(str, list(set(rows['Location'].values))))
            cyclone_trajectories.loc[i,'Location'] = locations
            origins = ','.join(map(str, list(set(rows['Origin'].values))))
            cyclone_trajectories.loc[i,'Origin'] = origins.replace("nan,", "")
            associated_types = ','.join(map(str, list(set(rows['Associated Types'].values))))
            cyclone_trajectories.loc[i,'Associated Types'] = associated_types.replace("nan,", "")
            cyclone_trajectories.loc[i,'OFDA Response'] = 'Yes' if 'Yes' in rows['OFDA Response'] else 'No'
            cyclone_trajectories.loc[i,'Appeal'] = 'Yes' if 'Yes' in rows['Appeal'] else 'No'
            cyclone_trajectories.loc[i,'Declaration'] = 'Yes' if 'Yes' in rows['Declaration'] else 'No'
            cyclone_trajectories.loc[i,'Magnitude'] = max(rows['Magnitude'].values.astype(np.float32))
            cyclone_trajectories.loc[i,'Magnitude Scale'] = rows['Magnitude Scale'].values[0]
            cyclone_trajectories.loc[i,'Total Deaths'] = sum(rows['Total Deaths'].values.astype(np.float32))
            cyclone_trajectories.loc[i,'No. Injured'] = sum(rows['No. Injured'].values.astype(np.float32))
            cyclone_trajectories.loc[i,'No. Affected'] = sum(rows['No. Affected'].values.astype(np.float32))
            cyclone_trajectories.loc[i,'No. Homeless'] = sum(rows['No. Homeless'].values.astype(np.float32))
            cyclone_trajectories.loc[i,'Total Affected'] = sum(rows['Total Affected'].values.astype(np.float32))
            cyclone_trajectories.loc[i,"Insured Damage ('000 US$)"] = sum(rows["Insured Damage ('000 US$)"].values.astype(np.float32))
            cyclone_trajectories.loc[i,"Insured Damage, Adjusted ('000 US$)"] = sum(rows["Insured Damage, Adjusted ('000 US$)"].values.astype(np.float32))
            cyclone_trajectories.loc[i,"Total Damage ('000 US$)"] = sum(rows["Total Damage ('000 US$)"].values.astype(np.float32))
            cyclone_trajectories.loc[i,"Total Damage, Adjusted ('000 US$)"] = sum(rows["Total Damage, Adjusted ('000 US$)"].values.astype(np.float32))

    # event_names_clean = []
    # for event_name in event_names:
    #     name = re.findall(r"'(.*?)'", event_name)
    #     if name:
    #         event_names_clean.append(name[0])
    # event_names_clean = list(set(event_names_clean))
    #
    # event_names_clean_cap = [n.upper() for n in event_names_clean]
    # print(event_names_clean_cap)
    cyclone_trajectories.replace('nan', '', inplace=True)
    cyclone_trajectories.to_csv(OUTPUT_CSV_PATH, index=False)


