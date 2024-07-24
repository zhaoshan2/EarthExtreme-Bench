from datetime import datetime, timedelta

import numpy as np
import xarray as xr


def latlon2xy(lat: xr.DataArray, lon: xr.DataArray) -> tuple[np.array, np.array]:
    """
    Latitude, longitude to pixel index
    """
    x = (90 - lat.values.astype(np.float32)) * 4
    y = (lon.values.astype(np.float32)) * 4
    return x, y


def crop_mask(mask: np.array, lat: xr.DataArray, lon: xr.DataArray) -> np.array:
    x, y = latlon2xy(lat, lon)
    mask_cropped = mask[
        int(np.min(x)) : int(np.max(x)), int(np.min(y)) : int(np.max(y))
    ]
    return mask_cropped.astype(np.float32)


def west2numbers(min_lon, max_lon):
    if max_lon <= 0:
        return min(360 + min_lon, 359.75), min(360 + max_lon, 359.75)
    else:
        return min_lon, max_lon


def months_within_date_range(start_date, end_date):
    """
    Returns a list of months within the given date range.
    """
    months_list = []
    current_date = start_date

    while current_date <= end_date:
        months_list.append(current_date.strftime("%Y%m"))
        # Move to the next month
        month = current_date.month
        year = current_date.year
        if month == 12:
            month = 1
            year += 1
        else:
            month += 1
        current_date = datetime(year, month, 1)
    return list(months_list)


def days_within_date_range(start_date, end_date):
    """
    Returns a list of days within the given date range.
    """

    current_date = start_date
    days_list = []

    while current_date <= end_date:
        days_list.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)

    return days_list
