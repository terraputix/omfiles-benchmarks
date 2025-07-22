import os
from functools import lru_cache

import numpy as np
import xarray as xr


def configure_era5_request():
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "year": ["2024"],
        "month": [
            "01",
            # "02", "03", "04", "05", "06"
        ],
        "day": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
        ],
        "time": [
            "00:00",
            "01:00",
            "02:00",
            "03:00",
            "04:00",
            "05:00",
            "06:00",
            "07:00",
            "08:00",
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
            "17:00",
            "18:00",
            "19:00",
            "20:00",
            "21:00",
            "22:00",
            "23:00",
        ],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    return dataset, request


@lru_cache(maxsize=1)
def read_era5_data(file) -> np.ndarray:
    if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")

    print(f"Reading t2m variable from {file}...")
    ds = xr.open_dataset(file)
    data = ds["t2m"].values
    print(f"Loaded t2m data with shape: {data.shape}")
    return data


@lru_cache(maxsize=1)
def read_era5_data_to_temporal(file) -> np.ndarray:
    if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")

    print(f"Reading t2m variable from {file}...")
    ds = xr.open_dataset(file)
    data = ds["t2m"].values
    # convert to temporal format. Our data has a shape of (time, lat, lon) but
    # we want to have it shaped like (lat, lon, time)
    data = data.transpose((1, 2, 0)).copy()

    print(f"Loaded t2m data with shape: {data.shape}")
    return data
