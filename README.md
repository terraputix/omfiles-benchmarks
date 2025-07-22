# Omfiles Benchmarks

This project compares the performance of different multidimensional array storage formats:

- [netcdf4-python](https://github.com/Unidata/netcdf4-python)
- [h5py](https://github.com/h5py/h5py)
- [zarr-python](https://github.com/zarr-developers/zarr-python/)
- [zarrs-python](https://github.com/zarrs/zarrs-python)
- [python-omfiles](https://github.com/open-meteo/python-omfiles/)

## Development

Install uv according to the [documentation](https://docs.astral.sh/uv/getting-started/installation/). Then:

```bash
# download era5 data that will be used in the benchmarks
# more information with:
# uv run scripts/download_era5.py --help
uv run scripts/download_era5.py

# uv run scripts/era5_read_write_time_memory.py --help
# generate the files measuring how long generation takes
uv run scripts/era5_read_write_time_memory.py --mode time --op-mode write --iterations 1
# read from the files using different chunk sizes measuring the time
uv run scripts/era5_read_write_time_memory.py --mode time --op-mode read --iterations 10
```
