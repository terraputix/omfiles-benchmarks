# Omfiles Benchmarks

This project compares the performance of different multidimensional array storage formats:

- [netcdf4-python](https://github.com/Unidata/netcdf4-python)
- [h5py](https://github.com/h5py/h5py)
- [zarr-python](https://github.com/zarr-developers/zarr-python/)
- [zarrs-python](https://github.com/zarrs/zarrs-python)
- [python-omfiles](https://github.com/open-meteo/python-omfiles/)

## Development

Install uv according to the [documentation](https://docs.astral.sh/uv/getting-started/installation/). Then:

```
uv run scripts/read_benchmark.py
```
