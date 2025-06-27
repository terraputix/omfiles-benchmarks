import os
from typing import cast

import cdsapi
import typer
import xarray as xr
from zarr.core.buffer import NDArrayLike

from om_benchmarks.helpers.bm_writer import bm_write_all_formats
from om_benchmarks.helpers.era5 import configure_era5_request
from om_benchmarks.helpers.generate_data import generate_test_data
from om_benchmarks.helpers.parse_tuple import parse_tuple

# from om_benchmarks.helpers.plotting import create_benchmark_charts
from om_benchmarks.helpers.prints import print_bm_results, print_data_info
from om_benchmarks.helpers.results import BenchmarkResultsManager
from om_benchmarks.helpers.schemas import RunMetadata


def main(
    download_dataset: bool = True,
    download_again: bool = False,
    target_download: str = "downloaded_data.nc",
    generate_dataset: bool = False,
    array_size: str = "(100, 100, 200)",
    chunk_size: str = "(5, 5, 1440)",
    iterations: int = 1,
):
    # FIXME: Find a way to effectly benchmark against various chunk sizes.
    _chunk_size = cast(tuple[int], parse_tuple(chunk_size))
    del chunk_size

    if download_dataset:
        if os.path.exists(target_download) and not download_again:
            print(f"Dataset already exists at {target_download}. Skipping download.")
        else:
            dataset, request = configure_era5_request()
            client = cdsapi.Client()
            client.retrieve(dataset, request).download(target_download)

        print(f"Reading t2m variable from {target_download}...")
        ds = xr.open_dataset(target_download)
        data = cast(NDArrayLike, ds["t2m"].values)
        print(f"Loaded t2m data with shape: {data.shape}")

    elif generate_dataset:
        # Generate data and run benchmarks
        _array_size = parse_tuple(array_size)
        del array_size
        data = generate_test_data(_array_size, noise_level=5, amplitude=20, offset=20)
    else:
        raise ValueError("Either download_dataset or generate_dataset must be True")

    print_data_info(data, _chunk_size)

    # Run benchmarks
    results_manager = BenchmarkResultsManager()
    write_results = bm_write_all_formats(chunk_size=_chunk_size, iterations=1, data=data)
    metadata = RunMetadata(
        array_shape=data.shape,
        chunk_shape=_chunk_size,
        iterations=iterations,
    )
    current_df = results_manager.save_and_display_results(write_results, metadata, type="write")
    print_bm_results(results_manager=results_manager, results_df=current_df)
    # Create visualizations
    # create_benchmark_charts(current_df)

    print("All formats saved successfully!")


if __name__ == "__main__":
    typer.run(main)
