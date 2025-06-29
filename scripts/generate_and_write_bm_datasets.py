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
from om_benchmarks.helpers.plotting import (
    create_and_save_file_size_chart,
    create_and_save_memory_usage_chart,
    create_and_save_perf_chart,
)
from om_benchmarks.helpers.prints import print_bm_results, print_data_info
from om_benchmarks.helpers.results import BenchmarkResultsManager
from om_benchmarks.helpers.schemas import RunMetadata
from om_benchmarks.helpers.script_utils import get_script_dirs


def main(
    download_dataset: bool = typer.Option(
        True,
        help="Whether to download ERA5 data. If False, must set generate_dataset=True to create synthetic data.",
    ),
    download_again: bool = typer.Option(False, help="Whether to download the dataset again even if it already exists."),
    target_download: str = typer.Option("downloaded_data.nc", help="Path where downloaded dataset will be saved."),
    generate_dataset: bool = typer.Option(
        False,
        help="Whether to generate synthetic data instead of downloading. Only used if download_dataset=False.",
    ),
    array_size: str = typer.Option(
        "(100, 100, 200)",
        help="Size of the array for synthetic data generation in the format '(x, y, z)'.",
    ),
    chunk_size: str = typer.Option("(50, 50, 1440)", help="Chunk size for writing data in the format '(x, y, z)'."),
    iterations: int = typer.Option(1, help="Number of iterations to run for each benchmark."),
):
    # FIXME: Find a way to effectively benchmark against various chunk sizes.
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
    results_dir, plots_dir = get_script_dirs(__file__)
    results_manager = BenchmarkResultsManager(results_dir)
    metadata = RunMetadata(
        array_shape=data.shape,
        chunk_shape=_chunk_size,
        iterations=iterations,
    )
    write_results = bm_write_all_formats(chunk_size=_chunk_size, metadata=metadata, data=data)

    current_df = results_manager.save_and_display_results(write_results, type="write")
    print_bm_results(results_manager=results_manager, results_df=current_df)
    # Create visualizations
    create_and_save_perf_chart(current_df, plots_dir)
    create_and_save_file_size_chart(current_df, plots_dir)
    create_and_save_memory_usage_chart(current_df, plots_dir)

    print("All formats saved successfully!")


if __name__ == "__main__":
    typer.run(main)
