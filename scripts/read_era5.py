import os
from typing import cast

import typer

from om_benchmarks.helpers.AsyncTyper import AsyncTyper
from om_benchmarks.helpers.bm_reader import BMResultsDict, bm_read_format
from om_benchmarks.helpers.bm_writer import bm_write_format
from om_benchmarks.helpers.constants import DEFAULT_READ_FORMATS
from om_benchmarks.helpers.era5 import read_era5_data
from om_benchmarks.helpers.parse_tuple import parse_tuple
from om_benchmarks.helpers.plotting import (
    create_and_save_memory_usage_chart,
    create_and_save_perf_chart,
)
from om_benchmarks.helpers.prints import print_bm_results
from om_benchmarks.helpers.results import BenchmarkResultsManager
from om_benchmarks.helpers.schemas import RunMetadata
from om_benchmarks.helpers.script_utils import get_era5_path_for_format, get_script_dirs

app = AsyncTyper()


@app.command()
async def main(
    chunk_size: str = typer.Option("(5, 5, 1440)", help="Chunk size for writing data in the format '(x, y, z)'."),
    read_index: str = typer.Option(
        "(100, 200, 0..20)",
        help="Index range to read from datasets in the format '(x, y, z)' or '(x, y, start..end)' for slices.",
    ),
    iterations: int = typer.Option(10, help="Number of times to repeat each benchmark for more reliable results."),
):
    # FIXME: Improve format configuration
    formats = DEFAULT_READ_FORMATS
    # FIXME: Find a way to effectively benchmark against various chunk sizes.
    _chunk_size = cast(tuple[int], parse_tuple(chunk_size))
    del chunk_size
    print(f"Chunk size: {_chunk_size}")

    _read_index = parse_tuple(read_index)
    del read_index
    print("Read index:", _read_index)

    # Initialize results manager
    results_dir, plots_dir = get_script_dirs(__file__)
    results_manager = BenchmarkResultsManager(results_dir)

    read_results: BMResultsDict = {}
    for format in formats:
        reader_type = format.reader_class
        file_path = get_era5_path_for_format(reader_type, chunk_size=_chunk_size)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}. Generating it ...")

            target_download = "downloaded_data.nc"
            data = read_era5_data(target_download)

            metadata = RunMetadata(
                array_shape=data.shape,
                chunk_shape=_chunk_size,
                iterations=1,
            )
            await bm_write_format(_chunk_size, metadata, format.writer_class, file_path.__str__(), data)

        result = await bm_read_format(_read_index, iterations, reader_type, file_path.__str__(), False)
        read_results[format] = result

    current_df = results_manager.save_and_display_results(read_results, type="read")

    print_bm_results(results_manager=results_manager, results_df=current_df)
    # Create visualizations
    create_and_save_perf_chart(current_df, plots_dir)
    create_and_save_memory_usage_chart(current_df, plots_dir)


if __name__ == "__main__":
    app()
