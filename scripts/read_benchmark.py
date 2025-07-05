from typing import cast

import typer

from om_benchmarks.helpers.AsyncTyper import AsyncTyper
from om_benchmarks.helpers.bm_reader import bm_read_all_formats
from om_benchmarks.helpers.formats import AvailableFormats
from om_benchmarks.helpers.parse_tuple import parse_tuple
from om_benchmarks.helpers.plotting import (
    create_and_save_memory_usage_chart,
    create_and_save_perf_chart,
)
from om_benchmarks.helpers.results import BenchmarkResultsDF
from om_benchmarks.helpers.script_utils import get_script_dirs

app = AsyncTyper()

READ_FORMATS: list[AvailableFormats] = [
    AvailableFormats.HDF5,
    AvailableFormats.HDF5Hidefix,
    AvailableFormats.Zarr,
    AvailableFormats.ZarrTensorStore,
    AvailableFormats.ZarrPythonViaZarrsCodecs,
    AvailableFormats.NetCDF,
    AvailableFormats.OM,
]


@app.command()
async def main(
    read_index: str = typer.Option(
        "(...)",
        help="Index range to read from datasets in the format '(x, y, z)' or '(x, y, start..end)' for slices.",
    ),
    iterations: int = typer.Option(10, help="Number of times to repeat each benchmark for more reliable results."),
    chunk_size: str = typer.Option("(5, 5, 1440)", help="Chunk size for writing data in the format '(x, y, z)'."),
    plot_only: bool = typer.Option(
        False, help="If True, skips running benchmarks and only plots results from the last saved benchmark run."
    ),
):
    # FIXME: Find a way to effectively benchmark against various chunk sizes.
    _chunk_size = cast(tuple[int], parse_tuple(chunk_size))
    del chunk_size

    _read_index = parse_tuple(read_index)
    del read_index
    print("Read index:", _read_index)

    results_dir, plots_dir = get_script_dirs(__file__)
    results_df = BenchmarkResultsDF(results_dir)

    if not plot_only:
        read_results = await bm_read_all_formats(_read_index, iterations, READ_FORMATS)
        results_df.append(read_results)
        results_df.save_results()
    else:
        # Load results from file
        results_df.load_last_results()

    results_df.print_summary()
    # Create visualizations
    create_and_save_perf_chart(results_df.df, plots_dir)
    create_and_save_memory_usage_chart(results_df.df, plots_dir)


if __name__ == "__main__":
    app()
