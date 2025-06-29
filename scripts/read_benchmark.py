import asyncio

import typer

from om_benchmarks.helpers.AsyncTyper import AsyncTyper
from om_benchmarks.helpers.bm_reader import bm_read_all_formats
from om_benchmarks.helpers.parse_tuple import parse_tuple
from om_benchmarks.helpers.plotting import (
    create_and_save_memory_usage_chart,
    create_and_save_perf_chart,
)
from om_benchmarks.helpers.prints import print_bm_results
from om_benchmarks.helpers.results import BenchmarkResultsManager
from om_benchmarks.helpers.script_utils import get_script_dirs

app = AsyncTyper()


@app.command()
async def main(
    read_index: str = typer.Option(
        "(100, 200, 0..20)",
        help="Index range to read from datasets in the format '(x, y, z)' or '(x, y, start..end)' for slices.",
    ),
    iterations: int = typer.Option(10, help="Number of times to repeat each benchmark for more reliable results."),
    plot_only: bool = typer.Option(
        False, help="If True, skips running benchmarks and only plots results from the last saved benchmark run."
    ),
):
    _read_index = parse_tuple(read_index)
    del read_index
    print("Read index:", _read_index)

    # Initialize results manager
    results_dir, plots_dir = get_script_dirs(__file__)
    results_manager = BenchmarkResultsManager(results_dir)
    if not plot_only:
        read_results = await bm_read_all_formats(_read_index, iterations)
        current_df = results_manager.save_and_display_results(read_results, type="read")
    else:
        # Load results from file
        current_df = results_manager.load_last_results()

    print_bm_results(results_manager=results_manager, results_df=current_df)
    # Create visualizations
    create_and_save_perf_chart(current_df, plots_dir)
    create_and_save_memory_usage_chart(current_df, plots_dir)


if __name__ == "__main__":
    app()
