from typing import Tuple, cast

import polars as pl
import typer

from om_benchmarks.helpers.bm_reader import bm_read_all_formats
from om_benchmarks.helpers.bm_writer import bm_write_all_formats
from om_benchmarks.helpers.generate_data import generate_test_data
from om_benchmarks.helpers.parse_tuple import parse_tuple
from om_benchmarks.helpers.plotting import create_benchmark_charts
from om_benchmarks.helpers.prints import print_data_info
from om_benchmarks.helpers.results import BenchmarkResultsManager
from om_benchmarks.helpers.schemas import RunMetadata


def main(
    array_size: str = "(10, 10, 1, 10, 10, 10)",
    chunk_size: str = "(10, 5, 1, 1, 1, 1)",
    read_index: str = "(0, 0, 0, 0, ...)",
    iterations: int = 10,
    plot_only: bool = False,
):
    _array_size = cast(Tuple[int, ...], parse_tuple(array_size))
    _chunk_size = cast(Tuple[int, ...], parse_tuple(chunk_size))
    _read_index = parse_tuple(read_index)
    del array_size, chunk_size, read_index
    print("Array size:", _array_size)
    print("Chunk size:", _chunk_size)
    print("Read index:", _read_index)

    # Initialize results manager
    results_manager = BenchmarkResultsManager()
    if not plot_only:
        # Generate data and run benchmarks
        data = generate_test_data(_array_size, noise_level=5, amplitude=20, offset=20)
        print_data_info(data, _chunk_size)

        # Run benchmarks
        write_results = bm_write_all_formats(_chunk_size, iterations, data)
        read_results = bm_read_all_formats(_read_index, iterations)

        # Create type-safe metadata
        metadata = RunMetadata(
            array_shape=_array_size,
            chunk_shape=_chunk_size,
            iterations=iterations,
            read_index=_read_index,
        )

        # Save results and get DataFrame
        current_df = results_manager.save_and_display_results(write_results, read_results, metadata)
    else:
        # Load results from file
        current_df = results_manager.load_last_results()

    # Display current results
    with pl.Config(tbl_cols=-1, tbl_rows=-1):  # display all columns and rows
        print("\n" + "=" * 80)
        print("CURRENT BENCHMARK RESULTS")
        print("=" * 80)
        summary_df = results_manager.get_current_results_summary(current_df)
        print(summary_df)

        # Show performance comparison
        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON")
        print("=" * 80)
        comparison_df = results_manager.get_performance_comparison(current_df)
        print(comparison_df)

    # Create visualizations
    create_benchmark_charts(current_df)


if __name__ == "__main__":
    typer.run(main)
