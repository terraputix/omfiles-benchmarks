import polars as pl
from omfiles.types import BasicSelection
from zarr.core.buffer import NDArrayLike

from om_benchmarks.helpers.results import BenchmarkResultsManager


def print_data_info(data: NDArrayLike, chunk_size: BasicSelection) -> None:
    print(
        f"""
Data shape: {data.shape}
Data type: {data.dtype}
Chunk size: {chunk_size}
"""
    )


def print_bm_results(results_manager: BenchmarkResultsManager, results_df: pl.DataFrame) -> None:
    # Display current results
    with pl.Config(tbl_cols=-1, tbl_rows=-1):  # display all columns and rows
        print("\n" + "=" * 80)
        print("CURRENT BENCHMARK RESULTS")
        print("=" * 80)
        summary_df = results_manager.get_current_results_summary(results_df)
        print(summary_df)
