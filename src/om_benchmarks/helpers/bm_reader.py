from typing import Tuple

from om_benchmarks.helpers.formats import FormatFactory
from om_benchmarks.helpers.schemas import BenchmarkStats, RunMetadata
from om_benchmarks.helpers.stats import (
    measure_execution,
    run_multiple_benchmarks,
)

read_formats_and_filenames = {
    "h5": "benchmark_files/data.h5",
    # "h5hidefix": "benchmark_files/data.h5",
    "zarr": "benchmark_files/data.zarr",
    # "zarrTensorStore": "benchmark_files/data.zarr",
    # "zarrPythonViaZarrsCodecs": "benchmark_files/data.zarr",
    "nc": "benchmark_files/data.nc",
    "om": "benchmark_files/data.om",
}


def bm_read_all_formats(
    read_index, iterations, plot_read_data: bool = False
) -> dict[str, Tuple[BenchmarkStats, RunMetadata]]:
    read_results: dict[str, Tuple[BenchmarkStats, RunMetadata]] = {}
    for format_name, file in read_formats_and_filenames.items():
        print(f"Benchmarking {format_name}...")
        reader = FormatFactory.create_reader(format_name, file)

        @measure_execution
        def read():
            return reader.read(read_index)

        try:
            if plot_read_data:
                sample_data = reader.read(read_index)  # Get sample data for verification
                print(sample_data)
            read_stats = run_multiple_benchmarks(read, iterations)
            read_results[format_name] = (
                read_stats,
                RunMetadata(array_shape=reader.shape, chunk_shape=reader.chunk_shape, iterations=iterations),
            )

        except Exception as e:
            print(f"Error with {format_name}: {e}")
        finally:
            reader.close()

    return read_results
