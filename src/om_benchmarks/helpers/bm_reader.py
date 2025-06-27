from typing import Optional, Tuple

from om_benchmarks.helpers.formats import FormatFactory
from om_benchmarks.helpers.schemas import BenchmarkStats
from om_benchmarks.helpers.stats import (
    measure_execution,
    run_multiple_benchmarks,
)

read_formats_and_filenames = {
    "h5": "benchmark_files/data.h5",
    # "h5hidefix": "benchmark_files/data.h5",
    "zarr": "benchmark_files/data.zarr",
    "zarrTensorStore": "benchmark_files/data.zarr",
    "zarrPythonViaZarrsCodecs": "benchmark_files/data.zarr",
    "nc": "benchmark_files/data.nc",
    "om": "benchmark_files/data.om",
}


def bm_read_all_formats(read_index, iterations) -> Tuple[dict[str, BenchmarkStats], tuple, Optional[tuple]]:
    read_results = {}
    # TODO: Better approach to store chunk and array shape
    array_shape: tuple = (1000, 1000)
    chunk_shape: Optional[tuple] = (100, 100)
    for format_name, file in read_formats_and_filenames.items():
        reader = FormatFactory.create_reader(format_name, file)
        array_shape = reader.shape
        chunk_shape = reader.chunk_shape

        @measure_execution
        def read():
            return reader.read(read_index)

        try:
            sample_data = reader.read(read_index)  # Get sample data for verification
            print(sample_data)
            read_stats = run_multiple_benchmarks(read, iterations)
            read_results[format_name] = read_stats

        except Exception as e:
            print(f"Error with {format_name}: {e}")
        finally:
            reader.close()

    return read_results, array_shape, chunk_shape
