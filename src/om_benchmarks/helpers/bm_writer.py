from zarr.core.buffer import NDArrayLike

from .formats import FormatFactory
from .stats import (
    measure_execution,
    run_multiple_benchmarks,
)

# Define separate dictionaries for read and write formats and filenames
write_formats_and_filenames = {
    "h5": "benchmark_files/data.h5",
    "zarr": "benchmark_files/data.zarr",
    "nc": "benchmark_files/data.nc",
    "om": "benchmark_files/data.om",
}


def bm_write_all_formats(chunk_size: tuple, iterations: int, data: NDArrayLike):
    write_results = {}
    for format_name, file in write_formats_and_filenames.items():
        writer = FormatFactory.create_writer(format_name, file)

        @measure_execution
        def write():
            writer.write(data, chunk_size)

        try:
            write_stats = run_multiple_benchmarks(write, iterations)
            write_stats.file_size = writer.get_file_size()
            write_results[format_name] = write_stats
        except Exception as e:
            print(f"Error with {format_name}: {e}")

    return write_results
