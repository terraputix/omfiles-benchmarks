from om_benchmarks.helpers.formats import FormatFactory
from om_benchmarks.helpers.stats import (
    measure_execution,
    run_multiple_benchmarks,
)

read_formats_and_filenames = {
    "h5": "benchmark_files/data.h5",
    "h5hidefix": "benchmark_files/data.h5",
    "zarr": "benchmark_files/data.zarr",
    "zarrTensorStore": "benchmark_files/data.zarr",
    "zarrPythonViaZarrsCodecs": "benchmark_files/data.zarr",
    "nc": "benchmark_files/data.nc",
    "om": "benchmark_files/data.om",
}


def bm_read_all_formats(read_index, iterations):
    read_results = {}
    for format_name, file in read_formats_and_filenames.items():
        reader = FormatFactory.create_reader(format_name, file)

        @measure_execution
        def read():
            return reader.read(read_index)

        try:
            # sample_data = reader.read(args.read_index)  # Get sample data for verification
            read_stats = run_multiple_benchmarks(read, iterations)
            read_results[format_name] = read_stats

        except Exception as e:
            print(f"Error with {format_name}: {e}")
        finally:
            reader.close()

    return read_results
