from argparse import Namespace

from zarr.core.buffer import NDArrayLike

from helpers.args import parse_args
from helpers.formats import FormatFactory
from helpers.generate_data import generate_test_data
from helpers.plotting import create_benchmark_charts
from helpers.prints import print_data_info
from helpers.results import BenchmarkResultsManager
from helpers.schemas import RunMetadata
from helpers.stats import (
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

read_formats_and_filenames = {
    "h5": "benchmark_files/data.h5",
    "h5hidefix": "benchmark_files/data.h5",
    "zarr": "benchmark_files/data.zarr",
    "zarrTensorStore": "benchmark_files/data.zarr",
    "zarrPythonViaZarrsCodecs": "benchmark_files/data.zarr",
    "nc": "benchmark_files/data.nc",
    "om": "benchmark_files/data.om",
}


def bm_write_all_formats(args: Namespace, data: NDArrayLike):
    write_results = {}
    for format_name, file in write_formats_and_filenames.items():
        writer = FormatFactory.create_writer(format_name, file)

        @measure_execution
        def write():
            writer.write(data, args.chunk_size)

        try:
            write_stats = run_multiple_benchmarks(write, args.iterations)
            write_stats.file_size = writer.get_file_size()
            write_results[format_name] = write_stats
        except Exception as e:
            print(f"Error with {format_name}: {e}")

    return write_results


def bm_read_all_formats(args: Namespace):
    read_results = {}
    for format_name, file in read_formats_and_filenames.items():
        reader = FormatFactory.create_reader(format_name, file)

        @measure_execution
        def read():
            return reader.read(args.read_index)

        try:
            # sample_data = reader.read(args.read_index)  # Get sample data for verification
            read_stats = run_multiple_benchmarks(read, args.iterations)
            read_results[format_name] = read_stats

        except Exception as e:
            print(f"Error with {format_name}: {e}")
        finally:
            reader.close()

    return read_results


def main():
    args = parse_args()

    # Initialize results manager
    results_manager = BenchmarkResultsManager()

    if not args.plot_only:
        # Generate data and run benchmarks
        data = generate_test_data(args.array_size, noise_level=5, amplitude=20, offset=20)
        print_data_info(data, args.chunk_size)

        # Run benchmarks
        write_results = bm_write_all_formats(args, data)
        read_results = bm_read_all_formats(args)

        # Create type-safe metadata
        metadata = RunMetadata(
            array_shape=tuple(args.array_size) if hasattr(args.array_size, "__iter__") else (args.array_size,),
            chunk_shape=tuple(args.chunk_size) if hasattr(args.chunk_size, "__iter__") else (args.chunk_size,),
            iterations=args.iterations,
            read_index=getattr(args, "read_index", None),
        )

        # Save results and get DataFrame
        current_df = results_manager.save_and_display_results(write_results, read_results, metadata)
    else:
        # Load results from file
        current_df = results_manager.load_last_results()

    # Display current results
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
    main()
