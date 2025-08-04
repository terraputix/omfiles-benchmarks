import gc
import os
import resource
import shutil
import statistics
from pathlib import Path
from typing import List

import polars as pl
import typer

from om_benchmarks.AsyncTyper import AsyncTyper
from om_benchmarks.configurations import (
    REGISTERED_FORMAT_INVENTORY,
    get_config_by_hash,
)
from om_benchmarks.era5 import read_era5_data_to_temporal
from om_benchmarks.modes import MetricMode, OpMode
from om_benchmarks.mse import MSECache, mean_squared_error_destructive
from om_benchmarks.plotting.read_write_plots import (
    create_scatter_size_vs_mode,
    create_violin_plot,
)
from om_benchmarks.read_indices import generate_read_indices
from om_benchmarks.results import BenchmarkResultsDF
from om_benchmarks.schemas import BenchmarkRecord, BenchmarkStats
from om_benchmarks.script_utils import get_era5_path_for_hashed_config, get_script_dirs
from om_benchmarks.stats import _clear_cache, measure_memory, measure_time

app = AsyncTyper(pretty_exceptions_enable=False)

data_shape = (721, 1440, 744)

read_ranges: list[tuple[int, int, int]] = [
    (1, 1, 20),
    (5, 5, 200),
    (1, 1, 744),
    (5, 5, 744),
    (20, 20, 744),
    (721, 1440, 1),
]

# Limit memory usage to prevent freezing
resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024 * 25, 1024 * 1024 * 1024 * 25))


@app.command()
async def main(
    iterations: int = typer.Option(2, help="Number of times to repeat each benchmark for more reliable results."),
    clear_cache: bool = typer.Option(True, help="Clear the cache during single benchmark iterations."),
    mode: MetricMode = typer.Option(MetricMode.TIME, help="Benchmark measurement mode: 'time' or 'memory'."),
    op_mode: OpMode = typer.Option(OpMode.READ, help="Operation mode: 'read' or 'write'."),
    plot_only: bool = typer.Option(False, help="Only plot the results without running the benchmarks."),
):
    # Gather results
    results_dir, plots_dir = get_script_dirs(__file__)
    mse_cache = MSECache(Path(results_dir / "mse_cache.json"))

    measure_func = measure_memory if mode == MetricMode.MEMORY else measure_time

    # Generate read_indices: for each read_range, generate `iterations` tuples of slices
    read_indices = generate_read_indices(data_shape, iterations, read_ranges)

    for chunk_size, read_formats in REGISTERED_FORMAT_INVENTORY.items():
        bm_results: list[BenchmarkRecord] = []
        if not plot_only:
            for format, config_hash in read_formats:
                file_path = get_era5_path_for_hashed_config(format, chunk_size=chunk_size, hash=config_hash)

                config = get_config_by_hash(config_hash)

                if op_mode == OpMode.WRITE:
                    try:
                        writer = format.writer_class(file_path.__str__(), config)
                    except Exception as e:
                        print(f"No writer for {format.name}: {e}")
                        continue

                    # Write mode: always overwrite files
                    print(f"Writing file {file_path.__str__()} with format {format.name}")

                    # Remove existing file if it exists
                    if os.path.exists(file_path):
                        if os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        else:
                            os.remove(file_path)

                    target_download = "downloaded_data.nc"
                    data = read_era5_data_to_temporal(target_download)

                    assert data.shape == data_shape, f"Expected shape {data_shape}, got {data.shape}"

                    @measure_func
                    def measured_write():
                        writer.write(data)

                    # Run the write benchmark multiple times
                    write_times: List[float] = []
                    write_cpu_times: List[float] = []
                    write_memory_peak: List[float] = []
                    write_allocs: List[float] = []

                    for _ in range(iterations):
                        if clear_cache:
                            _clear_cache()

                        result = await measured_write()

                        if mode == MetricMode.MEMORY:
                            write_memory_peak.append(result.peak_memory)  # type: ignore
                            write_allocs.append(result.total_allocations)  # type: ignore
                        else:
                            write_times.append(result.elapsed)  # type: ignore
                            write_cpu_times.append(result.cpu_elapsed)  # type: ignore

                    file_size = writer.get_file_size()
                    del writer

                    # read the data again once to calculate information loss via MSE
                    control_data = await (await format.reader_class.create(file_path.__str__())).read(...)
                    data_mse = mean_squared_error_destructive(control_data, data)
                    del control_data
                    mse_cache.set(file_path.__str__(), data_mse)
                    gc.collect()
                    print(data_mse)

                    write_stats = BenchmarkStats(
                        mean=statistics.mean(write_times) if write_times else 0.0,
                        std=statistics.stdev(write_times) if len(write_times) > 1 else 0.0,
                        min=min(write_times) if write_times else 0.0,
                        max=max(write_times) if write_times else 0.0,
                        cpu_mean=statistics.mean(write_cpu_times) if write_cpu_times else 0.0,
                        cpu_std=statistics.stdev(write_cpu_times) if len(write_cpu_times) > 1 else 0.0,
                        memory_peak=statistics.mean(write_memory_peak) if len(write_memory_peak) > 0 else 0.0,
                        memory_total_allocated=statistics.mean(write_allocs) if len(write_allocs) > 0 else 0.0,
                        file_size=file_size,
                        samples=write_times if mode == MetricMode.TIME else write_memory_peak,
                    )
                    write_result = BenchmarkRecord.from_benchmark_stats(
                        stats=write_stats,
                        format=format,
                        config_id=config_hash,
                        operation=op_mode,
                        array_shape=data.shape,
                        read_index=None,
                        iterations=iterations,
                        data_mse=data_mse,
                    )
                    bm_results.append(write_result)

                elif op_mode == OpMode.READ:
                    # Read mode: error if file doesn't exist
                    if not os.path.exists(file_path):
                        raise FileNotFoundError(
                            f"File not found: {file_path}. Please run in write mode first to generate the file."
                        )

                    for read_length, indices in read_indices.items():
                        times: List[float] = []
                        cpu_times: List[float] = []
                        memory_peak: List[float] = []
                        memory_allocs: List[float] = []
                        file_size: int = 0
                        for i, read_index in enumerate(indices):
                            if clear_cache:
                                _clear_cache()
                            print(f"Reading file {file_path.__str__()} with format {format.name}")
                            reader_type = format.reader_class
                            reader = await reader_type.create(file_path.__str__())

                            @measure_func
                            async def measured_read():
                                return await reader.read(read_index)

                            try:
                                if mode == MetricMode.MEMORY:
                                    result = await measured_read()
                                    memory_peak.append(result.peak_memory)  # type: ignore
                                    memory_allocs.append(result.total_allocations)  # type: ignore
                                else:
                                    result = await measured_read()
                                    times.append(result.elapsed)  # type: ignore
                                    cpu_times.append(result.cpu_elapsed)  # type: ignore

                                file_size = reader.get_file_size()

                            except Exception as e:
                                raise e
                            finally:
                                reader.close()

                        read_stats = BenchmarkStats(
                            mean=statistics.mean(times) if times else 0.0,
                            std=statistics.stdev(times) if len(times) > 1 else 0.0,
                            min=min(times) if times else 0.0,
                            max=max(times) if times else 0.0,
                            cpu_mean=statistics.mean(cpu_times) if times else 0.0,
                            cpu_std=statistics.stdev(cpu_times) if len(cpu_times) > 1 else 0.0,
                            memory_peak=statistics.mean(memory_peak) if len(memory_peak) > 0 else 0.0,
                            memory_total_allocated=statistics.mean(memory_allocs) if len(memory_allocs) > 0 else 0.0,
                            samples=times if mode == MetricMode.TIME else memory_peak,
                            file_size=file_size,
                        )

                        result = BenchmarkRecord.from_benchmark_stats(
                            stats=read_stats,
                            format=format,
                            config_id=config_hash,
                            operation=op_mode,
                            array_shape=data_shape,
                            read_index=read_length,
                            iterations=iterations,
                            data_mse=mse_cache.get(file_path.__str__()),
                        )
                        bm_results.append(result)
                        gc.collect()

        chunk_size_str = "_".join(map(str, chunk_size))

        results_df = BenchmarkResultsDF(
            results_dir,
            base_file_name=f"benchmark_results_{chunk_size_str}_{op_mode.value}_{mode.value}",
        )
        if not plot_only:
            results_df.append(bm_results)
            results_df.save_results()
        else:
            results_df.load_last_results()

        results_df.print_latex_tabular()
        results_df.print_summary()

        plotting_df = results_df.prepare_for_plotting()

        # Common visualizations
        create_scatter_size_vs_mode(
            plotting_df.filter(pl.col("operation") == op_mode.value),
            op_mode,
            mode,
            plots_dir,
            file_name=f"scatter_size_vs_{mode.value}_{chunk_size_str}_{op_mode.value}.png",
        )
        create_violin_plot(
            plotting_df.filter(pl.col("operation") == op_mode.value),
            op_mode,
            mode,
            plots_dir,
            file_name=f"violin_plot_{chunk_size_str}_{op_mode.value}_{mode.value}.png",
        )

        gc.collect()


if __name__ == "__main__":
    app()
