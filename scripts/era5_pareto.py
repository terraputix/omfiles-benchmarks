import gc
import os
import random
import shutil
import statistics
from dataclasses import replace
from typing import List, Tuple

import hdf5plugin
import numcodecs
import numcodecs.zarr3
import polars as pl
import typer

from om_benchmarks.helpers.AsyncTyper import AsyncTyper
from om_benchmarks.helpers.era5 import read_era5_data_to_temporal
from om_benchmarks.helpers.formats import AvailableFormats
from om_benchmarks.helpers.io.writer_configs import FormatWriterConfig, HDF5Config, NetCDFConfig, OMConfig, ZarrConfig
from om_benchmarks.helpers.modes import MetricMode, OpMode
from om_benchmarks.helpers.plotting import (
    create_and_save_compression_ratio_chart,
    create_and_save_memory_usage_chart,
    create_and_save_perf_chart,
    create_scatter_size_vs_mode,
    plot_radviz_results,
)
from om_benchmarks.helpers.results import BenchmarkResultsDF
from om_benchmarks.helpers.schemas import BenchmarkRecord, BenchmarkStats, RunMetadata
from om_benchmarks.helpers.script_utils import get_era5_path_for_config, get_script_dirs
from om_benchmarks.helpers.stats import _clear_cache, measure_memory, measure_time

app = AsyncTyper()

data_shape = (721, 1440, 744)

read_ranges: list[tuple[int, int, int]] = [
    (1, 1, 20),
    (5, 5, 200),
    (1, 1, 744),
    (5, 5, 744),
    (20, 20, 744),
    (721, 1440, 1),
]

chunk_sizes = {
    "small": (5, 5, 744),
    "medium": (10, 10, 744),
    "large": (20, 20, 744),
    # "xtra_large": (40, 40, 744),
    # "xtra_xtra_large": (100, 100, 744),
}

READ_FORMATS: List[Tuple[AvailableFormats, FormatWriterConfig]] = [
    # netcdf baseline: no compression
    (AvailableFormats.NetCDF, NetCDFConfig(chunk_size=chunk_sizes["small"], compression=None)),
    # hdf5 baseline: no compression
    (AvailableFormats.HDF5, HDF5Config(chunk_size=chunk_sizes["small"])),
    # zarr baseline: no compression
    (AvailableFormats.Zarr, ZarrConfig(chunk_size=chunk_sizes["small"], compressor=None)),
    (
        AvailableFormats.NetCDF,
        NetCDFConfig(chunk_size=chunk_sizes["small"], compression="szip", significant_digits=2),
    ),
    (AvailableFormats.NetCDF, NetCDFConfig(chunk_size=chunk_sizes["small"], compression="szip", scale_factor=1.0)),
    (
        AvailableFormats.HDF5,
        # https://hdfgroup.github.io/hdf5/develop/group___s_z_i_p.html#ga688fde8106225adf9e6ccd2a168dec74
        # https://hdfgroup.github.io/hdf5/develop/_h5_d__u_g.html#title6
        # 1st 'nn' stands for: H5_SZIP_NN_OPTION_MASK
        # 2nd 32 stands for: 32 pixels per block
        HDF5Config(chunk_size=chunk_sizes["small"], compression="szip", compression_opts=("nn", 32), scale_offset=2),
    ),
    (
        AvailableFormats.HDF5,
        # https://hdfgroup.github.io/hdf5/develop/group___s_z_i_p.html#ga688fde8106225adf9e6ccd2a168dec74
        # https://hdfgroup.github.io/hdf5/develop/_h5_d__u_g.html#title6
        # 1st 'nn' stands for: H5_SZIP_NN_OPTION_MASK
        # 2nd 32 stands for: 32 pixels per block
        HDF5Config(
            chunk_size=chunk_sizes["small"],
            compression="szip",
            compression_opts=("nn", 8),
            explicitly_convert_to_int=True,
        ),
    ),
    (
        AvailableFormats.HDF5,
        HDF5Config(
            chunk_size=chunk_sizes["small"],
            compression=hdf5plugin.Blosc(cname="zstd", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE),
        ),
    ),
    (
        AvailableFormats.HDF5,
        HDF5Config(
            chunk_size=chunk_sizes["small"],
            compression=hdf5plugin.Blosc(cname="lz4", clevel=4, shuffle=hdf5plugin.Blosc.SHUFFLE),
        ),
    ),
    (
        AvailableFormats.HDF5,
        HDF5Config(
            chunk_size=chunk_sizes["small"],
            compression=hdf5plugin.SZ(absolute=0.01),
        ),
    ),
    (
        AvailableFormats.Zarr,
        ZarrConfig(
            chunk_size=chunk_sizes["small"],
            compressor=numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE),
        ),
    ),
    (
        AvailableFormats.Zarr,
        ZarrConfig(
            chunk_size=chunk_sizes["small"],
            compressor=numcodecs.Blosc(),
        ),
    ),
    (
        AvailableFormats.Zarr,
        ZarrConfig(
            zarr_format=3,
            chunk_size=chunk_sizes["small"],
            serializer=numcodecs.zarr3.PCodec(level=8, mode_spec="auto"),
            filter=numcodecs.zarr3.FixedScaleOffset(offset=0, scale=100, dtype="f4", astype="i4"),
        ),
    ),
    (
        AvailableFormats.Zarr,
        ZarrConfig(
            zarr_format=3,
            chunk_size=chunk_sizes["small"],
            serializer=numcodecs.zarr3.PCodec(),
        ),
    ),
    (
        AvailableFormats.ZarrTensorStore,
        ZarrConfig(
            chunk_size=chunk_sizes["small"],
            compressor=numcodecs.Blosc(),
        ),
    ),
    (
        AvailableFormats.ZarrPythonViaZarrsCodecs,
        ZarrConfig(
            chunk_size=chunk_sizes["small"],
            compressor=numcodecs.Blosc(),
        ),
    ),
    (
        AvailableFormats.Zarr,
        ZarrConfig(
            chunk_size=chunk_sizes["small"],
            compressor=numcodecs.Blosc(cname="lz4", clevel=6, shuffle=numcodecs.Blosc.BITSHUFFLE),
        ),
    ),
    (
        AvailableFormats.ZarrTensorStore,
        ZarrConfig(
            chunk_size=chunk_sizes["small"],
            compressor=numcodecs.Blosc(cname="lz4", clevel=6, shuffle=numcodecs.Blosc.BITSHUFFLE),
        ),
    ),
    (
        AvailableFormats.ZarrPythonViaZarrsCodecs,
        ZarrConfig(
            chunk_size=chunk_sizes["small"],
            compressor=numcodecs.Blosc(cname="lz4", clevel=6, shuffle=numcodecs.Blosc.BITSHUFFLE),
        ),
    ),
    (AvailableFormats.NetCDF, NetCDFConfig(chunk_size=chunk_sizes["small"], compression="zlib", compression_level=3)),
    (
        AvailableFormats.OM,
        OMConfig(
            chunk_size=chunk_sizes["small"],
            compression="pfor_delta_2d",
            scale_factor=100,
            add_offset=0,
        ),
    ),
    (
        AvailableFormats.OM,
        OMConfig(
            chunk_size=chunk_sizes["small"],
            compression="pfor_delta_2d",
            scale_factor=20,
            add_offset=0,
        ),
    ),
    (
        AvailableFormats.OM,
        OMConfig(chunk_size=chunk_sizes["small"], compression="fpx_xor_2d"),
    ),
]


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

    measure_func = measure_memory if mode == MetricMode.MEMORY else measure_time

    # Generate read_indices: for each read_range, generate read_iterations tuples of slices
    read_indices: dict[tuple[int, int, int], list[tuple[slice, slice, slice]]] = {}
    for read_range in read_ranges:
        slices: list[tuple[slice, slice, slice]] = []
        for _ in range(iterations):
            # For each dimension, pick a random start so that start + length <= dim_size
            starts = [random.randint(0, dim_size - req_len) for dim_size, req_len in zip(data_shape, read_range)]
            s0 = slice(starts[0], starts[0] + read_range[0])
            s1 = slice(starts[1], starts[1] + read_range[1])
            s2 = slice(starts[2], starts[2] + read_range[2])
            sliced: tuple[slice, slice, slice] = (s0, s1, s2)
            slices.append(sliced)
        read_indices[read_range] = slices

    for _, chunk_size in chunk_sizes.items():
        bm_results: list[BenchmarkRecord] = []
        if not plot_only:
            for format, _config in READ_FORMATS:
                config_for_this_run = replace(_config, chunk_size=chunk_size)
                file_path = get_era5_path_for_config(format, config=config_for_this_run)

                if op_mode == OpMode.WRITE:
                    try:
                        writer = format.writer_class(file_path.__str__(), config_for_this_run)
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
                    write_memory_usages: List[float] = []

                    for _ in range(iterations):
                        if clear_cache:
                            _clear_cache()

                        result = await measured_write()

                        if mode == MetricMode.MEMORY:
                            write_memory_usages.append(result.memory_total_allocations)  # type: ignore
                        else:
                            write_times.append(result.elapsed)  # type: ignore
                            write_cpu_times.append(result.cpu_elapsed)  # type: ignore

                    write_stats = BenchmarkStats(
                        mean=statistics.mean(write_times) if write_times else 0.0,
                        std=statistics.stdev(write_times) if len(write_times) > 1 else 0.0,
                        min=min(write_times) if write_times else 0.0,
                        max=max(write_times) if write_times else 0.0,
                        cpu_mean=statistics.mean(write_cpu_times) if write_cpu_times else 0.0,
                        cpu_std=statistics.stdev(write_cpu_times) if len(write_cpu_times) > 1 else 0.0,
                        memory_usage=statistics.mean(write_memory_usages) if len(write_memory_usages) > 0 else 0.0,
                        file_size=writer.get_file_size(),
                    )

                    metadata = RunMetadata(
                        array_shape=data.shape,
                        read_index=None,
                        format_config=config_for_this_run,
                        iterations=iterations,
                    )
                    write_result = BenchmarkRecord.from_benchmark_stats(write_stats, format, op_mode, metadata)
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
                        memory_usages: List[float] = []
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
                                    memory_usages.append(result.memory_total_allocations)  # type: ignore
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
                            memory_usage=statistics.mean(memory_usages) if len(memory_usages) > 0 else 0.0,
                            file_size=file_size,
                        )

                        metadata = RunMetadata(
                            array_shape=data_shape,
                            read_index=read_length,
                            format_config=config_for_this_run,
                            iterations=iterations,
                        )
                        result = BenchmarkRecord.from_benchmark_stats(read_stats, format, op_mode, metadata)
                        bm_results.append(result)
                        gc.collect()

        results_df = BenchmarkResultsDF(
            results_dir,
            all_runs_name="benchmark_results_all.csv",
            current_run_name=f"benchmark_results_{chunk_size}_{op_mode.value}.csv",
        )
        if not plot_only:
            results_df.append(bm_results)
            results_df.save_results()
        else:
            results_df.load_last_results()

        results_df.print_summary()

        # Create visualizations based on mode
        if mode == MetricMode.TIME:
            create_and_save_perf_chart(
                results_df.df.filter(pl.col("operation") == op_mode.value),
                plots_dir,
                file_name=f"performance_chart_{chunk_size}_{op_mode.value}_{mode.value}.png",
            )
        elif mode == MetricMode.MEMORY:
            create_and_save_memory_usage_chart(
                results_df.df.filter(pl.col("operation") == op_mode.value),
                plots_dir,
                file_name=f"memory_usage_chart_{chunk_size}_{op_mode.value}_{mode.value}.png",
            )

        # Common visualizations
        create_scatter_size_vs_mode(
            results_df.df.filter(pl.col("operation") == op_mode.value),
            op_mode,
            mode,
            plots_dir,
            file_name=f"scatter_size_vs_{mode.value}_{chunk_size}_{op_mode.value}.png",
        )
        plot_radviz_results(
            results_df.df.filter(pl.col("operation") == op_mode.value),
            plots_dir,
            file_name=f"radviz_results_{chunk_size}_{op_mode.value}_{mode.value}.png",
        )
        create_and_save_compression_ratio_chart(
            results_df.df.filter(pl.col("operation") == op_mode.value),
            plots_dir,
            file_name=f"compression_ratio_chart_{chunk_size}_{op_mode.value}_{mode.value}.png",
        )

        gc.collect()


if __name__ == "__main__":
    app()
