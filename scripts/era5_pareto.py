import os
import random
import statistics
from dataclasses import replace
from typing import List, Tuple

import hdf5plugin
import numcodecs
import numcodecs.zarr3
import typer

from om_benchmarks.helpers.AsyncTyper import AsyncTyper
from om_benchmarks.helpers.bm_writer import bm_write_format
from om_benchmarks.helpers.era5 import read_era5_data
from om_benchmarks.helpers.formats import AvailableFormats
from om_benchmarks.helpers.io.writer_configs import FormatWriterConfig, HDF5Config, NetCDFConfig, OMConfig, ZarrConfig
from om_benchmarks.helpers.plotting import (
    create_and_save_memory_usage_chart,
    create_and_save_perf_chart,
    create_scatter_size_vs_time,
    plot_radviz_results,
)
from om_benchmarks.helpers.results import BenchmarkResultsDF
from om_benchmarks.helpers.schemas import BenchmarkRecord, BenchmarkStats, RunMetadata
from om_benchmarks.helpers.script_utils import get_era5_path_for_config, get_script_dirs
from om_benchmarks.helpers.stats import _clear_cache, measure_memory, measure_time

app = AsyncTyper()

# read_indices = [
#     (100, 200, slice(0, 20, 1)),
#     (slice(100, 104), slice(200, 204), slice(0, 20, 1)),
#     (slice(100, 104), slice(200, 204), ...),
# ]

data_shape = (744, 721, 1440)

read_ranges: list[tuple[int, int, int]] = [
    (1, 1, 20),
    (5, 5, 200),
    (1, 1, 1440),
    (5, 5, 1440),
    (20, 20, 1440),
    (744, 721, 1),
]

chunk_sizes = {
    "small": (5, 5, 1440),
    "medium": (10, 10, 1440),
    "large": (20, 20, 1440),
}

READ_FORMATS: List[Tuple[AvailableFormats, FormatWriterConfig]] = [
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
            compression_opts=("nn", 32),
            explicitly_convert_to_int=True,
        ),
    ),
    (AvailableFormats.HDF5, HDF5Config(chunk_size=chunk_sizes["small"])),
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
    # (AvailableFormats.HDF5Hidefix, HDF5Config(chunk_size=chunk_sizes["small"])),
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
        OMConfig(chunk_size=chunk_sizes["small"], compression="fpx_xor_2d"),
    ),
]


@app.command()
async def main(
    read_iterations: int = typer.Option(2, help="Number of times to repeat each benchmark for more reliable results."),
    write_iterations: int = typer.Option(1, help="Number of times to repeat each benchmark for more reliable results."),
    clear_cache: bool = typer.Option(True, help="Clear the cache during single benchmark iterations."),
    plot_only: bool = typer.Option(False, help="Only plot the results without running the benchmarks."),
    skip_measure_memory: bool = typer.Option(True, help="Measure memory usage during benchmarking."),
):
    # Gather results
    results_dir, plots_dir = get_script_dirs(__file__)

    # Generate read_indices: for each read_range, generate read_iterations tuples of slices
    read_indices: dict[tuple[int, int, int], list[tuple[slice, slice, slice]]] = {}
    for read_range in read_ranges:
        slices: list[tuple[slice, slice, slice]] = []
        for _ in range(read_iterations):
            # For each dimension, pick a random start so that start + length <= dim_size
            starts = [random.randint(0, dim_size - req_len) for dim_size, req_len in zip(data_shape, read_range)]
            s0 = slice(starts[0], starts[0] + read_range[0])
            s1 = slice(starts[1], starts[1] + read_range[1])
            s2 = slice(starts[2], starts[2] + read_range[2])
            sliced: tuple[slice, slice, slice] = (s0, s1, s2)
            slices.append(sliced)
        read_indices[read_range] = slices

    for _name, chunk_size in chunk_sizes.items():
        read_results: list[BenchmarkRecord] = []
        write_results: list[BenchmarkRecord] = []
        if not plot_only:
            for format, _config in READ_FORMATS:
                config_for_this_run = replace(_config, chunk_size=chunk_size)
                file_path = get_era5_path_for_config(format, config=config_for_this_run)
                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}. Generating it ...")

                    target_download = "downloaded_data.nc"
                    data = read_era5_data(target_download)
                    assert data.shape == data_shape, f"Expected shape {data_shape}, got {data.shape}"

                    metadata = RunMetadata(
                        array_shape=data.shape,
                        read_index=None,
                        format_config=config_for_this_run,
                        iterations=write_iterations,
                    )
                    write_result = await bm_write_format(
                        metadata, format, config_for_this_run, file_path.__str__(), data
                    )
                    write_results.append(write_result)

                if clear_cache:
                    _clear_cache()

                for read_length, indices in read_indices.items():
                    times: List[float] = []
                    cpu_times: List[float] = []
                    memory_usages: List[float] = []
                    file_size: int = 0
                    for read_index in indices:
                        print(f"Reading file {file_path.__str__()}")
                        reader_type = format.reader_class
                        reader = await reader_type.create(file_path.__str__())

                        @measure_time
                        async def time_read():
                            return await reader.read(read_index)

                        @measure_memory
                        async def memory_read():
                            return await reader.read(read_index)

                        try:
                            result = await time_read()
                            times.append(result.elapsed)
                            cpu_times.append(result.cpu_elapsed)

                            if skip_measure_memory:
                                memory_usages.append(0)
                            else:
                                result = await memory_read()
                                memory_usages.append(result.memory_total_allocations)

                            file_size = reader.get_file_size()

                        except Exception as e:
                            raise e
                        finally:
                            reader.close()

                    read_stats = BenchmarkStats(
                        mean=statistics.mean(times),
                        std=statistics.stdev(times) if len(times) > 1 else 0.0,
                        min=min(times),
                        max=max(times),
                        cpu_mean=statistics.mean(cpu_times),
                        cpu_std=statistics.stdev(cpu_times) if len(cpu_times) > 1 else 0.0,
                        memory_usage=statistics.mean(memory_usages),
                    )
                    read_stats.file_size = file_size

                    metadata = RunMetadata(
                        array_shape=data_shape,
                        read_index=read_length,
                        format_config=config_for_this_run,
                        iterations=read_iterations,
                    )
                    result = BenchmarkRecord.from_benchmark_stats(read_stats, format, "read", metadata)
                    read_results.append(result)

        results_df = BenchmarkResultsDF(
            results_dir,
            all_runs_name="benchmark_results_all.csv",
            current_run_name=f"benchmark_results_{chunk_size}.csv",
        )
        if not plot_only:
            results_df.append(read_results)
            results_df.append(write_results)
            results_df.save_results()
        else:
            results_df.load_last_results()

        results_df.print_summary()

        # Create visualizations
        create_and_save_perf_chart(results_df.df, plots_dir, file_name=f"performance_chart_{chunk_size}.png")
        create_and_save_memory_usage_chart(results_df.df, plots_dir, file_name=f"memory_usage_chart_{chunk_size}.png")
        create_scatter_size_vs_time(results_df.df, plots_dir, file_name=f"scatter_size_vs_time_{chunk_size}.png")
        plot_radviz_results(results_df.df, plots_dir, file_name=f"radviz_results_{chunk_size}.png")


if __name__ == "__main__":
    app()
