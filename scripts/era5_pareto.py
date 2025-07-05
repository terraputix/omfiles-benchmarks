import os

import hdf5plugin
import numcodecs
import typer

from om_benchmarks.helpers.AsyncTyper import AsyncTyper
from om_benchmarks.helpers.bm_reader import bm_read_format
from om_benchmarks.helpers.bm_writer import bm_write_format
from om_benchmarks.helpers.era5 import read_era5_data
from om_benchmarks.helpers.formats import AvailableFormats
from om_benchmarks.helpers.io.writer_configs import FormatWriterConfig, HDF5Config, NetCDFConfig, OMConfig, ZarrConfig
from om_benchmarks.helpers.plotting import (
    create_and_save_memory_usage_chart,
    create_and_save_perf_chart,
)
from om_benchmarks.helpers.results import BenchmarkResultsDF
from om_benchmarks.helpers.schemas import BenchmarkRecord, RunMetadata
from om_benchmarks.helpers.script_utils import get_era5_path_for_config, get_script_dirs

app = AsyncTyper()

read_indices = [
    (100, 200, slice(0, 20, 1)),
    (slice(100, 104), slice(200, 204), slice(0, 20, 1)),
    (slice(100, 104), slice(200, 204), ...),
]

chunk_sizes = {
    "small": (5, 5, 1440),
    "medium": (10, 10, 1440),
    "large": (20, 20, 1440),
}

READ_FORMATS: dict[AvailableFormats, FormatWriterConfig] = {
    AvailableFormats.HDF5: HDF5Config(chunk_size=chunk_sizes["small"]),
    AvailableFormats.HDF5: HDF5Config(
        chunk_size=chunk_sizes["small"],
        compression=hdf5plugin.Blosc(cname="zstd", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE),
    ),
    AvailableFormats.HDF5Hidefix: HDF5Config(chunk_size=chunk_sizes["small"]),
    AvailableFormats.Zarr: ZarrConfig(
        chunk_size=chunk_sizes["small"],
        compressor=numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE),
    ),
    AvailableFormats.Zarr: ZarrConfig(
        chunk_size=chunk_sizes["small"],
        compressor=numcodecs.Blosc(),
    ),
    # AvailableFormats.ZarrTensorStore: ZarrConfig(
    #     chunk_size=chunk_sizes["small"],
    #     compressor=numcodecs.Blosc(),
    # ),
    AvailableFormats.ZarrPythonViaZarrsCodecs: ZarrConfig(
        chunk_size=chunk_sizes["small"],
        compressor=numcodecs.Blosc(),
    ),
    AvailableFormats.NetCDF: NetCDFConfig(chunk_size=chunk_sizes["small"], compression="zlib", compression_level=3),
    AvailableFormats.OM: OMConfig(
        chunk_size=chunk_sizes["small"],
        compression="pfor_delta_2d",
        scale_factor=100,
        add_offset=0,
    ),
}


@app.command()
async def main(
    iterations: int = typer.Option(10, help="Number of times to repeat each benchmark for more reliable results."),
):
    # Gather results
    results_dir, plots_dir = get_script_dirs(__file__)
    read_results: list[BenchmarkRecord] = []
    write_results: list[BenchmarkRecord] = []

    for format, config in READ_FORMATS.items():
        file_path = get_era5_path_for_config(format, config=config)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}. Generating it ...")

            target_download = "downloaded_data.nc"
            data = read_era5_data(target_download)

            metadata = RunMetadata(
                array_shape=data.shape,
                format_config=config,
                iterations=1,
            )
            write_result = await bm_write_format(metadata, format, config, file_path.__str__(), data)
            write_results.append(write_result)

        for read_index in read_indices:
            result = await bm_read_format(
                read_index,
                iterations,
                format,
                config,
                file_path.__str__(),
                False,
            )
            read_results.append(result)

    results_df = BenchmarkResultsDF(results_dir)
    results_df.append(read_results)
    results_df.append(write_results)

    results_df.save_results()
    results_df.print_summary()

    # Create visualizations
    create_and_save_perf_chart(results_df.df, plots_dir)
    create_and_save_memory_usage_chart(results_df.df, plots_dir)


if __name__ == "__main__":
    app()
