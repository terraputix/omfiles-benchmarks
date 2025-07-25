import asyncio
import concurrent.futures
import statistics
import time
from typing import List, Tuple

import hdf5plugin
import numcodecs
import typer

from om_benchmarks.formats import AvailableFormats
from om_benchmarks.io.writer_configs import (
    BaselineConfig,
    FormatWriterConfig,
    HDF5Config,
    # NetCDFConfig,
    OMConfig,
    ZarrConfig,
)
from om_benchmarks.plotting.concurrency_plots import plot_concurrency_scaling
from om_benchmarks.read_indices import random_indices_for_read_range
from om_benchmarks.script_utils import get_era5_path_for_config, get_script_dirs

app = typer.Typer()

CHUNK_SIZE = (5, 5, 744)

# Example: test these formats/configs
TEST_FORMATS: List[Tuple[AvailableFormats, FormatWriterConfig]] = [
    # numpy memmap as a baseline
    (AvailableFormats.Baseline, BaselineConfig(chunk_size=CHUNK_SIZE)),
    (
        AvailableFormats.HDF5,
        HDF5Config(
            chunk_size=CHUNK_SIZE,
            compression=hdf5plugin.Blosc(cname="lz4", clevel=4, shuffle=hdf5plugin.Blosc.SHUFFLE),
        ),
    ),
    # (
    #     AvailableFormats.ZarrPythonViaZarrsCodecs,
    #     ZarrConfig(
    #         chunk_size=CHUNK_SIZE,
    #         compressor=numcodecs.Blosc(cname="lz4", clevel=4, shuffle=numcodecs.Blosc.BITSHUFFLE),
    #     ),
    # ),
    (
        AvailableFormats.Zarr,
        ZarrConfig(
            chunk_size=CHUNK_SIZE,
            compressor=numcodecs.Blosc(cname="lz4", clevel=4, shuffle=numcodecs.Blosc.BITSHUFFLE),
        ),
    ),
    (
        AvailableFormats.ZarrTensorStore,
        ZarrConfig(
            chunk_size=CHUNK_SIZE,
            compressor=numcodecs.Blosc(cname="lz4", clevel=4, shuffle=numcodecs.Blosc.BITSHUFFLE),
        ),
    ),
    # NetCDF-Python will segfault when accessed concurrently: https://github.com/Unidata/netcdf4-python/issues/844
    # (AvailableFormats.NetCDF, NetCDFConfig(chunk_size=CHUNK_SIZE, compression="zlib", compression_level=3)),
    (
        AvailableFormats.OM,
        OMConfig(
            chunk_size=CHUNK_SIZE,
            compression="pfor_delta_2d",
            scale_factor=100,
            add_offset=0,
        ),
    ),
]

DATA_SHAPE = (721, 1440, 744)
READ_RANGE = (20, 20, 20)  # needs to access at least 4 chunks!
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def parallel_read_task(reader_class, file_path, read_index):
    reader = asyncio.run(reader_class.create(str(file_path)))
    t0 = time.perf_counter()
    _arr = asyncio.run(reader.read(read_index))
    t1 = time.perf_counter()
    reader.close()
    return t1 - t0


def run_parallel_reads(reader_class, file_path, num_parallel):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel) as executor:
        read_index = random_indices_for_read_range(READ_RANGE, DATA_SHAPE)
        futures = [
            executor.submit(parallel_read_task, reader_class, file_path, read_index) for _ in range(num_parallel)
        ]
        results = [f.result() for f in futures]
    return results


@app.command()
def main():
    _, plots_dir = get_script_dirs(__file__)

    # {formats: {concurrency: [latencies]}}
    results: dict[AvailableFormats, dict[int, list[float]]] = {}

    for format, config in TEST_FORMATS:
        format_results = results.get(format, {})
        file_path = get_era5_path_for_config(format, config)
        print(f"\nBenchmarking {format.name} scaling...")

        for concurrency in CONCURRENCY_LEVELS:
            latencies = run_parallel_reads(format.reader_class, file_path, concurrency)
            mean_latency = statistics.mean(latencies)
            throughput = concurrency / mean_latency if mean_latency > 0 else 0
            format_results[concurrency] = latencies
            print(f"Concurrency {concurrency}: mean latency {mean_latency:.4f}s, throughput {throughput:.2f} req/s")
        results[format] = format_results

    plot_concurrency_scaling(results, plots_dir)
    # plot_latencies(results, plots_dir)


if __name__ == "__main__":
    main()
