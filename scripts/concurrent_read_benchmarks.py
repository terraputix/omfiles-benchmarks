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
    OMConfig,
    ZarrConfig,
)
from om_benchmarks.read_indices import random_indices_for_read_range
from om_benchmarks.script_utils import get_era5_path_for_config

app = typer.Typer()

CHUNK_SIZE = (5, 5, 744)

# Example: test these formats/configs
TEST_FORMATS: List[Tuple[AvailableFormats, FormatWriterConfig]] = [
    # numpy memmap as a baseline
    (AvailableFormats.Baseline, BaselineConfig(chunk_size=CHUNK_SIZE)),
    (
        AvailableFormats.HDF5,
        # https://hdfgroup.github.io/hdf5/develop/group___s_z_i_p.html#ga688fde8106225adf9e6ccd2a168dec74
        # https://hdfgroup.github.io/hdf5/develop/_h5_d__u_g.html#title6
        # 1st 'nn' stands for: H5_SZIP_NN_OPTION_MASK
        # 2nd 32 stands for: 32 pixels per block
        HDF5Config(chunk_size=CHUNK_SIZE, compression="szip", compression_opts=("nn", 32), scale_offset=2),
    ),
    (
        AvailableFormats.HDF5,
        HDF5Config(
            chunk_size=CHUNK_SIZE,
            compression=hdf5plugin.Blosc(cname="lz4", clevel=4, shuffle=hdf5plugin.Blosc.SHUFFLE),
        ),
    ),
    (
        AvailableFormats.Zarr,
        ZarrConfig(
            chunk_size=CHUNK_SIZE,
            compressor=numcodecs.Blosc(cname="lz4", clevel=6, shuffle=numcodecs.Blosc.BITSHUFFLE),
        ),
    ),
    (
        AvailableFormats.ZarrTensorStore,
        ZarrConfig(
            chunk_size=CHUNK_SIZE,
            compressor=numcodecs.Blosc(cname="lz4", clevel=6, shuffle=numcodecs.Blosc.BITSHUFFLE),
        ),
    ),
    (
        AvailableFormats.ZarrPythonViaZarrsCodecs,
        ZarrConfig(
            chunk_size=CHUNK_SIZE,
            compressor=numcodecs.Blosc(cname="lz4", clevel=6, shuffle=numcodecs.Blosc.BITSHUFFLE),
        ),
    ),
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


async def concurrent_read_task(reader, read_index):
    t0 = time.perf_counter()
    _arr = await reader.read(read_index)
    t1 = time.perf_counter()
    return t1 - t0


def parallel_read_task(reader_class, file_path, read_index):
    reader = asyncio.run(reader_class.create(str(file_path)))
    t0 = time.perf_counter()
    _arr = asyncio.run(reader.read(read_index))
    t1 = time.perf_counter()
    reader.close()
    return t1 - t0


def run_parallel_reads(reader_class, file_path, num_parallel):
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_parallel) as executor:
        read_index = random_indices_for_read_range(READ_RANGE, DATA_SHAPE)
        futures = [
            executor.submit(parallel_read_task, reader_class, file_path, read_index) for _ in range(num_parallel)
        ]
        results = [f.result() for f in futures]
    return results


async def run_concurrent_reads(reader_class, file_path, read_index, num_concurrent):
    # Launch N concurrent read tasks
    reader = await reader_class.create(str(file_path))
    tasks = [concurrent_read_task(reader, read_index) for _ in range(num_concurrent)]
    results = await asyncio.gather(*tasks)
    reader.close()
    return results  # List of latencies


@app.command()
def main(
    num_concurrent: int = typer.Option(200, help="Number of concurrent read requests"),
    iterations: int = typer.Option(10, help="Number of times to repeat the benchmark"),
):
    for format, config in TEST_FORMATS:
        file_path = get_era5_path_for_config(format, config)
        print(f"\nBenchmarking {format.name} with {num_concurrent} concurrent reads...")

        all_latencies = []
        for i in range(iterations):
            # print(f"Iteration {i + 1}/{iterations}")
            latencies = run_parallel_reads(format.reader_class, file_path, num_concurrent)
            all_latencies.extend(latencies)
            # print(f"Latencies: {latencies}")

        mean_latency = statistics.mean(all_latencies)
        std_latency = statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0.0
        print(f"Mean latency: {mean_latency:.4f}s, Std: {std_latency:.4f}s")
        print(f"Throughput: {num_concurrent / mean_latency:.2f} reads/sec (approx)")


if __name__ == "__main__":
    app()
