import asyncio
import concurrent.futures
import statistics
import time
from typing import List, Tuple

import hdf5plugin
import matplotlib.pyplot as plt
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
        HDF5Config(
            chunk_size=CHUNK_SIZE,
            compression=hdf5plugin.Blosc(cname="lz4", clevel=4, shuffle=hdf5plugin.Blosc.SHUFFLE),
        ),
    ),
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
    (
        AvailableFormats.ZarrPythonViaZarrsCodecs,
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel) as executor:
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
def main():
    for format, config in TEST_FORMATS:
        file_path = get_era5_path_for_config(format, config)
        print(f"\nBenchmarking {format.name} scaling...")

        results = []
        for concurrency in CONCURRENCY_LEVELS:
            latencies = run_parallel_reads(format.reader_class, file_path, concurrency)
            mean_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            throughput = concurrency / mean_latency if mean_latency > 0 else 0
            results.append(
                {
                    "concurrency": concurrency,
                    "mean_latency": mean_latency,
                    "median_latency": median_latency,
                    "std_latency": std_latency,
                    "throughput": throughput,
                    "latencies": latencies,
                }
            )
            print(f"Concurrency {concurrency}: mean latency {mean_latency:.4f}s, throughput {throughput:.2f} req/s")

        # Plot throughput and latency
        conc = [r["concurrency"] for r in results]
        thr = [r["throughput"] for r in results]
        mean_lat = [r["mean_latency"] for r in results]
        median_lat = [r["median_latency"] for r in results]

        plt.figure(figsize=(10, 5))
        ax = plt.subplot(1, 2, 1)
        plt.plot(conc, thr, marker="o")
        plt.xlabel("Concurrency")
        ax.set_yscale("log", base=2)
        ax.set_xscale("log", base=2)
        plt.ylabel("Throughput (req/s)")
        plt.title(f"{format.name} Throughput Scaling")

        ax = plt.subplot(1, 2, 2)
        plt.plot(conc, mean_lat, marker="o", label="Mean")
        plt.plot(conc, median_lat, marker="x", label="Median")
        plt.xlabel("Concurrency")
        plt.ylabel("Latency (s)")
        ax.set_yscale("log", base=2)
        ax.set_xscale("log", base=2)
        plt.title(f"{format.name} Latency Scaling")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{format.name}_scaling.png")
        # plt.show()

        # Optional: Latency distribution as boxplot
        plt.figure(figsize=(8, 5))
        plt.boxplot([r["latencies"] for r in results], positions=conc)
        plt.xlabel("Concurrency")
        plt.ylabel("Latency (s)")
        plt.title(f"{format.name} Latency Distribution")
        plt.savefig(f"{format.name}_latency_boxplot.png")
        # plt.show()


if __name__ == "__main__":
    main()
