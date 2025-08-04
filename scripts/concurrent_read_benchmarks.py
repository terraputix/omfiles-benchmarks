import asyncio
import concurrent.futures
import statistics
import time
from typing import List, Tuple

import typer

from om_benchmarks.configurations import _BASELINE_CONFIG, _HDF5_BEST, _OM_BEST, _ZARR_BEST, register_config
from om_benchmarks.formats import AvailableFormats
from om_benchmarks.io.writer_configs import (
    FormatWriterConfig,
)
from om_benchmarks.plotting.concurrency_plots import plot_concurrency_scaling, plot_concurrency_violin
from om_benchmarks.read_indices import random_indices_for_read_range
from om_benchmarks.script_utils import get_era5_path_for_hashed_config, get_script_dirs
from om_benchmarks.stats import _clear_cache

app = typer.Typer()

CHUNK_SIZE = (5, 5, 744)

# Example: test these formats/configs
TEST_FORMAT_CONFIGS: List[Tuple[AvailableFormats, FormatWriterConfig]] = [
    # numpy memmap as a baseline
    (AvailableFormats.Baseline, _BASELINE_CONFIG),
    (AvailableFormats.HDF5, _HDF5_BEST),
    (AvailableFormats.ZarrPythonViaZarrsCodecs, _ZARR_BEST),
    (AvailableFormats.Zarr, _ZARR_BEST),
    (AvailableFormats.ZarrTensorStore, _ZARR_BEST),
    # NetCDF-Python will segfault when accessed concurrently: https://github.com/Unidata/netcdf4-python/issues/844
    # (AvailableFormats.NetCDF, _NETCDF_BEST),
    (AvailableFormats.OM, _OM_BEST),
]

TEST_FORMATS = [(format, register_config(config)) for format, config in TEST_FORMAT_CONFIGS]

DATA_SHAPE = (721, 1440, 744)
READ_RANGE = (20, 20, 20)  # needs to access at least 4 chunks!
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def parallel_read_task(reader, read_index) -> float:
    t0 = time.perf_counter()
    _arr = asyncio.run(reader.read(read_index))
    t1 = time.perf_counter()
    return t1 - t0


def run_parallel_reads(reader_class, file_path, num_parallel, min_iterations=500) -> list[float]:
    latencies = []
    reader = asyncio.run(reader_class.create(str(file_path)))

    while len(latencies) < min_iterations:
        # Each batch gives num_parallel samples
        read_index = random_indices_for_read_range(READ_RANGE, DATA_SHAPE)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel) as executor:
            futures = [executor.submit(parallel_read_task, reader, read_index) for _ in range(num_parallel)]
            batch_results = [f.result() for f in futures]
            latencies.extend(batch_results)

    reader.close()
    return latencies


@app.command()
def main():
    _, plots_dir = get_script_dirs(__file__)

    # {formats: {concurrency: [latencies]}}
    results: dict[AvailableFormats, dict[int, list[float]]] = {}

    for format, config_hash in TEST_FORMATS:
        _clear_cache()

        format_results = results.get(format, {})
        file_path = get_era5_path_for_hashed_config(format, CHUNK_SIZE, config_hash)
        print(f"\nBenchmarking {format.name} scaling...")

        for concurrency in CONCURRENCY_LEVELS:
            latencies = run_parallel_reads(format.reader_class, file_path, concurrency)
            mean_latency = statistics.mean(latencies)
            throughput = concurrency / mean_latency if mean_latency > 0 else 0
            format_results[concurrency] = latencies
            print(f"Concurrency {concurrency}: mean latency {mean_latency:.4f}s, throughput {throughput:.2f} req/s")
        results[format] = format_results

    plot_concurrency_scaling(results, plots_dir)
    plot_concurrency_violin(results, plots_dir)


if __name__ == "__main__":
    main()
