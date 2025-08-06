# Run with GIL or without:
# PYTHON_GIL=0 uv run scripts/concurrent_read_benchmarks.py
# PYTHON_GIL=1 uv run scripts/concurrent_read_benchmarks.py
import asyncio
import concurrent.futures
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import List, Tuple, Type

import typer

from om_benchmarks.configurations import _BASELINE_CONFIG, _HDF5_BEST, _OM_BEST, _ZARR_BEST, register_config
from om_benchmarks.formats import AvailableFormats
from om_benchmarks.io.readers import BaseReader
from om_benchmarks.io.writer_configs import FormatWriterConfig
from om_benchmarks.plotting.concurrency_plots import plot_concurrency_scaling, plot_concurrency_violin
from om_benchmarks.read_indices import generate_read_indices_single_range
from om_benchmarks.script_utils import get_era5_path_for_hashed_config, get_script_dirs
from om_benchmarks.stats import _clear_cache

app = typer.Typer()

CHUNK_SIZE = (32, 32, 32)
DATA_SHAPE = (721, 1440, 744)
READ_RANGE = (15, 15, 100)  # needs to access at least 6 chunks!
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]

TEST_FORMAT_CONFIGS: List[Tuple[AvailableFormats, FormatWriterConfig]] = [
    # numpy memmap as a baseline
    (AvailableFormats.Baseline, replace(_BASELINE_CONFIG, chunk_size=CHUNK_SIZE)),
    # HDF5 can still not be used with freethreaded Python
    # https://github.com/h5py/h5py/issues/2475
    (AvailableFormats.HDF5, replace(_HDF5_BEST, chunk_size=CHUNK_SIZE)),
    (AvailableFormats.ZarrPythonViaZarrsCodecs, replace(_ZARR_BEST, chunk_size=CHUNK_SIZE)),
    (AvailableFormats.Zarr, replace(_ZARR_BEST, chunk_size=CHUNK_SIZE)),
    (AvailableFormats.ZarrTensorStore, replace(_ZARR_BEST, chunk_size=CHUNK_SIZE)),
    # NetCDF-Python will segfault when accessed concurrently: https://github.com/Unidata/netcdf4-python/issues/844
    # (AvailableFormats.NetCDF, replace(_NETCDF_BEST, chunk_size=CHUNK_SIZE)),
    (AvailableFormats.OM, replace(_OM_BEST, chunk_size=CHUNK_SIZE)),
]

TEST_FORMATS = [(format, register_config(config)) for format, config in TEST_FORMAT_CONFIGS]


def parallel_read_task(reader, read_index) -> float:
    t0 = time.perf_counter()
    _arr = asyncio.run(reader.read(read_index))
    t1 = time.perf_counter()
    return t1 - t0


def run_parallel_reads(
    reader_class: Type[BaseReader], file_path: Path, concurrency_level: int, min_iterations: int = 2000
) -> list[float]:
    latencies = []
    reader = asyncio.run(reader_class.create(str(file_path)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_level) as executor:
        while len(latencies) < min_iterations:
            # Each batch gives num_parallel samples
            read_indices = generate_read_indices_single_range(
                DATA_SHAPE, read_iterations=concurrency_level, read_range=READ_RANGE
            )
            futures = [executor.submit(parallel_read_task, reader, read_indices[i]) for i in range(concurrency_level)]
            batch_results = [f.result() for f in futures]
            latencies.extend(batch_results)

    reader.close()
    return latencies


@app.command()
def main():
    _, plots_dir = get_script_dirs(__file__)

    # {formats: {concurrency: ([latencies], total_time)}}
    results: dict[AvailableFormats, dict[int, tuple[list[float], float]]] = {}

    for format, config_hash in TEST_FORMATS:
        format_results = results.get(format, {})
        file_path = get_era5_path_for_hashed_config(format, CHUNK_SIZE, config_hash)
        print(f"\nBenchmarking {format.name} scaling...")

        for concurrency in CONCURRENCY_LEVELS:
            _clear_cache()

            t0 = time.perf_counter()
            latencies = run_parallel_reads(format.reader_class, file_path, concurrency)
            total_time = time.perf_counter() - t0
            mean_latency = statistics.mean(latencies)
            throughput = len(latencies) / total_time
            format_results[concurrency] = (latencies, total_time)
            print(f"Concurrency {concurrency}: mean latency {mean_latency:.4f}s, throughput {throughput:.2f} req/s")
        results[format] = format_results

    plot_concurrency_scaling(results, plots_dir)
    plot_concurrency_violin(results, plots_dir)


if __name__ == "__main__":
    main()
