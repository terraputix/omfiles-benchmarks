import gc
import multiprocessing
import os
import platform
import statistics
import subprocess
import time
from functools import wraps
from typing import Any, Callable, List, NamedTuple, TypeVar

import psutil

from .schemas import BenchmarkStats

T = TypeVar("T")


class MeasurementResult(NamedTuple):
    result: Any
    elapsed: float
    cpu_elapsed: float
    memory_delta: float


def get_rss():
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss
    return rss


# Note: Tracking memory usage in Python is sometimes tricky because Python does not
# necessarily directly release memory back to the operating system when it is no longer
# needed or garbage collected.
# Therefore, we are spawning a subprocess to measure memory usage more accurately.
def _subprocess_target(func, args, kwargs, queue):
    gc.collect()
    rss_before = get_rss()
    start_time = time.time()
    cpu_start_time = time.process_time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    cpu_elapsed_time = time.process_time() - cpu_start_time
    gc.collect()
    rss_after = get_rss()
    memory_delta = rss_after - rss_before
    queue.put((result, elapsed_time, cpu_elapsed_time, memory_delta))


def measure_execution(func: Callable[..., T]) -> Callable[..., MeasurementResult]:
    @wraps(func)
    def wrapper(*args, **kwargs) -> MeasurementResult:
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=_subprocess_target, args=(func, args, kwargs, queue))
        p.start()
        p.join()
        if not queue.empty():
            result, elapsed_time, cpu_elapsed_time, memory_delta = queue.get()
        else:
            raise RuntimeError("Subprocess did not return results")
        return MeasurementResult(
            result=result,
            elapsed=elapsed_time,
            cpu_elapsed=cpu_elapsed_time,
            memory_delta=memory_delta,
        )

    return wrapper


# stolen from https://github.com/zarrs/zarr_benchmarks/blob/9679f36ca795cce65adc603ae41147324208d3d9/scripts/_run_benchmark.py#L5
def clear_cache():
    if platform.system() == "Darwin":
        subprocess.call(["sync", "&&", "sudo", "purge"])
    elif platform.system() == "Linux":
        subprocess.call(["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"])
    else:
        raise Exception("Unsupported platform")


def run_multiple_benchmarks(func: Callable[..., MeasurementResult], iterations: int = 5) -> BenchmarkStats:
    times: List[float] = []
    cpu_times: List[float] = []
    memory_usages: List[float] = []

    for _ in range(iterations):
        result = func()
        times.append(result.elapsed)
        cpu_times.append(result.cpu_elapsed)
        memory_usages.append(result.memory_delta)

    return BenchmarkStats(
        mean=statistics.mean(times),
        std=statistics.stdev(times) if len(times) > 1 else 0.0,
        min=min(times),
        max=max(times),
        cpu_mean=statistics.mean(cpu_times),
        cpu_std=statistics.stdev(cpu_times) if len(cpu_times) > 1 else 0.0,
        memory_usage=statistics.mean(memory_usages),
    )
