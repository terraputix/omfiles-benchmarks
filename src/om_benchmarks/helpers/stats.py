import asyncio
import os
import platform
import statistics
import subprocess
import tempfile
import time
from functools import partial, wraps
from typing import Any, Awaitable, Callable, Coroutine, List, NamedTuple, Union

import memray

from om_benchmarks.helpers.schemas import BenchmarkStats


class TimeMeasurement(NamedTuple):
    elapsed: float
    cpu_elapsed: float


class MemoryMeasurement(NamedTuple):
    memory_total_allocations: float


# class MeasurementResult(NamedTuple):
#     elapsed: float
#     cpu_elapsed: float
#     memory_delta: float


# def get_rss():
#     process = psutil.Process(os.getpid())
#     rss = process.memory_info().rss
#     return rss


# https://stackoverflow.com/questions/72382098/how-to-check-if-a-callable-object-is-async-using-inspect-module-python
def is_async_callable(obj: Any) -> bool:
    while isinstance(obj, partial):
        obj = obj.func

    return asyncio.iscoroutinefunction(obj) or (callable(obj) and asyncio.iscoroutinefunction(obj.__call__))


# # Note: Tracking memory usage in Python is sometimes tricky because Python does not
# # necessarily directly release memory back to the operating system when it is no longer
# # needed or garbage collected.
# # Therefore, we are spawning a subprocess to measure memory usage more accurately.
# def _subprocess_target(func, args, kwargs, queue):
#     gc.collect()
#     rss_before = get_rss()
#     start_time = time.perf_counter()
#     cpu_start_time = time.process_time()
#     # We need to handle coroutines when we are running them in a subprocess
#     if is_async_callable(func):
#         _result = asyncio.run(func(*args, **kwargs))
#     else:
#         _result = func(*args, **kwargs)
#     elapsed_time = time.perf_counter() - start_time
#     cpu_elapsed_time = time.process_time() - cpu_start_time
#     gc.collect()
#     rss_after = get_rss()
#     memory_delta = rss_after - rss_before
#     queue.put((elapsed_time, cpu_elapsed_time, memory_delta))


def measure_time(
    func: Callable[[], Union[Coroutine[Any, Any, Any], None]],
) -> Callable[..., Awaitable[TimeMeasurement]]:
    @wraps(func)
    async def wrapper() -> TimeMeasurement:
        start_time = time.perf_counter()
        cpu_start_time = time.process_time()
        if is_async_callable(func):
            await func()  # type: ignore
        else:
            func()
        elapsed_time = time.perf_counter() - start_time
        cpu_elapsed_time = time.process_time() - cpu_start_time
        return TimeMeasurement(elapsed_time, cpu_elapsed_time)

    return wrapper


def sum_all_allocations(records: List[memray.AllocationRecord]) -> int:
    return sum(record.size * record.n_allocations for record in records)


def measure_memory(
    func: Callable[[], Union[Coroutine[Any, Any, Any], None]],
) -> Callable[..., Awaitable[MemoryMeasurement]]:
    @wraps(func)
    async def wrapper() -> MemoryMeasurement:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.bin")
            with memray.Tracker(path, native_traces=True, follow_fork=True):
                if is_async_callable(func):
                    await func()  # type: ignore
                else:
                    func()

            total_memory = 0
            with memray.FileReader(path) as reader:
                all_allocation = reader.get_allocation_records()
                total_memory = sum_all_allocations(all_allocation)

            return MemoryMeasurement(total_memory)

    return wrapper


# T = TypeVar("T")

# def measure_execution(func: Callable[..., T]) -> Callable[..., Awaitable[MeasurementResult]]:
#     @wraps(func)
#     async def wrapper(*args, **kwargs) -> MeasurementResult:
#         ctx = multiprocessing.get_context("spawn")
#         queue = ctx.Queue()
#         p = ctx.Process(target=_subprocess_target, args=(func, args, kwargs, queue))
#         p.start()
#         p.join()
#         if not queue.empty():
#             elapsed_time, cpu_elapsed_time, memory_delta = queue.get()
#         else:
#             raise RuntimeError("Subprocess did not return results")

#         return MeasurementResult(
#             elapsed=elapsed_time,
#             cpu_elapsed=cpu_elapsed_time,
#             memory_delta=memory_delta,
#         )

#     return wrapper


# stolen from https://github.com/zarrs/zarr_benchmarks/blob/9679f36ca795cce65adc603ae41147324208d3d9/scripts/_run_benchmark.py#L5
def _clear_cache():
    if platform.system() == "Darwin":
        subprocess.call(["sync", "&&", "sudo", "purge"])
    elif platform.system() == "Linux":
        subprocess.call(["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"])
    else:
        raise Exception("Unsupported platform")


async def run_multiple_benchmarks(
    time_measurement: Callable[[], Awaitable[TimeMeasurement]],
    memory_measurement: Callable[[], Awaitable[MemoryMeasurement]],
    time_iterations: int = 5,
    memory_iterations: int = 1,
    clear_cache: bool = False,
) -> BenchmarkStats:
    times: List[float] = []
    cpu_times: List[float] = []
    memory_usages: List[float] = []

    for _ in range(time_iterations):
        if clear_cache:
            _clear_cache()
        result = await time_measurement()
        times.append(result.elapsed)
        cpu_times.append(result.cpu_elapsed)

    for _ in range(memory_iterations):
        result = await memory_measurement()
        memory_usages.append(result.memory_total_allocations)

    print(f"times: {times}")

    return BenchmarkStats(
        mean=statistics.mean(times),
        std=statistics.stdev(times) if len(times) > 1 else 0.0,
        min=min(times),
        max=max(times),
        cpu_mean=statistics.mean(cpu_times),
        cpu_std=statistics.stdev(cpu_times) if len(cpu_times) > 1 else 0.0,
        memory_usage=statistics.mean(memory_usages),
    )
