import asyncio
import os
import platform
import subprocess
import tempfile
import time
from functools import partial, wraps
from typing import Any, Awaitable, Callable, Coroutine, NamedTuple, Union

# import memray


class TimeMeasurement(NamedTuple):
    elapsed: float
    cpu_elapsed: float


class MemoryMeasurement(NamedTuple):
    total_allocations: float
    peak_memory: float


# stolen from https://github.com/zarrs/zarr_benchmarks/blob/9679f36ca795cce65adc603ae41147324208d3d9/scripts/_run_benchmark.py#L5
def _clear_cache():
    if platform.system() == "Darwin":
        subprocess.call(["sync", "&&", "sudo", "purge"])
    elif platform.system() == "Linux":
        subprocess.call(["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"])
    else:
        raise Exception("Unsupported platform")


# https://stackoverflow.com/questions/72382098/how-to-check-if-a-callable-object-is-async-using-inspect-module-python
def is_async_callable(obj: Any) -> bool:
    while isinstance(obj, partial):
        obj = obj.func

    return asyncio.iscoroutinefunction(obj) or (callable(obj) and asyncio.iscoroutinefunction(obj.__call__))


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


def measure_memory(
    func: Callable[[], Union[Coroutine[Any, Any, Any], None]],
) -> Callable[..., Awaitable[MemoryMeasurement]]:
    @wraps(func)
    async def wrapper() -> MemoryMeasurement:
        import memray

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.bin")
            with memray.Tracker(path, native_traces=True, trace_python_allocators=True, follow_fork=True):
                if is_async_callable(func):
                    await func()  # type: ignore
                else:
                    func()

            stats = memray._memray.compute_statistics(path)

            return MemoryMeasurement(stats.total_memory_allocated, stats.metadata.peak_memory)

    return wrapper
