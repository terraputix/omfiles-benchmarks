import gc
import statistics
import time
import tracemalloc
from functools import wraps
from typing import Any, Callable, List, NamedTuple, TypeVar

from .schemas import BenchmarkStats

T = TypeVar("T")


class MeasurementResult(NamedTuple):
    result: Any
    elapsed: float
    cpu_elapsed: float
    memory_delta: float


def measure_execution(func: Callable[..., T]) -> Callable[..., MeasurementResult]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> MeasurementResult:
        # measure time
        start_time = time.time()
        cpu_start_time = time.process_time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        cpu_elapsed_time = time.process_time() - cpu_start_time

        # measure memory
        del result
        gc.collect()
        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()
        result = func(*args, **kwargs)
        end_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        memory_delta = sum(stat.size_diff for stat in end_snapshot.compare_to(start_snapshot, "lineno"))

        # fmt: off
        return MeasurementResult(
            result=result,
            elapsed=elapsed_time,
            cpu_elapsed=cpu_elapsed_time,
            memory_delta=memory_delta
        )

    return wrapper


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
