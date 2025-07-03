from typing import List, Tuple

from zarr.core.buffer import NDArrayLike

from om_benchmarks.helpers.schemas import BenchmarkStats, RunMetadata
from om_benchmarks.helpers.script_utils import get_file_path_for_format

from .formats import AvailableFormats
from .stats import (
    measure_execution,
    run_multiple_benchmarks,
)


async def bm_write_all_formats(
    chunk_size: tuple,
    metadata: RunMetadata,
    data: NDArrayLike,
    formats: List[AvailableFormats],
) -> dict[AvailableFormats, Tuple[BenchmarkStats, RunMetadata]]:
    write_results: dict[AvailableFormats, Tuple[BenchmarkStats, RunMetadata]] = {}
    for format_name in formats:
        print(f"Benchmarking {format_name.name}...")
        writer_type = format_name.writer_class
        file = get_file_path_for_format(writer_type).__str__()
        writer = writer_type(file)

        @measure_execution
        def write():
            writer.write(data, chunk_size)

        try:
            write_stats = await run_multiple_benchmarks(write, metadata.iterations)
            write_stats.file_size = writer.get_file_size()
            write_results[format_name] = (write_stats, metadata)
        except Exception as e:
            print(f"Error with {format_name}: {e}")

    return write_results
