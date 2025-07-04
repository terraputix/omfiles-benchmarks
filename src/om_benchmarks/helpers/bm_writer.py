from typing import List, Tuple, Type

from zarr.core.buffer import NDArrayLike

from om_benchmarks.helpers.bm_reader import BMResult
from om_benchmarks.helpers.io.writers import BaseWriter
from om_benchmarks.helpers.schemas import BenchmarkStats, RunMetadata
from om_benchmarks.helpers.script_utils import get_file_path_for_format

from .formats import AvailableFormats
from .stats import (
    measure_execution,
    run_multiple_benchmarks,
)


async def bm_write_format(
    chunk_size: tuple,
    metadata: RunMetadata,
    writer_type: Type[BaseWriter],
    file: str,
    data: NDArrayLike,
) -> BMResult:
    print(f"Writing file {file}...")

    writer = writer_type(file)

    @measure_execution
    def write():
        writer.write(data, chunk_size)

    write_stats = await run_multiple_benchmarks(write, metadata.iterations)
    write_stats.file_size = writer.get_file_size()
    return (write_stats, metadata)


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

        write_results[format_name] = await bm_write_format(chunk_size, metadata, writer_type, file, data)

    return write_results
