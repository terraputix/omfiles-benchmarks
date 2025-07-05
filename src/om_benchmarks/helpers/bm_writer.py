from typing import List

from zarr.core.buffer import NDArrayLike

from om_benchmarks.helpers.formats import AvailableFormats
from om_benchmarks.helpers.schemas import BenchmarkRecord, RunMetadata
from om_benchmarks.helpers.script_utils import get_file_path_for_format
from om_benchmarks.helpers.stats import measure_execution, run_multiple_benchmarks


async def bm_write_format(
    chunk_size: tuple,
    metadata: RunMetadata,
    format: AvailableFormats,
    file: str,
    data: NDArrayLike,
) -> BenchmarkRecord:
    print(f"Writing file {file}...")

    writer = format.writer_class(file)

    @measure_execution
    def write():
        writer.write(data, chunk_size)

    write_stats = await run_multiple_benchmarks(write, metadata.iterations)
    write_stats.file_size = writer.get_file_size()
    benchmark_record = BenchmarkRecord.from_benchmark_stats(write_stats, format, "write", metadata)
    return benchmark_record


async def bm_write_all_formats(
    chunk_size: tuple,
    metadata: RunMetadata,
    data: NDArrayLike,
    formats: List[AvailableFormats],
) -> list[BenchmarkRecord]:
    write_results: list[BenchmarkRecord] = []
    for format in formats:
        print(f"Benchmarking {format.name}...")
        file = get_file_path_for_format(format).__str__()
        result = await bm_write_format(chunk_size, metadata, format, file, data)
        write_results.append(result)

    return write_results
