from typing import Dict

from zarr.core.buffer import NDArrayLike

from om_benchmarks.helpers.formats import AvailableFormats
from om_benchmarks.helpers.io.writer_configs import FormatWriterConfig
from om_benchmarks.helpers.modes import OpMode
from om_benchmarks.helpers.schemas import BenchmarkRecord, RunMetadata
from om_benchmarks.helpers.script_utils import get_file_path_for_format
from om_benchmarks.helpers.stats import measure_memory, measure_time, run_multiple_benchmarks


async def bm_write_format(
    metadata: RunMetadata,
    format: AvailableFormats,
    config: FormatWriterConfig,
    file: str,
    data: NDArrayLike,
) -> BenchmarkRecord:
    print(f"Writing file {file}...")

    writer = format.writer_class(file, config)

    @measure_time
    def time_write():
        writer.write(data)

    @measure_memory
    def memory_write():
        writer.write(data)

    write_stats = await run_multiple_benchmarks(
        time_write,
        memory_write,
        time_iterations=metadata.iterations,
        memory_iterations=0,
    )
    write_stats.file_size = writer.get_file_size()
    benchmark_record = BenchmarkRecord.from_benchmark_stats(write_stats, format, OpMode.WRITE, metadata)
    return benchmark_record


async def bm_write_all_formats(
    metadata: RunMetadata,
    data: NDArrayLike,
    formats: Dict[AvailableFormats, FormatWriterConfig],
) -> list[BenchmarkRecord]:
    write_results: list[BenchmarkRecord] = []
    for format, config in formats.items():
        print(f"Benchmarking {format.name}...")
        file = get_file_path_for_format(format).__str__()
        result = await bm_write_format(metadata, format, config, file, data)
        write_results.append(result)

    return write_results
