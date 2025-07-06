from typing import List, Tuple

from om_benchmarks.helpers.formats import AvailableFormats
from om_benchmarks.helpers.io.writer_configs import FormatWriterConfig
from om_benchmarks.helpers.schemas import BenchmarkRecord, RunMetadata
from om_benchmarks.helpers.script_utils import get_file_path_for_format
from om_benchmarks.helpers.stats import (
    measure_memory,
    measure_time,
    run_multiple_benchmarks,
)


async def bm_read_format(
    read_index,
    iterations,
    format: AvailableFormats,
    format_config: FormatWriterConfig,
    file: str,
    clear_cache: bool = False,
    print_read_data: bool = False,
) -> BenchmarkRecord:
    print(f"Reading file {file}")
    reader_type = format.reader_class
    reader = await reader_type.create(file)

    @measure_time
    async def time_read():
        return await reader.read(read_index)

    @measure_memory
    async def memory_read():
        return await reader.read(read_index)

    try:
        if print_read_data:
            sample_data = await reader.read(read_index)
            print(sample_data)
        read_stats = await run_multiple_benchmarks(
            time_measurement=time_read,
            memory_measurement=memory_read,
            time_iterations=iterations,
            memory_iterations=1,
        )
        metadata = RunMetadata(array_shape=reader.shape, format_config=format_config, iterations=iterations)
        benchmark_record = BenchmarkRecord.from_benchmark_stats(read_stats, format, "read", metadata)
        return benchmark_record

    except Exception as e:
        raise e
    finally:
        reader.close()


async def bm_read_all_formats(
    read_index,
    iterations,
    formats: List[Tuple[AvailableFormats, FormatWriterConfig]],
    plot_read_data: bool = False,
) -> list[BenchmarkRecord]:
    read_results: list[BenchmarkRecord] = []
    for format, config in formats:
        print(f"Benchmarking {format.name}...")
        file = get_file_path_for_format(format).__str__()
        results = await bm_read_format(read_index, iterations, format, config, file, plot_read_data)
        read_results.append(results)

    return read_results
