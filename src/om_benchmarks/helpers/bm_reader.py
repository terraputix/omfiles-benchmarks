from typing import List

from om_benchmarks.helpers.formats import AvailableFormats
from om_benchmarks.helpers.schemas import BenchmarkRecord, RunMetadata
from om_benchmarks.helpers.script_utils import get_file_path_for_format
from om_benchmarks.helpers.stats import measure_execution, run_multiple_benchmarks


async def bm_read_format(
    read_index,
    iterations,
    format: AvailableFormats,
    file: str,
    plot_read_data: bool = False,
) -> BenchmarkRecord:
    print(f"Reading file {file}")
    reader_type = format.reader_class
    reader = await reader_type.create(file)

    @measure_execution
    async def read():
        return await reader.read(read_index)

    try:
        if plot_read_data:
            sample_data = await reader.read(read_index)  # Get sample data for verification
            print(sample_data)
        read_stats = await run_multiple_benchmarks(read, iterations)
        metadata = RunMetadata(array_shape=reader.shape, chunk_shape=reader.chunk_shape, iterations=iterations)
        benchmark_record = BenchmarkRecord.from_benchmark_stats(read_stats, format, "read", metadata)
        return benchmark_record

    except Exception as e:
        raise e
    finally:
        reader.close()


async def bm_read_all_formats(
    read_index,
    iterations,
    formats: List[AvailableFormats],
    plot_read_data: bool = False,
) -> list[BenchmarkRecord]:
    read_results: list[BenchmarkRecord] = []
    for format in formats:
        print(f"Benchmarking {format.name}...")
        file = get_file_path_for_format(format).__str__()
        results = await bm_read_format(read_index, iterations, format, file, plot_read_data)
        read_results.append(results)

    return read_results
