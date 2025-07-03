from typing import List, Tuple

from om_benchmarks.helpers.formats import AvailableFormats
from om_benchmarks.helpers.schemas import BenchmarkStats, RunMetadata
from om_benchmarks.helpers.script_utils import get_file_path_for_format
from om_benchmarks.helpers.stats import (
    measure_execution,
    run_multiple_benchmarks,
)

type BMResultsDict = dict[AvailableFormats, Tuple[BenchmarkStats, RunMetadata]]


async def bm_read_format(
    read_index,
    iterations,
    format: AvailableFormats,
    plot_read_data: bool = False,
    read_results: BMResultsDict = {},
):
    print(f"Benchmarking {format.name}...")
    reader_type = format.reader_class
    file = get_file_path_for_format(reader_type).__str__()
    print(f"Reading file {file}")
    reader = await reader_type.create(file)

    @measure_execution
    async def read():
        return await reader.read(read_index)

    try:
        if plot_read_data:
            sample_data = await reader.read(read_index)  # Get sample data for verification
            print(sample_data)
        read_stats = await run_multiple_benchmarks(read, iterations)
        read_results[format] = (
            read_stats,
            RunMetadata(array_shape=reader.shape, chunk_shape=reader.chunk_shape, iterations=iterations),
        )

    except Exception as e:
        print(f"Error with {format}: {e}")
    finally:
        reader.close()


async def bm_read_all_formats(
    read_index,
    iterations,
    formats: List[AvailableFormats],
    plot_read_data: bool = False,
) -> BMResultsDict:
    read_results: BMResultsDict = {}
    for format_name in formats:
        await bm_read_format(read_index, iterations, format_name, plot_read_data, read_results)

    return read_results
