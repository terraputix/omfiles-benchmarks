from typing import List, Tuple, Type

from om_benchmarks.helpers.formats import AvailableFormats
from om_benchmarks.helpers.io.readers import BaseReader
from om_benchmarks.helpers.schemas import BenchmarkStats, RunMetadata
from om_benchmarks.helpers.script_utils import get_file_path_for_format
from om_benchmarks.helpers.stats import (
    measure_execution,
    run_multiple_benchmarks,
)

type BMResult = Tuple[BenchmarkStats, RunMetadata]
type BMResultsDict = dict[AvailableFormats, BMResult]


async def bm_read_format(
    read_index,
    iterations,
    reader_type: Type[BaseReader],
    file: str,
    plot_read_data: bool = False,
) -> BMResult:
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
        return (
            read_stats,
            RunMetadata(array_shape=reader.shape, chunk_shape=reader.chunk_shape, iterations=iterations),
        )

    except Exception as e:
        raise e
    finally:
        reader.close()


async def bm_read_all_formats(
    read_index,
    iterations,
    formats: List[AvailableFormats],
    plot_read_data: bool = False,
) -> BMResultsDict:
    read_results: BMResultsDict = {}
    for format in formats:
        print(f"Benchmarking {format.name}...")
        reader_type = format.reader_class
        file = get_file_path_for_format(reader_type).__str__()
        results = await bm_read_format(read_index, iterations, reader_type, file, plot_read_data)
        read_results[format] = results

    return read_results
