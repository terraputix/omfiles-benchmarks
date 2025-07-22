from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import polars as pl

from om_benchmarks.formats import AvailableFormats
from om_benchmarks.io.writer_configs import FormatWriterConfig
from om_benchmarks.modes import OpMode

BENCHMARK_SCHEMA = pl.Schema(
    {
        "operation": pl.Utf8,
        "format": pl.Utf8,
        "array_shape": pl.Utf8,
        "compression": pl.Utf8,
        "data_mse": pl.Float64,
        "chunk_shape": pl.Utf8,
        "read_index": pl.Utf8,
        "iterations": pl.Int64,
        "mean_time": pl.Float64,
        "std_time": pl.Float64,
        "min_time": pl.Float64,
        "max_time": pl.Float64,
        "cpu_mean_time": pl.Float64,
        "cpu_std_time": pl.Float64,
        "memory_usage_bytes": pl.Float64,
        "file_size_bytes": pl.Int64,
    }
)


@dataclass
class BenchmarkStats:
    mean: float
    std: float
    min: float
    max: float
    cpu_mean: float
    cpu_std: float
    memory_usage: float
    file_size: float = 0


@dataclass
class BenchmarkRecord:
    """Type-safe schema for benchmark CSV records"""

    operation: Literal["read", "write"]
    format: str
    array_shape: str  # serialized tuple as string
    compression: str
    data_mse: float
    chunk_shape: str  # serialized tuple as string
    read_index: str  # serialized tuple as string
    iterations: int
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    cpu_mean_time: float
    cpu_std_time: float
    memory_usage_bytes: float
    file_size_bytes: float

    @classmethod
    def from_benchmark_stats(
        cls,
        stats: BenchmarkStats,
        format: AvailableFormats,
        writer_config: FormatWriterConfig,
        operation: OpMode,
        array_shape: Tuple[int, ...],
        read_index: Optional[Tuple[int, ...]],
        iterations: int,
        data_mse: float,
    ) -> "BenchmarkRecord":
        """Convert BenchmarkStats to BenchmarkRecord"""
        return cls(
            operation=operation.value,
            format=format.name,
            array_shape=str(array_shape),
            compression=writer_config.compression_pretty_name,
            data_mse=data_mse,
            chunk_shape=str(writer_config.chunk_size),
            read_index=str(read_index) if read_index is not None else "None",
            iterations=iterations,
            mean_time=stats.mean,
            std_time=stats.std,
            min_time=stats.min,
            max_time=stats.max,
            cpu_mean_time=stats.cpu_mean,
            cpu_std_time=stats.cpu_std,
            memory_usage_bytes=stats.memory_usage,
            file_size_bytes=stats.file_size,
        )
