from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional, Tuple

import polars as pl

from om_benchmarks.helpers.formats import AvailableFormats

BENCHMARK_SCHEMA = pl.Schema(
    {
        "run_id": pl.Utf8,
        "timestamp": pl.Utf8,
        "operation": pl.Utf8,
        "format": pl.Utf8,
        "array_shape": pl.Utf8,
        "chunk_shape": pl.Utf8,
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

    run_id: str
    timestamp: str
    operation: Literal["read", "write"]
    format: str
    array_shape: str  # serialized tuple as string
    chunk_shape: str  # serialized tuple as string
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
        operation: Literal["read", "write"],
        run_metadata: "RunMetadata",
    ) -> "BenchmarkRecord":
        """Convert BenchmarkStats to BenchmarkRecord"""
        return cls(
            run_id=run_metadata.run_id,
            timestamp=run_metadata.timestamp,
            operation=operation,
            format=format.name,
            array_shape=run_metadata.array_shape_str,
            chunk_shape=run_metadata.chunk_shape_str,
            iterations=run_metadata.iterations,
            mean_time=stats.mean,
            std_time=stats.std,
            min_time=stats.min,
            max_time=stats.max,
            cpu_mean_time=stats.cpu_mean,
            cpu_std_time=stats.cpu_std,
            memory_usage_bytes=stats.memory_usage,
            file_size_bytes=stats.file_size,
        )


@dataclass
class RunMetadata:
    """Metadata for a benchmark run"""

    array_shape: Tuple[int, ...]
    chunk_shape: Optional[Tuple[int, ...]]
    iterations: int

    def __post_init__(self):
        """Generate derived fields after initialization"""
        self.timestamp = datetime.now().isoformat()
        self.array_shape_str = str(self.array_shape)
        self.chunk_shape_str = str(self.chunk_shape)
        self.run_id = f"{self.timestamp}_{self.array_shape_str}"
