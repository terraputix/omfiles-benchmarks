from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Tuple

from .stats import BenchmarkStats


@dataclass
class BenchmarkRecord:
    """Type-safe schema for benchmark CSV records"""

    run_id: str
    timestamp: str
    operation: str  # 'read' or 'write'
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
        cls, stats: BenchmarkStats, format_name: str, operation: str, run_metadata: "RunMetadata"
    ) -> "BenchmarkRecord":
        """Convert BenchmarkStats to BenchmarkRecord"""
        return cls(
            run_id=run_metadata.run_id,
            timestamp=run_metadata.timestamp,
            operation=operation,
            format=format_name,
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation"""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "operation": self.operation,
            "format": self.format,
            "array_shape": self.array_shape,
            "chunk_shape": self.chunk_shape,
            "iterations": self.iterations,
            "mean_time": self.mean_time,
            "std_time": self.std_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "cpu_mean_time": self.cpu_mean_time,
            "cpu_std_time": self.cpu_std_time,
            "memory_usage_bytes": self.memory_usage_bytes,
            "file_size_bytes": self.file_size_bytes,
        }


@dataclass
class RunMetadata:
    """Metadata for a benchmark run"""

    array_shape: Tuple[int, ...]
    chunk_shape: Tuple[int, ...]
    iterations: int
    read_index: Any = None

    def __post_init__(self):
        """Generate derived fields after initialization"""
        self.timestamp = datetime.now().isoformat()
        self.array_shape_str = str(self.array_shape)
        self.chunk_shape_str = str(self.chunk_shape)
        self.run_id = f"{self.timestamp}_{self.array_shape_str}"
