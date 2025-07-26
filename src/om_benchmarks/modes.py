from enum import Enum

from om_benchmarks.plotting.formatters import BYTES_FORMATTER, TIME_FORMATTER


class OpMode(str, Enum):
    READ = "read"
    WRITE = "write"


class MetricMode(str, Enum):
    MEMORY = "memory"
    TIME = "time"

    @property
    def scatter_size_target_column(self) -> str:
        if self == MetricMode.MEMORY:
            return "memory_usage_bytes"
        elif self == MetricMode.TIME:
            return "mean_time"
        else:
            raise ValueError(f"Unknown metric mode: {self}")

    @property
    def target_values_formatter(self):
        if self == MetricMode.MEMORY:
            return BYTES_FORMATTER
        elif self == MetricMode.TIME:
            return TIME_FORMATTER
        else:
            raise ValueError(f"Unknown metric mode: {self}")

    @property
    def vs_title(self):
        if self == MetricMode.MEMORY:
            return "Memory Usage"
        elif self == MetricMode.TIME:
            return "Processing Time"
        else:
            raise ValueError(f"Unknown metric mode: {self}")

    @property
    def log_base(self):
        if self == MetricMode.MEMORY:
            return 2
        elif self == MetricMode.TIME:
            return 10
        else:
            raise ValueError(f"Unknown metric mode: {self}")

    @property
    def y_label(self):
        if self == MetricMode.MEMORY:
            return "Memory Usage"
        elif self == MetricMode.TIME:
            return "Processing Time"
        else:
            raise ValueError(f"Unknown metric mode: {self}")
