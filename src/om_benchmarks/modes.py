from enum import Enum


def format_bytes(x: float, pos: int) -> str:
    """Format bytes into human readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if x < 1024.0:
            return f"{x:.1f}\\,{unit}"
        x /= 1024.0
    return f"{x:.1f}\\,TB"


def format_time(x: float, pos: int) -> str:
    """Format time into human readable format"""
    if x < 1:
        return f"{x * 1000:.1f}\\,ms"
    elif x < 60:
        return f"{x:.2f}\\,s"
    else:
        return f"{x / 60:.1f}\\,min"


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
            return format_bytes
        elif self == MetricMode.TIME:
            return format_time
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
