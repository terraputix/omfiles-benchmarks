from pathlib import Path
from typing import Dict, List

import polars as pl

from .schemas import BenchmarkRecord, BenchmarkStats, RunMetadata


class BenchmarkResultsManager:
    """Type-safe benchmark results manager"""

    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.csv_path = self.results_dir / "benchmark_results.csv"
        self.last_run_path = self.results_dir / "benchmark_results_last.csv"

    def save_and_display_results(
        self, write_results: Dict[str, BenchmarkStats], read_results: Dict[str, BenchmarkStats], metadata: RunMetadata
    ) -> pl.DataFrame:
        """Save benchmark results to CSV and return DataFrame for display"""

        records: List[BenchmarkRecord] = []

        # Process write results
        for format_name, stats in write_results.items():
            record = BenchmarkRecord.from_benchmark_stats(stats, format_name, "write", metadata)
            records.append(record)

        # Process read results
        for format_name, stats in read_results.items():
            record = BenchmarkRecord.from_benchmark_stats(stats, format_name, "read", metadata)
            records.append(record)

        # Convert to DataFrame
        df = pl.DataFrame([record.to_dict() for record in records])
        df.write_csv(self.last_run_path)
        print(f"Latest results saved to {self.last_run_path}")

        # Save to CSV
        self._append_to_csv(df)

        return df

    def _append_to_csv(self, df: pl.DataFrame) -> None:
        """Append DataFrame to CSV file"""
        if self.csv_path.exists():
            existing_df = pl.read_csv(self.csv_path)
            combined_df = pl.concat([existing_df, df])
        else:
            combined_df = df

        combined_df.write_csv(self.csv_path)
        print(f"Results saved to {self.csv_path}")

    def get_current_results_summary(self, df: pl.DataFrame) -> pl.DataFrame:
        """Get a nicely formatted summary of current results"""
        return (
            df.with_columns(
                [
                    pl.col("mean_time").round(6).alias("mean_s"),
                    pl.col("std_time").round(6).alias("std_s"),
                    pl.col("min_time").round(6).alias("min_s"),
                    pl.col("max_time").round(6).alias("max_s"),
                    pl.col("cpu_mean_time").round(6).alias("cpu_s"),
                    (pl.col("memory_usage_bytes") / 1024).round(2).alias("memory_kb"),
                    (pl.col("file_size_bytes") / (1024 * 1024)).round(2).alias("size_mb"),
                ]
            )
            .select(["operation", "format", "mean_s", "std_s", "min_s", "max_s", "cpu_s", "memory_kb", "size_mb"])
            .sort(["operation", "mean_s"])
        )

    def load_last_results(self) -> pl.DataFrame:
        """Load last benchmark results"""
        if not self.csv_path.exists():
            return pl.DataFrame()
        return pl.read_csv(self.last_run_path)

    def load_historical_data(self) -> pl.DataFrame:
        """Load all historical benchmark data"""
        if not self.csv_path.exists():
            return pl.DataFrame()
        return pl.read_csv(self.csv_path)

    def get_performance_comparison(self, df: pl.DataFrame) -> pl.DataFrame:
        """Get performance comparison between formats"""
        return (
            df.group_by(["operation", "format"])
            .agg(
                [
                    pl.col("mean_time").mean().alias("avg_mean_time"),
                    pl.col("mean_time").min().alias("best_time"),
                    pl.col("file_size_bytes").last().alias("file_size"),
                ]
            )
            .sort(["operation", "avg_mean_time"])
        )
