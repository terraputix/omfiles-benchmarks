from pathlib import Path

import polars as pl

from om_benchmarks.helpers.constants import RESULTS_DIR
from om_benchmarks.helpers.schemas import BENCHMARK_SCHEMA, BenchmarkRecord


class BenchmarkResultsDF:
    schema = BENCHMARK_SCHEMA

    def __init__(
        self,
        results_dir: str | Path = RESULTS_DIR,
        all_runs_name: str = "benchmark_results_all.csv",
        current_run_name: str = "benchmark_results_last.csv",
    ):
        if not isinstance(results_dir, Path):
            results_dir = Path(results_dir)
        self.results_dir = results_dir
        self.results_dir.mkdir(exist_ok=True)
        self.all_runs_path = self.results_dir / all_runs_name
        self.current_run_path = self.results_dir / current_run_name

        self.df = pl.DataFrame(schema=self.schema)

    def append(self, records: list[BenchmarkRecord]) -> None:
        new_df = pl.concat([self.df, pl.DataFrame(records, schema=self.schema)])
        self.df = new_df

    def save_results(self) -> None:
        """Save benchmark results to CSV and return DataFrame for display"""
        # Save to last run CSV
        self.df.write_csv(self.current_run_path)
        print(f"Latest results saved to {self.current_run_path}")

        # Save to all runs CSV
        with open(self.all_runs_path, mode="a") as f:
            self.df.write_csv(f, include_header=False)

    def load_last_results(self):
        """Load last benchmark results"""
        if not self.current_run_path.exists():
            raise FileNotFoundError(f"File {self.current_run_path} does not exist")
        self.df = pl.read_csv(self.current_run_path)

    def print_summary(self) -> None:
        """Print benchmark summary to console"""

        summary_df = (
            self.df.with_columns(
                [
                    pl.col("mean_time").round(6).alias("mean_s"),
                    pl.col("std_time").round(6).alias("std_s"),
                    pl.col("min_time").round(6).alias("min_s"),
                    pl.col("max_time").round(6).alias("max_s"),
                    pl.col("cpu_mean_time").round(6).alias("cpu_s"),
                    (pl.col("memory_usage_bytes") / 1024).round(2).alias("memory_kb"),
                    (pl.col("file_size_bytes") / (1024 * 1024)).round(2).alias("size_mb"),
                    pl.col("data_mse").round(6).alias("mse"),
                ]
            )
            .select(
                [
                    "operation",
                    "format",
                    "read_index",
                    "compression",
                    "mean_s",
                    "std_s",
                    "min_s",
                    "max_s",
                    "cpu_s",
                    "memory_kb",
                    "size_mb",
                    "mse",
                ]
            )
            .sort(["operation", "mean_s"])
        )

        with pl.Config(tbl_cols=-1, tbl_rows=-1):  # display all columns and rows
            print("\n" + "=" * 80)
            print("CURRENT BENCHMARK RESULTS")
            print("=" * 80)
            print(summary_df)
