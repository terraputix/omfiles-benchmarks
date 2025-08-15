import re
import time
from pathlib import Path

import polars as pl
import seaborn as sns

from om_benchmarks.configurations import get_config_by_hash
from om_benchmarks.constants import RESULTS_DIR
from om_benchmarks.formats import AvailableFormats
from om_benchmarks.schemas import BENCHMARK_SCHEMA, BenchmarkRecord
from om_benchmarks.utils import _uncompressed_size_from_array_shape


class BenchmarkResultsDF:
    schema = BENCHMARK_SCHEMA

    def __init__(self, results_dir: str | Path = RESULTS_DIR, base_file_name: str = "benchmark_results"):
        if not isinstance(results_dir, Path):
            results_dir = Path(results_dir)
        self.results_dir = results_dir
        self.results_dir.mkdir(exist_ok=True)
        self.base_name = base_file_name
        self.all_runs_path = self.results_dir / f"{self.base_name}_all.parquet"
        # We don't set current_run_path until save because we version them with a timestamp
        self.current_run_path = None

        self.df = pl.DataFrame(schema=self.schema)

    def append(self, records: list[BenchmarkRecord]) -> None:
        new_df = pl.concat([self.df, pl.DataFrame(records, schema=self.schema)])
        self.df = new_df

    def save_results(self) -> None:
        """Save benchmark results to parquet"""
        # Save to last run parquet
        timestamp = int(time.time())
        self.current_run_path = self.results_dir / f"{self.base_name}_{timestamp}.parquet"
        self.df.write_parquet(self.current_run_path)
        print(f"Latest results saved to {self.current_run_path}")

        # Save to all runs parquet
        if self.all_runs_path.exists():
            all_df = pl.read_parquet(self.all_runs_path)
            all_df = pl.concat([all_df, self.df])
        else:
            all_df = self.df

        self.df.write_parquet(self.all_runs_path)
        all_df.write_parquet(self.all_runs_path)

    def load_last_results(self):
        """Load the latest benchmark results file matching the pattern"""
        pattern = re.compile(rf"{self.base_name}_(\d+)\.parquet")
        files = list(self.results_dir.glob(f"{self.base_name}_*.parquet"))
        # Extract timestamps and sort
        candidates = []
        for f in files:
            m = pattern.match(f.name)
            if m:
                candidates.append((int(m.group(1)), f))
        if not candidates:
            raise FileNotFoundError(f"No matching results files found for pattern {self.base_name}_*.parquet")
        # Get the file with the latest timestamp
        latest_file = max(candidates, key=lambda x: x[0])[1]
        self.current_run_path = latest_file
        self.df = pl.read_parquet(latest_file)
        print(f"Loaded latest results from {latest_file}")

    def prepare_for_plotting(self) -> pl.DataFrame:
        config_list = [get_config_by_hash(c) for c in self.df["config_id"].to_list()]
        chunk_shape_list = [config.chunk_size for config in config_list]

        # Get normalized plot labels for color mapping
        plot_labels = [config.plot_label for config in config_list]
        unique_labels = list(dict.fromkeys(plot_labels))
        palette = sns.color_palette("colorblind", n_colors=len(unique_labels))
        color_map = dict(zip(unique_labels, palette))
        color_list = [color_map[label] for label in plot_labels]

        format_order = {fmt.value: fmt.format_order for fmt in AvailableFormats}

        df_prepared = self.df.with_columns(
            [
                pl.Series("config", config_list, dtype=pl.Object),
                pl.Series("compression_label", plot_labels),
                pl.struct(["array_shape", "file_size_bytes"])
                .map_elements(
                    lambda row: _uncompressed_size_from_array_shape(row["array_shape"]) / row["file_size_bytes"],
                    return_dtype=pl.Float32,
                )
                .alias("compression_factor"),
                pl.Series("chunk_shape", chunk_shape_list),
                pl.Series("color", color_list),
                pl.col("format").replace(format_order).alias("format_order"),
            ]
        ).sort("format_order", "compression_label")

        return df_prepared

    def print_summary(self) -> None:
        """Print benchmark summary to console"""

        summary_df = (
            self.prepare_for_plotting()
            .with_columns(
                [
                    pl.col("mean_time").round(6).alias("mean_s"),
                    pl.col("std_time").round(6).alias("std_s"),
                    pl.col("min_time").round(6).alias("min_s"),
                    pl.col("max_time").round(6).alias("max_s"),
                    pl.col("cpu_mean_time").round(6).alias("cpu_s"),
                    (pl.col("memory_peak_bytes") / 1024).round(2).alias("memory_kb"),
                    (pl.col("file_size_bytes") / (1024 * 1024)).round(2).alias("size_mb"),
                    pl.col("data_mse").round(6).alias("mse"),
                ]
            )
            .select(
                [
                    "operation",
                    "format",
                    "compression_label",
                    "read_index",
                    "config_id",
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
        )

        with pl.Config(tbl_cols=-1, tbl_rows=-1):  # display all columns and rows
            print("\n" + "=" * 80)
            print("CURRENT BENCHMARK RESULTS")
            print("=" * 80)
            print(summary_df)

    def print_latex_tabular(self) -> None:
        """Print a LaTeX tabular of format, compression_label, size_mb, mse to stdout."""
        df = (
            self.prepare_for_plotting()
            .with_columns([(pl.col("file_size_bytes") / (1024 * 1024)).alias("size_mb")])
            .select(
                [
                    "format",
                    "compression_label",
                    "size_mb",
                    "data_mse",
                ]
            )
            .unique(maintain_order=True)
        )

        # Print LaTeX tabular header
        print("\\begin{tabular}{|l|l|r|r|}")
        print("\\toprule")
        print("\\textbf{Format} & \\textbf{Compression Label} & \\textbf{Size (MB)} & \\textbf{MSE} \\\\")
        print("\\midrule")

        # Print each row
        for row in df.iter_rows():
            # Escape LaTeX special characters in compression_label
            format_, label, size_mb, mse = row
            label = str(label).replace("_", "\\_")
            print(f"{format_} & {label} & {float(size_mb):.2f} & {float(mse):.6f} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
