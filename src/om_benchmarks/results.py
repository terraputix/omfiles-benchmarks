from pathlib import Path

import polars as pl
import seaborn as sns

from om_benchmarks.configurations import get_config_by_hash
from om_benchmarks.constants import RESULTS_DIR
from om_benchmarks.schemas import BENCHMARK_SCHEMA, BenchmarkRecord
from om_benchmarks.utils import _uncompressed_size_from_array_shape


class BenchmarkResultsDF:
    schema = BENCHMARK_SCHEMA

    def __init__(
        self,
        results_dir: str | Path = RESULTS_DIR,
        all_runs_name: str = "benchmark_results_all.parquet",
        current_run_name: str = "benchmark_results_last.parquet",
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
        """Save benchmark results to parquet"""
        # Save to last run parquet
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
        """Load last benchmark results"""
        if not self.current_run_path.exists():
            raise FileNotFoundError(f"File {self.current_run_path} does not exist")
        self.df = pl.read_parquet(self.current_run_path)

    def prepare_for_plotting(self) -> pl.DataFrame:
        config_list = [get_config_by_hash(c) for c in self.df["config_id"].to_list()]
        chunk_shape_list = [config.chunk_size for config in config_list]

        # Get normalized plot labels for color mapping
        normalized_labels = [config.normalized_plot_label for config in config_list]
        unique_labels = list(dict.fromkeys(normalized_labels))
        palette = sns.color_palette("colorblind", n_colors=len(unique_labels))
        color_map = dict(zip(unique_labels, palette))
        color_list = [color_map[label] for label in normalized_labels]

        df_prepared = self.df.with_columns(
            [
                pl.Series("config", config_list, dtype=pl.Object),
                pl.struct(["array_shape", "file_size_bytes"])
                .map_elements(
                    lambda row: _uncompressed_size_from_array_shape(row["array_shape"]) / row["file_size_bytes"],
                    return_dtype=pl.Float32,
                )
                .alias("compression_factor"),
                pl.Series("chunk_shape", chunk_shape_list),
                pl.Series("color", color_list),
            ]
        )
        return df_prepared

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
                    (pl.col("memory_peak_bytes") / 1024).round(2).alias("memory_kb"),
                    (pl.col("file_size_bytes") / (1024 * 1024)).round(2).alias("size_mb"),
                    pl.col("data_mse").round(6).alias("mse"),
                ]
            )
            .select(
                [
                    "operation",
                    "format",
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
            .sort(["operation", "mean_s"])
        )

        with pl.Config(tbl_cols=-1, tbl_rows=-1):  # display all columns and rows
            print("\n" + "=" * 80)
            print("CURRENT BENCHMARK RESULTS")
            print("=" * 80)
            print(summary_df)
