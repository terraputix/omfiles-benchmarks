import os
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from om_benchmarks.plotting.formatters import TIME_FORMATTER
from om_benchmarks.plotting.params import _set_matplotlib_behaviour

_set_matplotlib_behaviour()


def create_and_save_violin_plot(df: pl.DataFrame, plots_dir: Path, file_name: str = "violin_plot.png"):
    # Convert to pandas for seaborn compatibility
    pdf = df.to_pandas()

    # Explode the all_times column so each row is a single timing
    pdf = pdf.explode("all_times")
    pdf["all_times"] = pdf["all_times"].astype(float)

    plt.figure(figsize=(8, 6))
    sns.violinplot(
        x="implementation",
        y="all_times",
        data=pdf,
        density_norm="width",
        palette="Set2",
        inner="quartile",
        log_scale=10,
    )
    plt.xlabel("Language")
    plt.ylabel("Elapsed Time")
    plt.title("Benchmark Timing Distribution by Language")

    ax = plt.gca()
    ax.yaxis.set_major_formatter(TIME_FORMATTER)

    # add minor ticks if we have fewer than 2 major ticks
    ymin, ymax = ax.get_ylim()
    if ymax / ymin < 100:
        ax.yaxis.set_minor_formatter(TIME_FORMATTER)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, file_name))
    plt.close()
