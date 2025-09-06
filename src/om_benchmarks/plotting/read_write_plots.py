from typing import Any, Tuple

import matplotlib.axes
import matplotlib.figure
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from numcodecs.zarr3 import math
from numpy import isin

from om_benchmarks.formats import AvailableFormats
from om_benchmarks.modes import MetricMode, OpMode
from om_benchmarks.plotting.params import _set_matplotlib_behaviour

_set_matplotlib_behaviour()

MSE_COLUMN = "data_mse"


def edgecolor_and_linewidth(row_dict):
    """
    Estimates edge color and line width for a given row
    dictionary based on the mse value (indicator for lossiness)
    """
    lossiness = row_dict.get(MSE_COLUMN, 0.0)
    if not lossiness > 0.0:
        return "black", 0.5

    mse = lossiness
    # Define scaling range for mse values
    min_mse = 0.00001
    max_mse = 10.0
    min_width = 0.1
    max_width = 6.0

    if mse <= min_mse:
        return "red", min_width

    if mse >= max_mse:
        return "red", max_width

    # Log-scale mapping preserving width range
    log_min = np.log10(min_mse)
    log_max = np.log10(max_mse)
    log_value = np.log10(mse)
    normalized = (log_value - log_min) / (log_max - log_min)
    width = min_width + normalized * (max_width - min_width)

    return "red", width


def _get_label(format: AvailableFormats, compression_label: str) -> str:
    return f"{format.plot_label} \n({compression_label})"


def get_marker_style(row_dict: dict[str, Any]):
    fmt = AvailableFormats(row_dict["format"])
    label = _get_label(fmt, row_dict["compression_label"])
    face_color = row_dict["color"]
    if isinstance(face_color, list) and len(face_color) >= 3:
        face_color = [*face_color, 0.6]  # Set alpha to 0.6
    edgecolor, linewidth = edgecolor_and_linewidth(row_dict)
    return {
        "marker": fmt.scatter_plot_marker,
        "markerfacecolor": face_color,
        "markeredgecolor": edgecolor,
        "markeredgewidth": linewidth,
        "markersize": 8,
        "label": label,
    }


def get_5_4_subplot_grid(n_rows: int, n_cols: int) -> Tuple[matplotlib.figure.Figure, np.ndarray]:
    fig_width = 5 * n_cols
    fig_height = 4 * n_rows
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=True,
    )
    plt.subplots_adjust(hspace=1.35)
    return fig, axes


def _assert_just_one_chunk_shape(df: pl.DataFrame) -> Any:
    chunk_shapes = df["chunk_shape"].unique().to_list()
    assert len(chunk_shapes) == 1, f"Expected 1 chunk shape, got {len(chunk_shapes)}"
    return chunk_shapes[0]


def create_scatter_size_vs_mode(
    df: pl.DataFrame,
    operation: OpMode,
    mode: MetricMode,
    save_dir,
    file_name="scatter_size_vs_mode.png",
) -> None:
    output_path = save_dir / file_name

    if df.height == 0:
        raise ValueError("No data in dataframe.")

    read_indices = df["read_index"].unique().sort().to_list()
    chunk_shape = _assert_just_one_chunk_shape(df)

    if len(read_indices) == 1:
        n_rows = 1
        n_cols = 1
    elif len(read_indices) % 2 == 0:
        n_rows: int = int(len(read_indices) / 2)
        n_cols = 2
    else:
        raise ValueError("Expected an even number of read indices.")

    fig_width = max(10, 5 * n_cols)
    fig_height = max(8, 4 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    plt.subplots_adjust(right=0.76)

    # plot subplots
    for row_idx in range(0, n_rows):
        for col_idx in range(0, n_cols):
            read_index = read_indices[row_idx * 2 + col_idx]
            ax: matplotlib.axes.Axes = axes[row_idx, col_idx]
            filtered_df = df.filter((pl.col("chunk_shape") == chunk_shape) & (pl.col("read_index") == read_index))
            if filtered_df.height == 0:
                ax.set_visible(False)
                continue

            # Plot each (format, compression) as a point
            for row_dict in filtered_df.iter_rows(named=True):
                marker_props = get_marker_style(row_dict)
                ax.scatter(
                    row_dict["compression_factor"],
                    row_dict[mode.scatter_size_target_column],
                    facecolor=marker_props["markerfacecolor"],
                    marker=marker_props["marker"],
                    s=80,
                    edgecolor=marker_props["markeredgecolor"],
                    linewidth=marker_props["markeredgewidth"],
                )

            ax.set_xlabel("Compression Factor")
            ax.set_ylabel(mode.y_label)
            ax.set_yscale("log", base=mode.log_base)
            # fix y-axis limits so at least two major ticks are visible
            y_min, y_max = ax.get_ylim()
            if y_max / y_min < 50:
                # round y_max to the nearest power of mode.log_base
                y_max = mode.log_base ** math.ceil(math.log(y_max, mode.log_base))
                ax.set_ylim(y_min, y_max)

            # ax.yaxis.set_major_formatter(mode.target_values_formatter)
            # ax.yaxis.set_minor_formatter(mode.target_values_formatter)
            ax.minorticks_on()
            # ax.grid(which="minor", alpha=0.2)
            if operation == OpMode.READ:
                ax.set_title(f"Random read of size {str(read_index)}")

    # Add legend
    handles = []
    df_for_legend = df.unique(subset=["config_id", "format"]).sort("format_order", "compression_label")
    for row_dict in df_for_legend.iter_rows(named=True):
        handles.append(
            matplotlib.lines.Line2D(
                [0],
                [0],
                color="w",
                **get_marker_style(row_dict=row_dict),
            )
        )
    fig.legend(handles=handles, loc="center right", frameon=True)

    title = f"Compression Factor vs. {mode.vs_title}"
    subtitle = "File Chunk Shape " + str(tuple(chunk_shape))
    fig.suptitle(rf"\Large {title}" + "\n" + rf"\normalsize {subtitle}")
    plt.savefig(output_path)
    plt.close()


def create_violin_plot(
    df: pl.DataFrame,
    operation: OpMode,
    mode: MetricMode,
    save_dir,
    file_name="violin_plot.png",
) -> None:
    output_path = save_dir / file_name

    if df.height == 0:
        raise ValueError("No data in dataframe.")

    read_indices = df["read_index"].unique().sort().to_list()
    chunk_shape = _assert_just_one_chunk_shape(df)

    if len(read_indices) == 1:
        n_rows = 1
        n_cols = 1
    elif len(read_indices) % 2 == 0:
        n_rows: int = int(len(read_indices) / 2)
        n_cols = 2
    else:
        raise ValueError("Expected an even number of read indices.")

    fig_width = max(12, 6 * n_cols)
    fig_height = max(30, 6 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    plt.subplots_adjust(wspace=0.50, hspace=0.2, top=0.95)

    def flatten_samples(filtered_df):
        rows = []
        for row_dict in filtered_df.iter_rows(named=True):
            samples = row_dict.get("samples", [])
            for sample in samples:
                rows.append(
                    {
                        "format": row_dict["format"],
                        "label": _get_label(AvailableFormats(row_dict["format"]), row_dict["compression_label"]),
                        "chunk_shape": row_dict["chunk_shape"],
                        "read_index": row_dict["read_index"],
                        "sample": sample,
                    }
                )
        return rows

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            read_index = read_indices[row_idx * 2 + col_idx]
            ax: matplotlib.axes.Axes = axes[row_idx, col_idx]
            filtered_df = df.filter(pl.col("chunk_shape") == chunk_shape)
            filtered_df = filtered_df.filter(pl.col("read_index") == read_index)
            if filtered_df.height == 0:
                ax.set_visible(False)
                continue

            plot_rows = flatten_samples(filtered_df)
            if not plot_rows:
                ax.set_visible(False)
                continue

            import pandas as pd

            plot_df = pd.DataFrame(plot_rows)

            sns.violinplot(
                data=plot_df,
                y="label",
                x="sample",
                hue="format",
                ax=ax,
                orient="h",
                inner="quartile",
                linewidth=1,
                legend=False,  # disables legend
                log_scale=mode.log_base,
            )

            ax.set_xlabel(mode.y_label)
            ax.set_ylabel("Format (Compression)")
            ax.set_title(f"{operation.value.title()} - Read Index: {str(read_index)}")
            ax.xaxis.set_major_formatter(mode.target_values_formatter)
            # ax.set_xscale("log", base=mode.log_base)

    title = f"Distribution of {mode.vs_title} by Format"
    subtitle = "File Chunk Shape " + str(chunk_shape)
    fig.suptitle(rf"\Large {title}" + "\n" + rf"\normalsize {subtitle}")
    plt.savefig(output_path)
    plt.close()
