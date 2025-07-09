from functools import reduce
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, cast

import matplotlib.axes
import matplotlib.figure
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter

from om_benchmarks.helpers.formats import AvailableFormats
from om_benchmarks.helpers.parse_tuple import pretty_read_index

# plotting backend does not need to be interactive
# this will otherwise cause problems when matplotlib objects are gc-ed
matplotlib.use("Agg")

# Configure matplotlib for better appearance
plt.style.use("seaborn-v0_8-whitegrid")  # Use seaborn style for better defaults
rcParams.update(
    {
        "text.usetex": True,  # Enable LaTeX rendering
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "axes.labelsize": 9,
        "axes.titlesize": 11,
        "figure.titlesize": 16,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


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


def normalize_compression(comp: Optional[str]) -> str:
    if comp is None or comp.lower() == "none" or comp == "":
        return "None"
    return comp


def _get_label(format: AvailableFormats, compression: Optional[str]) -> str:
    return f"{format.name} \n({normalize_compression(compression)})"


def get_marker_for_format(fmt: AvailableFormats) -> str:
    """Get marker symbol for a format."""
    marker_map: dict[AvailableFormats, str] = {
        AvailableFormats.HDF5: "o",
        AvailableFormats.HDF5Hidefix: "s",
        AvailableFormats.Zarr: "D",
        AvailableFormats.ZarrTensorStore: "^",
        AvailableFormats.ZarrPythonViaZarrsCodecs: "v",
        AvailableFormats.NetCDF: "P",
        AvailableFormats.OM: "*",
    }
    return marker_map[fmt]


def get_color_palette(categories: Sequence[str]) -> dict[str, tuple[float, float, float]]:
    unique = list(dict.fromkeys(categories))  # preserve order
    palette = sns.color_palette("colorblind", n_colors=len(unique))
    return dict(zip(unique, palette))


def get_subplot_grid(
    df: pl.DataFrame,
    operations: List[str],
) -> Tuple[matplotlib.figure.Figure, np.ndarray, List[str], List[str]]:
    chunk_shapes: list[str] = cast(list[str], df["chunk_shape"].unique().to_list())
    read_indices: list[str] = cast(list[str], df["read_index"].unique().to_list())
    n_rows = len(chunk_shapes) * len(read_indices)
    n_cols = len(operations)
    fig_width = max(8, 5 * n_cols)
    fig_height = max(8, 4 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    plt.subplots_adjust(wspace=0.35, hspace=1.35, top=0.82)
    return fig, axes, chunk_shapes, read_indices


def add_info_box(ax: matplotlib.axes.Axes, chunk_shape: Optional[str], read_index: Optional[str]) -> None:
    info_lines = []
    if chunk_shape is not None:
        info_lines.append(f"Chunk Size: {chunk_shape}")
    if read_index is not None:
        info_lines.append(f"Read Index: {pretty_read_index(read_index)}")
    info_text = "\n".join(info_lines)
    if info_text:
        ax.text(
            0.98,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
        )


def _uncompressed_size_from_array_shape(array_shape: str, bytes_per_element: int = 4) -> int:
    # Remove parentheses if present, then split and convert to int
    array_shape_tuple: tuple[int, ...] = tuple(int(x) for x in array_shape.strip("()").split(",") if x.strip())
    total_size_bytes = reduce(lambda x, y: x * y, array_shape_tuple) * bytes_per_element
    return total_size_bytes


def create_and_save_perf_chart(df: pl.DataFrame, save_dir: Path, file_name: str = "performance_chart.png") -> None:
    output_path = save_dir / file_name

    operations = df["operation"].unique().to_list()

    fig, axes, chunk_shapes, read_indices = get_subplot_grid(df, operations)
    color_map = get_color_palette([normalize_compression(c) for c in df["compression"].to_list()])

    # For each subplot
    for row_idx, (chunk_shape, read_index) in enumerate([(cs, ri) for cs in chunk_shapes for ri in read_indices]):
        for col_idx, operation in enumerate(operations):
            ax: matplotlib.axes.Axes = axes[row_idx, col_idx]
            filtered_df = df.filter(pl.col("operation") == operation)
            filtered_df = filtered_df.filter(pl.col("chunk_shape") == chunk_shape)
            filtered_df = filtered_df.filter(pl.col("read_index") == read_index)
            filtered_df = filtered_df.sort("mean_time", descending=True)

            labels, mean_times, bar_colors = [], [], []
            for row in filtered_df.iter_rows():
                row_dict = dict(zip(filtered_df.columns, row))
                label = _get_label(AvailableFormats(row_dict["format"]), row_dict.get("compression"))
                labels.append(label)
                mean_times.append(row_dict["mean_time"])
                bar_colors.append(color_map[normalize_compression(row_dict.get("compression"))])

            ax.bar(labels, mean_times, color=bar_colors, edgecolor="white", linewidth=0.5)
            ax.set_xlabel("Format (Compression)")
            ax.set_ylabel("Mean Time")
            ax.yaxis.set_major_formatter(FuncFormatter(format_time))
            ax.tick_params(axis="x", rotation=60, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(True, alpha=0.3, axis="y")

            add_info_box(ax, chunk_shape, read_index)

    title = "Performance Comparison by Format"
    subtitle = "Lower execution time indicates better performance"
    fig.suptitle(rf"\Large {title}" + "\n" + rf"\normalsize {subtitle}", y=0.85)

    plt.savefig(output_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close()


def create_and_save_compression_ratio_chart(
    df: pl.DataFrame,
    save_dir: Path,
    file_name: str = "compression_ratio_comparison.png",
) -> None:
    output_path = save_dir / file_name
    filtered_df = df.filter(pl.col("file_size_bytes") > 0)

    if filtered_df.height == 0:
        print("No write data with file sizes > 0 found")
        return

    # add compression_ratio column
    filtered_df = filtered_df.with_columns(
        pl.struct(["array_shape", "file_size_bytes"])
        .map_elements(
            lambda row: _uncompressed_size_from_array_shape(row["array_shape"]) / row["file_size_bytes"],
            return_dtype=pl.Float32,
        )
        .alias("compression_ratio")
    )
    filtered_df = filtered_df.sort("compression_ratio", descending=True)

    labels = [
        _get_label(AvailableFormats(fmt), filtered_df["compression"][i])
        for i, fmt in enumerate(filtered_df["format"].to_list())
    ]
    compression_ratios = filtered_df["compression_ratio"].to_list()

    color_map = get_color_palette(labels)
    colors = [color_map[label] for label in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, compression_ratios, color=colors, edgecolor="white", linewidth=0.5)

    title = "Compression Ratio Comparison for different formats and compression schemas"
    ax.set_title(rf"\Large {title}", pad=25)
    ax.set_xlabel("Format (Compression)")
    ax.set_ylabel("Compression Ratio")
    ax.tick_params(axis="x", rotation=90, labelsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.savefig(output_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close()


def create_and_save_memory_usage_chart(df: pl.DataFrame, save_dir: Path, file_name: str = "memory_usage.png") -> None:
    output_path = save_dir / file_name

    operations = df["operation"].unique().to_list()
    fig, axes, chunk_shapes, read_indices = get_subplot_grid(df, operations)

    color_map = get_color_palette([normalize_compression(c) for c in df["compression"].to_list()])

    for row_idx, (chunk_shape, read_index) in enumerate([(cs, ri) for cs in chunk_shapes for ri in read_indices]):
        for col_idx, operation in enumerate(operations):
            ax: matplotlib.axes.Axes = axes[row_idx, col_idx]
            filtered_df = df.filter(pl.col("operation") == operation)
            filtered_df = filtered_df.filter(pl.col("chunk_shape") == chunk_shape)
            filtered_df = filtered_df.filter(pl.col("read_index") == read_index)
            filtered_df = filtered_df.sort("memory_usage_bytes", descending=True)

            labels, memory_usages, bar_colors = [], [], []
            for row in filtered_df.iter_rows():
                row_dict = dict(zip(filtered_df.columns, row))
                label = _get_label(AvailableFormats(row_dict["format"]), row_dict.get("compression"))
                labels.append(label)
                memory_usages.append(row_dict["memory_usage_bytes"])
                bar_colors.append(color_map[normalize_compression(row_dict.get("compression"))])

            ax.bar(labels, memory_usages, color=bar_colors, edgecolor="white", linewidth=0.5)
            ax.set_xlabel("Format (Compression)")
            ax.set_ylabel("Memory Usage (bytes)")
            ax.yaxis.set_major_formatter(FuncFormatter(format_bytes))
            ax.tick_params(axis="x", rotation=60, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(True, alpha=0.3, axis="y")

            add_info_box(ax, chunk_shape, read_index)

    title = "Memory Usage by Format and Operation"
    subtitle = "Lower memory usage indicates better efficiency"
    fig.suptitle(rf"\Large {title}" + "\n" + rf"\normalsize {subtitle}", y=0.85)

    plt.savefig(output_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close()


def create_scatter_size_vs_time(df: pl.DataFrame, save_dir, file_name="scatter_size_vs_time.png") -> None:
    output_path = save_dir / file_name

    # Only consider 'read' operation
    df = df.filter(pl.col("operation") == "read")
    if df.height == 0:
        raise ValueError("No read operation data found.")

    read_indices = df["read_index"].unique().sort().to_list() if "read_index" in df.columns else [None]
    chunk_shapes = df["chunk_shape"].unique().to_list() if "chunk_shape" in df.columns else [None]

    assert len(chunk_shapes) == 1, f"Expected 1 chunk shape, got {len(chunk_shapes)}"
    assert len(read_indices) % 2 == 0, f"Expected even number of read indices, got {len(read_indices)}"

    n_rows: int = int(len(read_indices) / 2)
    n_cols = 2

    fig_width = max(10, 5 * n_cols)
    fig_height = max(8, 4 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    plt.subplots_adjust(wspace=0.35, hspace=0.3, top=0.84)
    plt.subplots_adjust(right=0.78)

    # Color by compression, marker by format
    color_map = get_color_palette([normalize_compression(c) for c in df["compression"].to_list()])

    chunk_shape = chunk_shapes[0]
    # Build legend (only unique labels)
    handles = []
    seen = set()

    for row_idx in range(0, n_rows):
        for col_idx in range(0, n_cols):
            read_index = read_indices[row_idx * 2 + col_idx]
            ax: matplotlib.axes.Axes = axes[row_idx, col_idx]
            filtered_df = df
            filtered_df = filtered_df.filter(pl.col("chunk_shape") == chunk_shape)
            filtered_df = filtered_df.filter(pl.col("read_index") == read_index)
            if filtered_df.height == 0:
                ax.set_visible(False)
                continue

            # Plot each (format, compression) as a point
            for row in filtered_df.iter_rows():
                row_dict = dict(zip(filtered_df.columns, row))
                fmt: AvailableFormats = AvailableFormats(row_dict["format"])
                compression = normalize_compression(row_dict.get("compression"))
                color = color_map[compression]
                marker = get_marker_for_format(fmt)
                label = _get_label(fmt, compression)
                total_size_bytes = _uncompressed_size_from_array_shape(row_dict["array_shape"])

                ax.scatter(
                    total_size_bytes / row_dict["file_size_bytes"],
                    row_dict["mean_time"],
                    color=color,
                    marker=marker,
                    s=80,
                    edgecolor="black",
                    label=label,
                )

            ax.set_xlabel("Compression Ratio")
            ax.set_ylabel("Mean Read Time")
            # ax.set_xscale("log", base=2)
            # ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(format_time))
            # ax.yaxis.set_minor_formatter(FuncFormatter(format_time))
            ax.minorticks_on()
            ax.grid(True, alpha=0.3, axis="both")
            ax.grid(which="minor", alpha=0.2)
            ax.tick_params(axis="both", which="both", labelsize=8)
            ax.set_title(f"Random read of size {str(read_index)}")

            for row in filtered_df.iter_rows():
                row_dict = dict(zip(filtered_df.columns, row))
                fmt: AvailableFormats = AvailableFormats(row_dict["format"])
                compression = normalize_compression(row_dict.get("compression"))
                color = color_map[compression]
                marker = get_marker_for_format(fmt)
                label = _get_label(fmt, compression)
                if label not in seen:
                    handles.append(
                        matplotlib.lines.Line2D(
                            [0],
                            [0],
                            marker=marker,
                            color="w",
                            markerfacecolor=color,
                            markeredgecolor="black",
                            markersize=8,
                            label=label,
                        )
                    )
                    seen.add(label)
        fig.legend(handles=handles, loc="center right", fontsize=8, frameon=True)

    title = "File Size vs. Mean Read Time"
    subtitle = "File Chunk Shape " + str(chunk_shape)
    fig.suptitle(rf"\Large {title}" + "\n" + rf"\normalsize {subtitle}", y=0.92)
    plt.savefig(output_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_radviz_results(df: pl.DataFrame, save_dir, file_name="radviz_results.png") -> None:
    output_path = save_dir / file_name

    # Convert to pandas DataFrame
    pdf = df.to_pandas()

    # Only use numeric columns for radviz axes
    radviz_cols = ["file_size_bytes", "mean_time"]
    if "memory_usage_bytes" in pdf.columns:
        radviz_cols.append("memory_usage_bytes")

    # Remove rows with missing values in these columns
    pdf = pdf.dropna(subset=radviz_cols)

    for col in radviz_cols:
        max_val = pdf[col].max()
        min_val = pdf[col].min()
        if max_val > min_val:
            pdf[col] = (max_val - pdf[col]) / (max_val - min_val)
        else:
            pdf[col] = 0.0  # All values are the same

    # Add a label column for color grouping (format+compression)
    def label_row(row):
        return _get_label(AvailableFormats(row["format"]), row["compression"])

    pdf["label"] = pdf.apply(label_row, axis=1)

    # Get unique read_indices
    read_indices = pdf["read_index"].unique() if "read_index" in pdf.columns else [None]
    n = len(read_indices)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows), squeeze=False)
    plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.9)

    for idx, read_index in enumerate(read_indices):
        row = idx // ncols
        col = idx % ncols
        ax: matplotlib.axes.Axes = axes[row, col]

        if read_index is not None:
            sub_pdf = pdf[pdf["read_index"] == read_index]
        else:
            sub_pdf = pdf

        if sub_pdf.empty:
            ax.set_visible(False)
            continue

        # Only keep radviz columns and label
        sub_pdf_plot: pd.DataFrame = cast(pd.DataFrame, sub_pdf[radviz_cols + ["label"]].copy())

        pd.plotting.radviz(
            sub_pdf_plot,
            class_column="label",
            colormap="tab10",
            ax=ax,
        )
        ax.set_title(f"Read Index: {pretty_read_index(read_index)}" if read_index is not None else "All Read Indices")
        ax.legend(loc="upper right", bbox_to_anchor=(1.05, 1), fontsize=8, frameon=True)

    # Hide unused axes
    for idx in range(len(read_indices), nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    fig.suptitle("Radviz Visualization by Read Index", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close()
