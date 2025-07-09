from functools import reduce
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, cast

import matplotlib.axes
import matplotlib.figure
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
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
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titlepad": 8,
        "figure.titlesize": 16,
        "axes.grid": True,
        "axes.grid.which": "both",
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 400,
        "savefig.facecolor": "white",  # optional, but recommended for consistency
        "figure.subplot.wspace": 0.35,
        "figure.subplot.hspace": 0.35,
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
    plt.subplots_adjust(hspace=1.35)
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


def add_compression_ratio_column(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.struct(["array_shape", "file_size_bytes"])
        .map_elements(
            lambda row: _uncompressed_size_from_array_shape(row["array_shape"]) / row["file_size_bytes"],
            return_dtype=pl.Float32,
        )
        .alias("compression_ratio")
    )
    return df


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
            ax.tick_params(axis="x", rotation=60)

            add_info_box(ax, chunk_shape, read_index)

    title = "Performance Comparison by Format"
    subtitle = "Lower execution time indicates better performance"

    fig.suptitle(rf"\Large {title}" + "\n" + rf"\normalsize {subtitle}", y=0.92)
    plt.savefig(output_path)
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
    fig.suptitle(rf"\Large {title}" + "\n" + rf"\normalsize {subtitle}", y=0.92)
    plt.savefig(output_path)
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
    filtered_df = add_compression_ratio_column(filtered_df)
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

    ax.set_xlabel("Format (Compression)")
    ax.set_ylabel("Compression Ratio")
    ax.tick_params(axis="x", rotation=90)

    title = "Compression Ratio Comparison for different formats and compression schemas"
    fig.suptitle(rf"\Large {title}")
    plt.savefig(output_path)
    plt.close()


def create_scatter_size_vs_time(df: pl.DataFrame, save_dir, file_name="scatter_size_vs_time.png") -> None:
    output_path = save_dir / file_name

    # Only consider 'read' operation
    df = df.filter(pl.col("operation") == "read")
    if df.height == 0:
        raise ValueError("No read operation data found.")

    read_indices = df["read_index"].unique().sort().to_list()
    chunk_shapes = df["chunk_shape"].unique().to_list()

    assert len(chunk_shapes) == 1, f"Expected 1 chunk shape, got {len(chunk_shapes)}"
    assert len(read_indices) % 2 == 0, f"Expected even number of read indices, got {len(read_indices)}"

    n_rows: int = int(len(read_indices) / 2)
    n_cols = 2

    fig_width = max(10, 5 * n_cols)
    fig_height = max(8, 4 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    plt.subplots_adjust(right=0.78)

    # Color by compression, marker by format
    color_map = get_color_palette([normalize_compression(c) for c in df["compression"].to_list()])

    chunk_shape = chunk_shapes[0]
    # --- Build legend handles ONCE for all unique (format, compression) ---
    unique_pairs = set()
    for row in df.iter_rows():
        fmt = AvailableFormats(row[df.columns.index("format")])
        compression = normalize_compression(row[df.columns.index("compression")])
        label = _get_label(fmt, compression)
        unique_pairs.add((label, fmt, compression))

    handles = []
    for label, fmt, compression in sorted(unique_pairs):
        color = color_map[compression]
        marker = get_marker_for_format(fmt)
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

    # plot subplots
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

            filtered_df = add_compression_ratio_column(filtered_df)

            # Plot each (format, compression) as a point
            for row in filtered_df.iter_rows():
                row_dict = dict(zip(filtered_df.columns, row))
                fmt: AvailableFormats = AvailableFormats(row_dict["format"])
                compression = normalize_compression(row_dict.get("compression"))
                color = color_map[compression]
                marker = get_marker_for_format(fmt)
                label = _get_label(fmt, compression)

                ax.scatter(
                    row_dict["compression_ratio"],
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
            # ax.grid(which="minor", alpha=0.2)
            ax.set_title(f"Random read of size {str(read_index)}")

    fig.legend(handles=handles, loc="center right", frameon=True)
    title = "File Size vs. Mean Read Time"
    subtitle = "File Chunk Shape " + str(chunk_shape)

    fig.suptitle(rf"\Large {title}" + "\n" + rf"\normalsize {subtitle}")
    plt.savefig(output_path)
    plt.close()


# https://github.com/pandas-dev/pandas/blob/d5f97ed21a872c2ea0bbe2a1de8b4242ec6a58d1/pandas/plotting/_matplotlib/misc.py#L136
def custom_radviz(
    df: pl.DataFrame,
    class_column: str,
    color_map: Dict[str, Tuple[float, float, float]],
    marker_map: Dict[str, str],
    ax: matplotlib.axes.Axes,
    value_columns: Optional[List[str]] = None,
    invert_columns: Optional[List[str]] = None,
    alpha: float = 0.8,
    s: float = 80,
) -> None:
    """Custom radviz implementation for polars DataFrames with built-in normalization."""
    # Determine which columns to use
    if value_columns is None:
        numeric_cols = [c for c in df.columns if c != class_column and df[c].dtype.is_numeric()]
    else:
        numeric_cols = value_columns

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found")

    # Work with a copy for normalization
    df_work = df.select(numeric_cols + [class_column])

    def invert_expr(expr: pl.Expr) -> pl.Expr:
        """Invert an expression: 1 - normalized_value"""
        return 0 - expr

    def normalize_expr(expr: pl.Expr) -> pl.Expr:
        col_min = expr.min()
        col_max = expr.max()

        return (
            pl.when(col_max > col_min)
            .then((expr - col_min) / (col_max - col_min))
            .otherwise(
                pl.lit(0.5)  # If all values are the same
            )
        )

    # Invert specified columns (so "lower is better" becomes "higher is better")
    if invert_columns:
        for col in invert_columns:
            if col in numeric_cols:
                df_work = df_work.with_columns(invert_expr(pl.col(col)).alias(col))

    # Normalize each column to 0-1 range
    for col in numeric_cols:
        df_work = df_work.with_columns(normalize_expr(pl.col(col)).alias(col))

    # Anchor points on the circle
    m = len(numeric_cols)
    anchor_points = np.array([(np.cos(t), np.sin(t)) for t in [2.0 * np.pi * (i / float(m)) for i in range(m)]])

    # Get data as numpy arrays
    numeric_data = df_work.select(numeric_cols).to_numpy()
    labels = df_work[class_column].to_list()

    # Compute positions using row maximum
    # Usual radviz would use row sum
    positions = []
    for row in numeric_data:
        row_max = np.max(row)
        if row_max == 0:
            pos = np.array([0.0, 0.0])
        else:
            # Normalize by maximum instead of sum
            pos = np.sum(anchor_points * row[:, np.newaxis], axis=0) / row_max
        positions.append(pos)

    # Plot each class with its color and marker
    unique_labels = list(dict.fromkeys(labels))
    for cls in unique_labels:
        if cls not in color_map or cls not in marker_map:
            continue

        class_indices = [i for i, label in enumerate(labels) if label == cls]
        if not class_indices:
            continue

        class_positions = np.array([positions[i] for i in class_indices])

        ax.scatter(
            class_positions[:, 0],
            class_positions[:, 1],
            color=color_map[cls],
            marker=marker_map[cls],
            s=s,
            edgecolor="black",
            alpha=alpha,
            linewidth=0.5,
        )

    # Draw the unit circle
    circle = matplotlib.patches.Circle((0, 0), 1, fill=False, color="black", linewidth=1)
    ax.add_patch(circle)

    # Draw anchor points and labels
    for point, col_name in zip(anchor_points, numeric_cols):
        # Draw anchor point
        ax.scatter(point[0], point[1], c="gray", s=50, zorder=10)

        # Position text labels depending on quadrant
        x, y = point[0], point[1]
        if x < 0.0 and y < 0.0:
            ax.text(
                x - 0.035,
                y - 0.035,
                col_name,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )
        elif x < 0.0 <= y:
            ax.text(
                x - 0.035,
                y + 0.035,
                col_name,
                ha="right",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )
        elif y < 0.0 <= x:
            ax.text(
                x + 0.035,
                y - 0.035,
                col_name,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )
        elif x >= 0.0 and y >= 0.0:
            ax.text(
                x + 0.035,
                y + 0.035,
                col_name,
                ha="left",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_radviz_results(df: pl.DataFrame, save_dir: Path, file_name: str = "radviz_results.png") -> None:
    output_path = save_dir / file_name

    # Add compression ratio if not already present
    df = add_compression_ratio_column(df)

    # Define which columns to use for radviz
    radviz_cols = ["compression_ratio", "mean_time", "memory_usage_bytes"]
    invert_cols = ["mean_time", "memory_usage_bytes"]

    # Remove rows with missing values in radviz columns
    mask = pl.col(radviz_cols[0]).is_not_null()
    for col in radviz_cols[1:]:
        mask = mask & pl.col(col).is_not_null()
    df = df.filter(mask)

    if df.height == 0:
        print("No data remaining after filtering null values")
        return

    # Build label to format mapping
    format_to_label_map = {}
    for row in df.select(["format", "compression"]).unique().iter_rows():
        fmt_enum = AvailableFormats(row[0])
        compression = row[1]
        label = _get_label(fmt_enum, compression)
        format_to_label_map[label] = fmt_enum

    # Add label column
    df = df.with_columns(
        pl.struct(["format", "compression"])
        .map_elements(lambda x: _get_label(AvailableFormats(x["format"]), x["compression"]), return_dtype=pl.Utf8)
        .alias("label")
    )

    # Setup subplots
    read_indices = sorted(df["read_index"].unique().to_list())
    n = len(read_indices)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows), squeeze=False)
    plt.subplots_adjust(right=0.84, hspace=0.25)

    # Build color and marker maps
    unique_labels = df["label"].unique().to_list()
    color_map = get_color_palette(unique_labels)
    marker_map = {label: get_marker_for_format(fmt) for label, fmt in format_to_label_map.items()}

    # Plot each subplot
    for idx, read_index in enumerate(read_indices):
        row_idx = idx // ncols
        col_idx = idx % ncols
        ax: matplotlib.axes.Axes = axes[row_idx, col_idx]

        sub_df = df.filter(pl.col("read_index") == read_index)
        if sub_df.height == 0:
            ax.set_visible(False)
            continue

        # Create radviz plot with normalization handled internally
        custom_radviz(
            sub_df,
            class_column="label",
            value_columns=radviz_cols,
            invert_columns=invert_cols,
            color_map=color_map,
            marker_map=marker_map,
            ax=ax,
        )
        ax.set_title(f"Read Index: {pretty_read_index(read_index)}")

    # Build legend handles
    handles = []
    for label in unique_labels:
        if label in color_map and label in marker_map:
            color = color_map[label]
            marker = marker_map[label]
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

    fig.legend(handles=handles, loc="center right", frameon=True)

    # Hide unused axes
    for idx in range(len(read_indices), nrows * ncols):
        row_idx = idx // ncols
        col_idx = idx % ncols
        axes[row_idx, col_idx].set_visible(False)

    fig.suptitle("Radviz Visualization by Read Index")
    plt.savefig(output_path)
    plt.close()
