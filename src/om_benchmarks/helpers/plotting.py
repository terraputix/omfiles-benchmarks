from pathlib import Path
from typing import cast

import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter

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
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
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


def format_bytes(x, pos):
    """Format bytes into human readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if x < 1024.0:
            return f"{x:.1f}\\,{unit}"
        x /= 1024.0
    return f"{x:.1f}\\,TB"


def format_time(x, pos):
    """Format time into human readable format"""
    if x < 1:
        return f"{x * 1000:.1f}\\,ms"
    elif x < 60:
        return f"{x:.2f}\\,s"
    else:
        return f"{x / 60:.1f}\\,min"


def _get_label(format_name, compression):
    if compression is None or compression == "none" or compression == "":
        return format_name
    return f"{format_name} \n({compression})"


def _get_color_map(df):
    """
    Returns a dict mapping label to color, using base color for format and shade for compression.
    Uses seaborn color palettes for academic consistency and colorblind-friendliness.
    """
    has_compression = "compression" in df.columns

    # Use seaborn colorblind palette for base formats
    unique_formats = df["format"].unique().to_list()
    base_colors = sns.color_palette("colorblind", n_colors=len(unique_formats))
    format_base_color_map = dict(zip(unique_formats, base_colors))

    if not has_compression:
        return {fmt: format_base_color_map[fmt] for fmt in unique_formats}, False

    # Get all unique (format, compression) pairs
    combos = df.select([pl.col("format"), pl.col("compression")]).unique().sort(["format", "compression"]).to_numpy()

    color_map = {}
    for format_name, compression in combos:
        label = _get_label(format_name, compression)
        base_color = format_base_color_map[format_name]
        if compression is None or compression == "none" or compression == "":
            color_map[label] = base_color
        else:
            # Use a lighter version of the base color for compressed variants
            # (You could also use a darker version if you prefer)
            rgb = mcolors.to_rgb(base_color)
            lighter_rgb = mcolors.to_rgb(mcolors.to_hex(rgb))
            # Blend with white for lighter shade
            blend_factor = 0.5  # 0=base, 1=white
            lighter_rgb = tuple((1 - blend_factor) * c + blend_factor * 1.0 for c in rgb)
            color_map[label] = lighter_rgb
    return color_map, True


def get_subplot_grid(df, operations):
    chunk_shapes: list[str] = cast(list[str], df["chunk_shape"].unique().to_list())
    read_indices: list[str] = cast(list[str], df["read_index"].unique().to_list())
    n_rows = len(chunk_shapes) * len(read_indices)
    n_cols = len(operations)
    fig_width = max(8, 5 * n_cols)
    fig_height = max(8, 4 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    plt.subplots_adjust(wspace=0.35, hspace=1.35, top=0.82)
    return fig, axes, chunk_shapes, read_indices


def add_info_box(ax, chunk_shape, read_index):
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


def create_and_save_perf_chart(df: pl.DataFrame, save_dir, file_name="performance_chart.png"):
    output_path = save_dir / file_name

    operations = df["operation"].unique().to_list()

    fig, axes, chunk_shapes, read_indices = get_subplot_grid(df, operations)

    color_map, has_compression = _get_color_map(df)

    # For each subplot
    for row_idx, (chunk_shape, read_index) in enumerate([(cs, ri) for cs in chunk_shapes for ri in read_indices]):
        for col_idx, operation in enumerate(operations):
            ax = axes[row_idx, col_idx]
            filtered_df = df.filter(pl.col("operation") == operation)
            filtered_df = filtered_df.filter(pl.col("chunk_shape") == chunk_shape)
            filtered_df = filtered_df.filter(pl.col("read_index") == read_index)
            filtered_df = filtered_df.sort("mean_time", descending=True)

            labels, mean_times, bar_colors = [], [], []
            for row in filtered_df.iter_rows():
                row_dict = dict(zip(filtered_df.columns, row))
                label = _get_label(row_dict["format"], row_dict.get("compression", None))
                labels.append(label)
                mean_times.append(row_dict["mean_time"])
                bar_colors.append(color_map[label])

            ax.bar(labels, mean_times, color=bar_colors, edgecolor="white", linewidth=0.5)
            ax.set_xlabel("Format" + ("(Compression)" if has_compression else ""), fontsize=9)
            ax.set_ylabel("Mean Time", fontsize=9)
            ax.yaxis.set_major_formatter(FuncFormatter(format_time))
            ax.tick_params(axis="x", rotation=60 if has_compression else 45, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(True, alpha=0.3, axis="y")

            add_info_box(ax, chunk_shape, read_index)

    # Main title
    main_title = "Performance Comparison by Format"
    subtitle = "Lower execution time indicates better performance"
    plt.suptitle(f"{main_title}\n{subtitle}", fontsize=14, y=0.94)

    plt.savefig(output_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close()


def create_and_save_file_size_chart(df: pl.DataFrame, save_dir: Path, file_name: str = "file_size_comparison.png"):
    output_path = save_dir / file_name
    write_df = df.filter(pl.col("operation") == "write").filter(pl.col("file_size_bytes") > 0)

    if write_df.height == 0:
        print("No write data with file sizes > 0 found")
        return

    # Sort by file size (descending)
    write_df = write_df.sort("file_size_bytes", descending=True)

    formats = write_df["format"].to_list()
    file_sizes = write_df["file_size_bytes"].to_list()

    # Create colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(formats)))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(formats, file_sizes, color=colors, edgecolor="white", linewidth=0.5)

    # Customize the plot
    ax.set_title(
        "\\textbf{File Size Comparison by Format}\n\\textit{Smaller file sizes indicate better compression}",
        fontsize=18,
        pad=25,
    )
    ax.set_xlabel("\\textbf{Format}", fontsize=14)
    ax.set_ylabel("\\textbf{File Size}", fontsize=14)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, value in zip(bars, file_sizes):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=90,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close()


def create_and_save_memory_usage_chart(df: pl.DataFrame, save_dir: Path, file_name: str = "memory_usage.png"):
    output_path = save_dir / file_name

    operations = df["operation"].unique().to_list()
    fig, axes, chunk_shapes, read_indices = get_subplot_grid(df, operations)

    color_map, has_compression = _get_color_map(df)

    for row_idx, (chunk_shape, read_index) in enumerate([(cs, ri) for cs in chunk_shapes for ri in read_indices]):
        for col_idx, operation in enumerate(operations):
            ax = axes[row_idx, col_idx]
            filtered_df = df.filter(pl.col("operation") == operation)
            filtered_df = filtered_df.filter(pl.col("chunk_shape") == chunk_shape)
            filtered_df = filtered_df.filter(pl.col("read_index") == read_index)
            filtered_df = filtered_df.sort("memory_usage_bytes", descending=True)

            labels, memory_usages, bar_colors = [], [], []
            for row in filtered_df.iter_rows():
                row_dict = dict(zip(filtered_df.columns, row))
                label = _get_label(row_dict["format"], row_dict.get("compression", None))
                labels.append(label)
                memory_usages.append(row_dict["memory_usage_bytes"])
                bar_colors.append(color_map[label])

            ax.bar(labels, memory_usages, color=bar_colors, edgecolor="white", linewidth=0.5)
            ax.set_xlabel("Format" + ("(Compression)" if has_compression else ""), fontsize=9)
            ax.set_ylabel("Memory Usage (bytes)", fontsize=9)
            ax.yaxis.set_major_formatter(FuncFormatter(format_bytes))
            ax.tick_params(axis="x", rotation=60 if has_compression else 45, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(True, alpha=0.3, axis="y")

            add_info_box(ax, chunk_shape, read_index)

    main_title = "Memory Usage by Format and Operation"
    subtitle = "Lower memory usage indicates better efficiency"
    plt.suptitle(f"{main_title}\n{subtitle}", fontsize=14, y=0.94)

    plt.savefig(output_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close()


def get_marker_for_compression(compression):
    # Assign a marker for each compression type for clarity
    markers = ["o", "s", "D", "^", "v", "P", "*", "X", "<", ">"]
    if compression is None or compression == "none" or compression == "":
        return "o"
    # Deterministic assignment
    return markers[hash(compression) % len(markers)]


def create_scatter_size_vs_time(df: pl.DataFrame, save_dir, file_name="scatter_size_vs_time.png"):
    output_path = save_dir / file_name

    # Only consider 'read' operation
    df = df.filter(pl.col("operation") == "read")
    if df.height == 0:
        raise ValueError("No read operation data found.")

    read_indices = df["read_index"].unique().to_list() if "read_index" in df.columns else [None]
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

    # Color by format, marker by compression
    unique_formats = df["format"].unique().to_list()
    base_colors = sns.color_palette("colorblind", n_colors=len(unique_formats))
    format_color_map = dict(zip(unique_formats, base_colors))

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
                fmt = row_dict["format"]
                compression = row_dict.get("compression", None)
                color = format_color_map[fmt]
                marker = get_marker_for_compression(compression)
                label = _get_label(fmt, compression)
                ax.scatter(
                    row_dict["file_size_bytes"],
                    row_dict["mean_time"],
                    color=color,
                    marker=marker,
                    s=80,
                    edgecolor="black",
                    label=label,
                )

            ax.set_xlabel("File Size", fontsize=11)
            ax.set_ylabel("Mean Read Time", fontsize=11)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(format_time))
            ax.yaxis.set_minor_formatter(FuncFormatter(format_time))
            ax.xaxis.set_major_formatter(FuncFormatter(format_bytes))
            ax.xaxis.set_minor_formatter(FuncFormatter(format_bytes))
            ax.minorticks_on()
            ax.grid(True, alpha=0.3, axis="both")
            ax.grid(which="minor", alpha=0.2)
            ax.tick_params(axis="both", which="both", labelsize=8)
            ax.set_title(f"Random read of size {str(read_index)}", fontsize=11)

            for row in filtered_df.iter_rows():
                row_dict = dict(zip(filtered_df.columns, row))
                fmt = row_dict["format"]
                compression = row_dict.get("compression", None)
                color = format_color_map[fmt]
                marker = get_marker_for_compression(compression)
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

    fig.suptitle(
        r"\Large File Size vs. Mean Read Time (log-log)"
        "\n"
        r"\normalsize File Chunk Shape " + str(chunk_shape),
        y=0.96,
    )
    plt.savefig(output_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_radviz_results(df: pl.DataFrame, save_dir, file_name="radviz_results.png"):
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
        if "compression" in pdf.columns:
            return _get_label(row["format"], row["compression"])
        else:
            return row["format"]

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
        sub_pdf_plot: pd.DataFrame = sub_pdf[radviz_cols + ["label"]].copy()

        pd.plotting.radviz(
            sub_pdf_plot,
            class_column="label",
            color=sns.color_palette("colorblind", n_colors=sub_pdf_plot["label"].nunique()),
            ax=ax,
        )
        ax.set_title(f"Read Index: {pretty_read_index(read_index)}" if read_index is not None else "All Read Indices")
        ax.legend(loc="upper right", bbox_to_anchor=(1.05, 1), fontsize=8, frameon=True)

    # Hide unused axes
    for idx in range(len(read_indices), nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    fig.suptitle("Radviz Visualization by Read Index", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close()
