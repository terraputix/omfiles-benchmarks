import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from om_benchmarks.formats import AvailableFormats
from om_benchmarks.plotting.formatters import TIME_FORMATTER
from om_benchmarks.plotting.params import _set_matplotlib_behaviour

_set_matplotlib_behaviour()

_PALETTE = sns.color_palette("colorblind", n_colors=len(AvailableFormats))
_FORMAT_COLORS = {fmt: _PALETTE[i] for i, fmt in enumerate(AvailableFormats)}
_LABEL_TO_COLOR = {fmt.plot_label: _PALETTE[i] for i, fmt in enumerate(AvailableFormats)}


def plot_concurrency_scaling(results: dict[AvailableFormats, dict[int, tuple[list[float], float]]], output_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))

    for format, res in results.items():
        concurrencies = list(res.keys())
        mean_latencies = [statistics.mean(latencies) for (latencies, _) in res.values()]
        throughput = [len(latencies) / total_time for _, (latencies, total_time) in res.items()]
        color = _FORMAT_COLORS[format]

        # Draw lines between points with width scaled by concurrency
        for j in range(1, len(concurrencies)):
            lw = 0.5 + 2.5 * (concurrencies[j] - min(concurrencies)) / (max(concurrencies) - min(concurrencies) + 1e-9)
            ax.plot(
                [throughput[j - 1], throughput[j]],
                [mean_latencies[j - 1], mean_latencies[j]],
                linestyle="-",
                linewidth=lw,
                color=color,
                alpha=0.7,
                zorder=1,
            )

        ax.scatter(
            throughput,
            mean_latencies,
            s=np.arange(len(concurrencies)) * 10 + 10,
            marker=getattr(format, "scatter_plot_marker", "o"),
            label=format.plot_label,
            edgecolor="k",
            color=color,
            alpha=0.7,
            zorder=2,
        )

    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Throughput (req/s)")
    ax.set_ylabel("Mean Latency (s)")
    # ax.yaxis.set_major_formatter(TIME_FORMATTER)
    ax.set_title("Throughput vs. Latency")
    ax.legend(title="Format", frameon=True, fancybox=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_concurrency_violin(results: dict[AvailableFormats, dict[int, tuple[list[float], float]]], output_path: str):
    """
    Plots a violin plot of latency distributions for each format at each concurrency level.
    """
    # Flatten results into a DataFrame
    records = []
    for fmt, conc_dict in results.items():
        for conc, (latencies, _) in conc_dict.items():
            for latency in latencies:
                records.append(
                    {
                        "Format": fmt.plot_label if hasattr(fmt, "plot_label") else str(fmt),
                        "Concurrency": conc,
                        "Latency": latency,
                    }
                )
    df = pd.DataFrame(records)

    plt.figure(figsize=(7, 12))
    ax = sns.violinplot(
        x="Latency",
        y="Concurrency",
        hue="Format",
        data=df,
        orient="h",
        density_norm="width",
        split=False,
        inner="quartile",
        log_scale=(True, False),
        palette=_LABEL_TO_COLOR,
    )
    ax.set_ylabel("Concurrency Level")
    ax.set_xlabel("Latency")
    ax.set_title("Latency Distribution by Concurrency and Format")
    ax.xaxis.set_major_formatter(TIME_FORMATTER)
    plt.legend(title="Format", frameon=True, fancybox=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
