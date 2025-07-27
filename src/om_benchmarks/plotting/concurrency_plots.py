import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from om_benchmarks.formats import AvailableFormats
from om_benchmarks.plotting.formatters import TIME_FORMATTER
from om_benchmarks.plotting.params import _set_matplotlib_behaviour

_set_matplotlib_behaviour()


def plot_concurrency_scaling(results: dict[AvailableFormats, dict[int, list[float]]], output_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 6))

    palette = sns.color_palette("colorblind", n_colors=len(results))
    format_colors = {fmt: palette[i] for i, fmt in enumerate(results.keys())}

    for i, (format, res) in enumerate(results.items()):
        concurrencies = list(res.keys())
        mean_latencies = [statistics.mean(latencies) for latencies in res.values()]
        throughput = [c / m if m > 0 else 0 for c, m in zip(concurrencies, mean_latencies)]
        color = format_colors[format]

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
            label=format.name,
            edgecolor="k",
            color=color,
            alpha=0.7,
            zorder=2,
        )

    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Throughput (req/s)")
    ax.set_ylabel("Latency")
    ax.yaxis.set_major_formatter(TIME_FORMATTER)
    ax.set_title("Throughput vs. Latency")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/concurrency_scaling.png")
    plt.close(fig)
