import statistics
from pathlib import Path

import matplotlib.axes
import matplotlib.pyplot as plt

from om_benchmarks.formats import AvailableFormats
from om_benchmarks.plotting.params import _set_matplotlib_behaviour

_set_matplotlib_behaviour()


def plot_concurrency_scaling(results: dict[AvailableFormats, dict[int, list[float]]], output_dir: Path):
    ncols: int = 2
    nrows: int = len(results)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 4 * nrows), squeeze=False)
    plt.subplots_adjust(hspace=0.25, wspace=0.2, top=0.94)
    for i, (format, res) in enumerate(results.items()):
        ax1: matplotlib.axes.Axes = axes[i, 0]
        ax2: matplotlib.axes.Axes = axes[i, 1]
        mean_latencies = [statistics.mean(latencies) for latencies in res.values()]
        median_latencies = [statistics.median(latencies) for latencies in res.values()]
        # std_latencies = [statistics.stdev(latencies) if len(latencies) > 1 else 0.0 for latencies in res.values()]
        concurrencies = list(res.keys())
        throughput = [
            concurrency / mean_latency if mean_latency > 0 else 0
            for concurrency, mean_latency in zip(concurrencies, mean_latencies)
        ]

        ax1.plot(concurrencies, throughput, marker="o")
        ax1.set_xlabel("Concurrency")
        ax1.set_xscale("log", base=2)
        ax1.set_ylabel("Throughput (req/s)")
        ax1.set_yscale("log", base=10)
        ax1.set_title(f"{format.name} Throughput Scaling")

        ax2.plot(concurrencies, mean_latencies, marker="o", label="Mean")
        ax2.plot(concurrencies, median_latencies, marker="x", label="Median")
        ax2.set_xlabel("Concurrency")
        ax2.set_xscale("log", base=2)
        ax2.set_ylabel("Latency (s)")
        ax2.set_yscale("log", base=10)
        ax2.minorticks_on()
        ax2.set_title(f"{format.name} Latency Scaling")
        ax2.legend(frameon=True, fancybox=True)

    fig.suptitle("Concurrency Scaling of Latency and Throughput")
    # plt.tight_layout()
    plt.savefig(f"{output_dir}/concurrency_scaling.png")


def plot_latency(conc, latencies, format, output_dir):
    plt.figure(figsize=(8, 5))
    plt.boxplot(latencies, positions=conc)
    plt.xlabel("Concurrency")
    plt.ylabel("Latency (s)")
    plt.title(f"{format.name} Latency Distribution")
    plt.savefig(f"{output_dir}/{format.name}_latency_boxplot.png")
