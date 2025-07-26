import os
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import polars as pl
import typer

from om_benchmarks.plotting.read_write_plots import create_and_save_perf_chart
from om_benchmarks.script_utils import get_script_dirs
from om_benchmarks.stats import _clear_cache

ReadSelection = Union[Tuple[int, int, int]]  # (t, y, x) dimensions


@dataclass
class BenchmarkResult:
    language: str
    file_path: str
    file_size: int
    elapsed_time: float
    elapsed_times: List[float]  # Store individual run times
    read_selection: Optional[ReadSelection] = None
    read_volume: Optional[int] = None  # Calculated from read_selection


class CLIImplementation:
    def __init__(self, language: str, command_parts: List[str]):
        self.language = language
        self.command_parts = command_parts  # List of command parts (executable and any initial args)

    def benchmark(self, file_path: str, read_selection: ReadSelection, iterations: int) -> BenchmarkResult:
        """Benchmark implementation using CLI binary"""
        cmd = self.command_parts.copy()

        # add file path, read selection indices, and iterations
        cmd.append(file_path)
        cmd.extend([str(d) for d in read_selection])
        cmd.extend([str(iterations)])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"{self.language} executable failed with code {result.returncode}")

        # Parse output: expect one float per line
        times = []
        for line in result.stdout.strip().splitlines():
            try:
                times.append(float(line.strip()))
            except ValueError:
                print(f"Warning: Could not parse line as float: {line}")

        if not times:
            raise Exception(f"No timing data parsed from {self.language} output")

        # Calculate read volume
        read_volume = read_selection[0] * read_selection[1] * read_selection[2]

        return BenchmarkResult(
            language=self.language,
            file_path=file_path,
            file_size=os.path.getsize(file_path),
            elapsed_time=statistics.mean(times),
            elapsed_times=times,
            read_selection=read_selection,
            read_volume=read_volume,
        )


def run_benchmarks(
    test_files: List[str],
    read_selection: ReadSelection,
    languages: List[str],
    implementations: Dict[str, CLIImplementation],
    iterations: int,
) -> List[BenchmarkResult]:
    """Run benchmarks for specified languages on all test files and read selections"""
    results = []

    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        for language in languages:
            if language not in implementations:
                print(f"Warning: No implementation for language {language}")
                continue

            _clear_cache()

            print(
                f"Benchmarking {language} implementation reading {file_path} "
                + f"with selection {read_selection or 'full file'}"
            )

            try:
                impl = implementations[language]
                result = impl.benchmark(file_path, read_selection, iterations)
                results.append(result)
            except Exception as e:
                print(f"Error benchmarking {language}: {e}")

    return results


# Fixme: Results handling
def save_results(results: List[BenchmarkResult], results_dir: Path) -> pl.DataFrame:
    """Save benchmark results to CSV and return as DataFrame"""
    # Convert results to a list of dictionaries
    results_dicts = []
    for r in results:
        # Convert ReadSelection to string for CSV storage
        read_index_str = str(r.read_selection) if r.read_selection else "None"

        # Extract filename from path
        filename = os.path.basename(r.file_path)

        result_dict = {
            "format": "OM",
            "compression": r.language,
            "operation": "read",
            "chunk_shape": "(5,5,744)",
            "file": filename,
            "file_path": r.file_path,
            "file_size": r.file_size,
            "mean_time": r.elapsed_time,
            "read_index": read_index_str,
            "read_volume": r.read_volume if r.read_volume else 0,
        }
        results_dicts.append(result_dict)

    # Create DataFrame
    df = pl.DataFrame(results_dicts)

    # Save to CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"benchmark_results_{platform.node()}_{timestamp}.csv")
    df.write_csv(csv_path)
    print(f"Results saved to {csv_path}")

    return df


def main(
    python_script: str = typer.Option("other_languages/python/om-python-bm.py", help="Path to Python script"),
    rust_binary: str = typer.Option("other_languages/rust/target/release/om-rust-bm", help="Path to Rust binary"),
    swift_binary: str = typer.Option("other_languages/Swift/.build/release/om-swift-bm", help="Path to Swift binary"),
    ts_script: str = typer.Option("other_languages/typescript/om-typescript-bm.js", help="Path to TypeScript script"),
    iterations: int = typer.Option(20, help="Number of iterations for each benchmark"),
    file: List[str] = typer.Argument(..., help="OM files to benchmark"),
):
    """
    Benchmark Open-Meteo format implementations across different languages
    """
    # Set up directories for results and plots
    results_dir, plots_dir = get_script_dirs(__file__)

    # read_selection = (721, 1440, 1)  # Worst case read scenario
    read_selection = (5, 5, 744)  # Best case read scenario

    # Determine languages to benchmark
    languages = ["python", "rust", "swift", "typescript"]

    # Initialize implementations
    implementations: Dict[str, CLIImplementation] = {
        "python": CLIImplementation("python", ["uv", "run", python_script]),
        "rust": CLIImplementation("rust", [rust_binary]),
        "swift": CLIImplementation("swift", [swift_binary]),
        "typescript": CLIImplementation("typescript", ["node", ts_script]),
    }

    results = run_benchmarks(file, read_selection, languages, implementations, iterations)

    # Save and visualize results
    df = save_results(results, results_dir)
    create_and_save_perf_chart(df, plots_dir, file_name=f"language_comparison_{read_selection}.png")

    print("Benchmarks complete!")


if __name__ == "__main__":
    typer.run(main)
