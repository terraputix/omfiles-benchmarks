import os
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple, Union

import polars as pl
import typer

from om_benchmarks.plotting import create_and_save_perf_chart
from om_benchmarks.script_utils import get_script_dirs
from om_benchmarks.stats import _clear_cache

# Type definitions
ReadSelection = Union[Tuple[int, int, int], None]  # (t, y, x) dimensions or None for full file


@dataclass
class BenchmarkResult:
    language: str
    file_path: str
    file_size: int
    elapsed_time: float
    elapsed_times: List[float]  # Store individual run times
    read_selection: Optional[ReadSelection] = None
    read_volume: Optional[int] = None  # Calculated from read_selection


class LanguageImplementation(Protocol):
    """Protocol defining how to benchmark a language implementation"""

    def benchmark(self, file_path: str, read_selection: ReadSelection, iterations: int) -> BenchmarkResult: ...


class PythonImplementation:
    def benchmark(self, file_path: str, read_selection: ReadSelection, iterations: int) -> BenchmarkResult:
        """Benchmark Python implementation using direct module import"""
        from omfiles import OmFilePyReader

        times: List[float] = []
        reader = OmFilePyReader(file_path)
        for _ in range(iterations):
            start_time = time.time()

            if read_selection is None:
                # Read entire file
                data = reader[...]
            else:
                # Use slice notation for the selection
                t_slice = slice(None, read_selection[0]) if read_selection[0] > 0 else slice(None)
                y_slice = slice(None, read_selection[1]) if read_selection[1] > 0 else slice(None)
                x_slice = slice(None, read_selection[2]) if read_selection[2] > 0 else slice(None)
                data = reader[t_slice, y_slice, x_slice]
                print(f"Read data shape: {data.shape}")

            elapsed = time.time() - start_time
            times.append(elapsed)

        reader.close()
        # Calculate read volume (number of elements read)
        read_volume = None
        if read_selection is not None:
            read_volume = read_selection[0] * read_selection[1] * read_selection[2]

        return BenchmarkResult(
            language="python",
            file_path=file_path,
            file_size=os.path.getsize(file_path),
            elapsed_time=statistics.mean(times),
            elapsed_times=times,
            read_selection=read_selection,
            read_volume=read_volume,
        )


class CLIImplementation:
    def __init__(self, language: str, command_parts: List[str]):
        self.language = language
        self.command_parts = command_parts  # List of command parts (executable and any initial args)

    def benchmark(self, file_path: str, read_selection: ReadSelection, iterations: int) -> BenchmarkResult:
        """Benchmark implementation using CLI binary"""
        cmd = self.command_parts.copy()  # Start with the executable and any initial args
        cmd.append(file_path)  # Add the file path

        # Add read selection arguments if provided
        if read_selection is not None:
            cmd.extend([str(d) for d in read_selection])

        times: List[float] = []
        for _ in range(iterations):
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                raise Exception(f"{self.language} executable failed with code {result.returncode}")
            elapsed = time.time() - start_time
            times.append(elapsed)

        # Calculate read volume if applicable
        read_volume = None
        if read_selection is not None:
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
    read_selections: List[ReadSelection],
    languages: List[str],
    implementations: Dict[str, LanguageImplementation],
    iterations: int,
) -> List[BenchmarkResult]:
    """Run benchmarks for specified languages on all test files and read selections"""
    results = []

    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        for read_selection in read_selections:
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

    # Set up read selections
    read_selections: List[ReadSelection] = [
        (721, 1440, 1),  # Worst case read scenario
    ]

    # Determine languages to benchmark
    languages = ["python", "rust", "swift", "typescript"]

    # Initialize implementations
    implementations: Dict[str, LanguageImplementation] = {
        "python": PythonImplementation(),
        "rust": CLIImplementation("rust", [rust_binary]),
        "swift": CLIImplementation("swift", [swift_binary]),
        "typescript": CLIImplementation("typescript", ["node", ts_script]),
    }

    results = run_benchmarks(file, read_selections, languages, implementations, iterations)

    # Save and visualize results
    df = save_results(results, results_dir)
    create_and_save_perf_chart(df, plots_dir, file_name="language_comparison.png")

    print("Benchmarks complete!")


if __name__ == "__main__":
    typer.run(main)
