import subprocess
from typing import Dict, List, Tuple, Union

import polars as pl
import typer

from om_benchmarks.plotting.lang_compare_plots import create_and_save_violin_plot
from om_benchmarks.script_utils import get_script_dirs
from om_benchmarks.stats import _clear_cache

ReadSelection = Union[Tuple[int, int, int]]  # (t, y, x) dimensions


class CLIImplementation:
    def __init__(self, language: str, command_parts: List[str], append_command_parts: List[str] = []):
        self.language = language
        self.command_parts = command_parts  # List of command parts (executable and any initial args)
        self.append_command_parts = append_command_parts  # List of command parts to append to the command

    def benchmark(self, file_path: str, read_selection: ReadSelection, iterations: int) -> List[float]:
        """Benchmark implementation using CLI binary"""
        cmd = self.command_parts.copy()

        # add file path, read selection indices, and iterations
        cmd.append(file_path)
        cmd.extend([str(d) for d in read_selection])
        cmd.extend([str(iterations)])

        for append_part in self.append_command_parts:
            cmd.append(append_part)

        # Capture run
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"{self.language} executable failed with code {result.returncode}")

        # Parse output: expect one float per line
        times: List[float] = []
        for line in result.stdout.strip().splitlines():
            try:
                times.append(float(line.strip()))
            except ValueError:
                print(f"Warning: Could not parse line as float: {line}")

        if not times:
            raise Exception(f"No timing data parsed from {self.language} output")

        return times


def run_benchmarks(
    file_path: str,
    read_selection: ReadSelection,
    implementations: List[CLIImplementation],
    iterations: int,
) -> Dict[str, List[float]]:
    """Run benchmarks for specified languages on all test files and read selections"""
    results = {}
    for impl in implementations:
        _clear_cache()

        print(
            f"Benchmarking {impl.language} implementation reading {file_path} "
            + f"with selection {read_selection or 'full file'}"
        )

        try:
            result = impl.benchmark(file_path, read_selection, iterations)
            results[impl.language] = result
        except Exception as e:
            print(f"Error benchmarking {impl.language}: {e}")

    return results


def to_df(results: Dict[str, List[float]]) -> pl.DataFrame:
    """Convert benchmark results to DataFrame"""
    # Convert results to a list of dictionaries
    results_dicts = [{"implementation": impl, "all_times": measurements} for impl, measurements in results.items()]
    df = pl.DataFrame(results_dicts)

    return df


def main(
    python_script: str = typer.Option("lang_compare/python/om-python-bm.py", help="Path to Python script"),
    rust_binary: str = typer.Option("lang_compare/rust/target/release/om-rust-bm", help="Path to Rust binary"),
    swift_binary: str = typer.Option("lang_compare/Swift/.build/release/om-swift-bm", help="Path to Swift binary"),
    ts_script: str = typer.Option("lang_compare/typescript/om-typescript-bm.js", help="Path to TypeScript script"),
    iterations: int = typer.Option(20, help="Number of iterations for each benchmark"),
    file: str = typer.Argument("", help="OM files to benchmark"),
):
    """
    Benchmark Open-Meteo format implementations across different languages
    """
    # Set up directories for results and plots
    results_dir, plots_dir = get_script_dirs(__file__)

    read_selection = (721, 1440, 1)  # Worst case read scenario
    # read_selection = (5, 5, 744)  # Best case read scenario
    # read_selection = (50, 50, 30)  # Balanced read scenario

    # Initialize implementations
    implementations: List[CLIImplementation] = [
        CLIImplementation("python", ["uv", "run", python_script]),
        CLIImplementation("rust", [rust_binary]),
        CLIImplementation("swift", [swift_binary]),
        CLIImplementation("javascript (prefetched)", ["node", ts_script], ["true"]),
        # CLIImplementation("javascript bun (prefetched)", ["bun", ts_script], ["true"]),
        CLIImplementation("javascript (file I/O)", ["node", ts_script]),
    ]

    results = run_benchmarks(file, read_selection, implementations, iterations)

    # Save and visualize results
    df = to_df(results)
    create_and_save_violin_plot(df, plots_dir, file_name=f"language_comparison_violin_{read_selection}.png")

    print("Benchmarks complete!")


if __name__ == "__main__":
    typer.run(main)
