from pathlib import Path
from typing import Tuple

from om_benchmarks.helpers.constants import PLOTS_DIR, RESULTS_DIR


def get_script_dirs(calling_file: str) -> Tuple[Path, Path]:
    """
    Get script-specific directories for results and plots.

    Args:
        calling_file: The path to the calling script (__file__)

    Returns:
        Tuple of (results_dir, plots_dir)
    """
    script_name = Path(calling_file).stem
    results_subdir = Path(RESULTS_DIR) / script_name
    plots_subdir = Path(PLOTS_DIR) / script_name

    # Create directories if they don't exist
    results_subdir.mkdir(parents=True, exist_ok=True)
    plots_subdir.mkdir(parents=True, exist_ok=True)

    return results_subdir, plots_subdir
