import re
from pathlib import Path
from typing import Tuple

from om_benchmarks.constants import FILES_DIR, PLOTS_DIR, RESULTS_DIR
from om_benchmarks.formats import AvailableFormats


def make_file_name_safe(s: str) -> str:
    # Replace any character that is not alphanumeric, dash, or underscore with underscore
    return re.sub(r"[^A-Za-z0-9_\-]", "_", s)


def get_era5_path_for_hashed_config(format: AvailableFormats, chunk_size: tuple[int, int, int], hash: str) -> Path:
    file_name = f"era5_chunks_{'_'.join(map(str, chunk_size))}_{hash}"
    return Path(f"{FILES_DIR}/{file_name}").with_suffix(f"{format.file_extension}")


def create_directory_with_confirmation(
    directory: Path,
) -> None:
    """
    Create a directory if it doesn't exist, asking for user confirmation first.

    Args:
        directory: The directory path to create

    Raises:
        RuntimeError: If the user doesn't confirm directory creation
    """
    # Check if the directory already exists
    if not directory.exists():
        confirmation = (
            input(
                f"The directory '{directory}' does not exist. Do you want to create it (and its parent-directories)? [y/N]: "
            )
            .strip()
            .lower()
        )

        if confirmation in ("y", "yes"):
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
        else:
            raise RuntimeError(
                f"Directory '{directory}' is required but does not exist. "
                f"Please create it manually or run the script again and confirm creation."
            )


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

    # Create directories with confirmation if they don't exist
    create_directory_with_confirmation(results_subdir)
    create_directory_with_confirmation(plots_subdir)

    return results_subdir, plots_subdir
