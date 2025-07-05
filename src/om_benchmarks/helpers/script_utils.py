from pathlib import Path
from typing import Tuple

from om_benchmarks.helpers.constants import FILES_DIR, PLOTS_DIR, RESULTS_DIR
from om_benchmarks.helpers.formats import AvailableFormats
from om_benchmarks.helpers.io.writer_configs import FormatWriterConfig


def get_era5_path_for_config(format: AvailableFormats, config: FormatWriterConfig) -> Path:
    chunk_size_str = "_".join(map(str, config.chunk_size))
    compression_str = config.compression_identifier
    file_name = f"era5_chunks_{chunk_size_str}_compr_{compression_str}"

    return Path(f"{FILES_DIR}/{file_name}").with_suffix(f"{format.file_extension}")


def get_era5_path_for_format(format: AvailableFormats, chunk_size: tuple[int]) -> Path:
    chunk_size_str = "_".join(map(str, chunk_size))
    return Path(f"{FILES_DIR}/era5_{chunk_size_str}").with_suffix(f"{format.file_extension}")


def get_file_path_for_format(format: AvailableFormats) -> Path:
    return Path(f"{FILES_DIR}/data").with_suffix(f"{format.file_extension}")


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
