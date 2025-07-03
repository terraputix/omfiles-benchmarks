import os

from om_benchmarks.helpers.formats import AvailableFormats

# Get directory paths from environment variables or use defaults
RESULTS_DIR = os.environ.get("BENCHMARK_RESULTS_DIR", "benchmark_results")
PLOTS_DIR = os.environ.get("BENCHMARK_PLOTS_DIR", "benchmark_plots")
FILES_DIR = os.environ.get("BENCHMARK_FILES_DIR", "benchmark_files")

# Make sure the paths are normalized
RESULTS_DIR = os.path.normpath(RESULTS_DIR)
PLOTS_DIR = os.path.normpath(PLOTS_DIR)
FILES_DIR = os.path.normpath(FILES_DIR)

DEFAULT_READ_FORMATS = [
    AvailableFormats.HDF5,
    AvailableFormats.Zarr,
    AvailableFormats.NetCDF,
    AvailableFormats.OM,
]

DEFAULT_WRITE_FORMATS = [
    AvailableFormats.HDF5,
    AvailableFormats.Zarr,
    AvailableFormats.NetCDF,
    AvailableFormats.OM,
]
