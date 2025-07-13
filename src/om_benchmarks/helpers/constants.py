import os

# Get directory paths from environment variables or use defaults
RESULTS_DIR = os.environ.get("BENCHMARK_RESULTS_DIR", "/run/media/fred/Ext4Data/Benchmarks/benchmark_results")
PLOTS_DIR = os.environ.get("BENCHMARK_PLOTS_DIR", "/run/media/fred/Ext4Data/Benchmarks/benchmark_plots")
FILES_DIR = os.environ.get("BENCHMARK_FILES_DIR", "/run/media/fred/Ext4Data/Benchmarks/benchmark_files")

# Make sure the paths are normalized
RESULTS_DIR = os.path.normpath(RESULTS_DIR)
PLOTS_DIR = os.path.normpath(PLOTS_DIR)
FILES_DIR = os.path.normpath(FILES_DIR)
