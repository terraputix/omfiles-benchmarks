import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_env_path(var_name, default):
    path = os.environ.get(var_name, default)
    path = os.path.normpath(path)
    # Check if path is set and is a directory
    if not path or not os.path.isdir(path):
        print(f"Warning: Environment variable {var_name} is not set. Using default: {default}")
        path = os.path.normpath(default)
    return path


RESULTS_DIR = get_env_path("BENCHMARK_RESULTS_DIR", "benchmark_results")
PLOTS_DIR = get_env_path("BENCHMARK_PLOTS_DIR", "benchmark_plots")
FILES_DIR = get_env_path("BENCHMARK_FILES_DIR", "benchmark_files")
