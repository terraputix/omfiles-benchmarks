#!/bin/bash

# ERA5 Read/Write Time and Memory Benchmark Script
# This script runs a series of benchmarks for ERA5 data operations

set -e  # Exit on any error

# Function to check if swap is enabled
check_swap_status() {
    if [ "$(swapon --show | wc -l)" -gt 0 ]; then
        return 0  # Swap is enabled
    else
        return 1  # Swap is disabled
    fi
}

# Function to cleanup and restore swap on exit
cleanup() {
    if [ "$SWAP_WAS_ENABLED" = "true" ]; then
        echo ""
        echo "Re-enabling swap..."
        sudo swapon -a
        echo "Swap re-enabled successfully."
    fi
}

# Set trap to ensure cleanup happens even if script fails
trap cleanup EXIT

echo "Starting ERA5 Read/Write Benchmark Suite..."
echo "=============================================="

# Check current swap status and disable if enabled
if check_swap_status; then
    echo "Swap is currently enabled. Disabling swap for accurate memory benchmarking..."
    SWAP_WAS_ENABLED="true"
    sudo swapoff -a
    echo "Swap disabled successfully."
else
    echo "Swap is already disabled."
    SWAP_WAS_ENABLED="false"
fi

echo ""
echo "1. Running write time benchmark (3 iterations)..."
uv run --extra xbitinfo scripts/era5_read_write_time_memory.py --mode time --op-mode write --iterations 3

echo ""
echo "2. Running write memory benchmark (1 iteration)..."
uv run --extra xbitinfo --extra memray scripts/era5_read_write_time_memory.py --mode memory --op-mode write --iterations 1

echo ""
echo "3. Running read time benchmark (20 iterations)..."
uv run --extra xbitinfo scripts/era5_read_write_time_memory.py --mode time --op-mode read --iterations 20

echo ""
echo "4. Running read memory benchmark (1 iteration)..."
uv run --extra xbitinfo --extra memray scripts/era5_read_write_time_memory.py --mode memory --op-mode read --iterations 1

echo ""
echo "=============================================="
echo "ERA5 Read/Write Benchmark Suite completed!"
