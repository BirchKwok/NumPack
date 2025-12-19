#!/usr/bin/env python3
"""
NumPack numpy array creation performance benchmark.

This benchmark measures the performance of converting NumPack data to numpy ndarrays,
which is the core operation we want to optimize.
"""

import numpy as np
import time
import tempfile
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numpack import NumPack

def create_test_file(shape, dtype, dirpath):
    """Create a test NumPack file with random data."""
    data = np.random.rand(*shape).astype(dtype)
    with NumPack(dirpath, drop_if_exists=True, warn_no_context=False) as npk:
        npk.save({'data': data})
    return data

def benchmark_single_row_access(lazy_array, num_iterations=1000):
    """Benchmark single row access (creates numpy array each time)."""
    num_rows = len(lazy_array)
    times = []
    
    for _ in range(num_iterations):
        idx = np.random.randint(0, num_rows)
        start = time.perf_counter()
        _ = lazy_array[idx]
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean_us': np.mean(times) * 1e6,
        'std_us': np.std(times) * 1e6,
        'min_us': np.min(times) * 1e6,
        'max_us': np.max(times) * 1e6,
        'p50_us': np.percentile(times, 50) * 1e6,
        'p99_us': np.percentile(times, 99) * 1e6,
    }

def benchmark_slice_access(lazy_array, slice_size, num_iterations=100):
    """Benchmark slice access (creates numpy array for slice)."""
    num_rows = len(lazy_array)
    times = []
    
    for _ in range(num_iterations):
        start_idx = np.random.randint(0, max(1, num_rows - slice_size))
        end_idx = min(start_idx + slice_size, num_rows)
        
        start = time.perf_counter()
        _ = lazy_array[start_idx:end_idx]
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean_us': np.mean(times) * 1e6,
        'std_us': np.std(times) * 1e6,
        'min_us': np.min(times) * 1e6,
        'max_us': np.max(times) * 1e6,
        'p50_us': np.percentile(times, 50) * 1e6,
        'p99_us': np.percentile(times, 99) * 1e6,
    }

def benchmark_batch_access(lazy_array, batch_size, num_iterations=100):
    """Benchmark batch (fancy indexing) access."""
    num_rows = len(lazy_array)
    times = []
    
    for _ in range(num_iterations):
        indices = np.random.choice(num_rows, size=min(batch_size, num_rows), replace=False)
        
        start = time.perf_counter()
        _ = lazy_array[indices]
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean_us': np.mean(times) * 1e6,
        'std_us': np.std(times) * 1e6,
        'min_us': np.min(times) * 1e6,
        'max_us': np.max(times) * 1e6,
        'p50_us': np.percentile(times, 50) * 1e6,
        'p99_us': np.percentile(times, 99) * 1e6,
    }

def benchmark_full_load(lazy_array, num_iterations=10):
    """Benchmark full array load (load all data to numpy)."""
    times = []
    
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = lazy_array[:]
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean_ms': np.mean(times) * 1e3,
        'std_ms': np.std(times) * 1e3,
        'min_ms': np.min(times) * 1e3,
        'max_ms': np.max(times) * 1e3,
    }

def print_results(name, results, unit='us'):
    """Pretty print benchmark results."""
    print(f"  {name}:")
    if unit == 'us':
        print(f"    Mean: {results['mean_us']:.2f} μs")
        print(f"    Std:  {results['std_us']:.2f} μs")
        print(f"    P50:  {results['p50_us']:.2f} μs")
        print(f"    P99:  {results['p99_us']:.2f} μs")
    else:  # ms
        print(f"    Mean: {results['mean_ms']:.2f} ms")
        print(f"    Std:  {results['std_ms']:.2f} ms")

def run_benchmark():
    """Run the complete benchmark suite."""
    print("=" * 70)
    print("NumPack → NumPy Array Creation Performance Benchmark")
    print("=" * 70)
    
    # Test configurations
    configs = [
        # (shape, dtype, description)
        ((10000, 128), np.float32, "10K x 128 float32 (typical embedding)"),
        ((10000, 128), np.float64, "10K x 128 float64"),
        ((10000, 128), np.int32, "10K x 128 int32"),
        ((10000, 128), np.int64, "10K x 128 int64"),
        ((100000, 64), np.float32, "100K x 64 float32 (large)"),
        ((1000, 1024), np.float32, "1K x 1024 float32 (wide)"),
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for shape, dtype, desc in configs:
            print(f"\n{'='*70}")
            print(f"Config: {desc}")
            print(f"Shape: {shape}, dtype: {dtype}")
            print("=" * 70)
            
            # Create test file
            dirpath = os.path.join(tmpdir, f"test_{dtype.__name__}")
            original_data = create_test_file(shape, dtype, dirpath)
            
            # Load as lazy array
            npk = NumPack(dirpath, warn_no_context=False)
            npk.open()
            lazy_array = npk.load('data', lazy=True)
            
            # Warm up
            _ = lazy_array[0]
            _ = lazy_array[0:10]
            
            # Run benchmarks
            print("\n[Single Row Access (1000 iterations)]")
            single_results = benchmark_single_row_access(lazy_array, num_iterations=1000)
            print_results("Random single row", single_results)
            
            print("\n[Slice Access (100 iterations)]")
            for slice_size in [10, 100, 1000]:
                if slice_size <= shape[0]:
                    slice_results = benchmark_slice_access(lazy_array, slice_size, num_iterations=100)
                    print_results(f"Slice size {slice_size}", slice_results)
            
            print("\n[Batch/Fancy Index Access (100 iterations)]")
            for batch_size in [10, 100, 1000]:
                if batch_size <= shape[0]:
                    batch_results = benchmark_batch_access(lazy_array, batch_size, num_iterations=100)
                    print_results(f"Batch size {batch_size}", batch_results)
            
            print("\n[Full Array Load (10 iterations)]")
            full_results = benchmark_full_load(lazy_array, num_iterations=10)
            print_results("Full load", full_results, unit='ms')
            
            # Calculate throughput
            total_bytes = np.prod(shape) * np.dtype(dtype).itemsize
            throughput_gbps = (total_bytes / 1e9) / (full_results['mean_ms'] / 1e3)
            print(f"    Throughput: {throughput_gbps:.2f} GB/s")
            
            # Close NumPack instance
            npk.close()

    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
