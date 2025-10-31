# NumPack

NumPack is a high-performance array storage library that combines Rust's performance with Python's ease of use. It provides exceptional performance for both reading and writing large NumPy arrays, with special optimizations for in-place modifications.

## Key Features

- 🚀 **458x faster** row replacement than NPY (improved from 397x)
- ⚡ **343x faster** data append than NPY  
- 💨 **46x faster** lazy loading than NPY mmap
- 📖 **1.50x faster** full data loading than NPY (improved from 1.3x)
- 🔄 **21x speedup** with Batch Mode for frequent modifications
- ⚡ **89x speedup** with Writable Batch Mode
- 💾 Zero-copy operations with minimal memory footprint
- 🛠 Seamless integration with existing NumPy workflows

## What's New in v0.4.2 ✨

### Major Performance Enhancements 🚀

- **50% faster Full Load** (8.27ms → 4.11ms for 38MB data)
- **27% faster Save** (16.15ms → 11.76ms)
- **43% faster Batch Mode** (34ms → 19.5ms)
- **38% faster Replace** operations (0.047ms → 0.029ms)
- **15% improvement** in competitive advantage (1.3x → 1.50x faster than NPY)

### New I/O Optimizations 🔧

1. **Adaptive Buffer Sizing**
   - Small arrays (<1MB): 256KB buffer → 96% memory saving
   - Medium arrays (1-10MB): 4MB buffer → balanced performance
   - Large arrays (>10MB): 16MB buffer → maximum throughput

2. **Smart Parallelization**
   - Automatically parallelizes only when beneficial (>10MB total data)
   - Avoids thread overhead for small datasets

3. **Fast Overwrite Path**
   - Same-shape array overwrite: 1.5-2.5x faster
   - Uses in-place update instead of file recreation

4. **SIMD Acceleration**
   - Large files (>10MB) use SIMD-optimized operations
   - Theoretical 2-4x speedup for memory-intensive operations

5. **Batch Mode Intelligence**
   - Smart dirty tracking: only flushes modified arrays
   - Zero-copy cache detection
   - Reduced metadata synchronization

### Core Advantages Enhanced

- Replace operations now **458x faster** than NPY (up from 397x) 🔥
- Full Load now **1.50x faster** than NPY (up from 1.3x) 📈
- System-wide optimizations benefit all operation modes

## Features

- **High Performance**: Optimized for both reading and writing large numerical arrays
- **Lazy Loading Support**: Efficient memory usage through on-demand data loading
- **In-place Operations**: Support for in-place array modifications without full file rewrite
- **Batch Processing Modes**: 
  - Batch Mode: 21x speedup for batch operations
  - Writable Batch Mode: 89x speedup for frequent modifications
- **Multiple Data Types**: Supports various numerical data types including:
  - Boolean
  - Unsigned integers (8-bit to 64-bit)
  - Signed integers (8-bit to 64-bit)
  - Floating point (16-bit, 32-bit and 64-bit)
  - Complex numbers (64-bit and 128-bit)

## Installation

### From PyPI (Recommended)

#### Prerequisites
- Python >= 3.9
- NumPy >= 1.26.0

```bash
pip install numpack
```

### From Source

#### Prerequisites (All Platforms including Windows)

- Python >= 3.9
- **Rust >= 1.70.0** (Required on all platforms, install from [rustup.rs](https://rustup.rs/))
- NumPy >= 1.26.0
- Appropriate C/C++ compiler
  - Windows: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  - macOS: Xcode Command Line Tools (`xcode-select --install`)
  - Linux: GCC/Clang (`build-essential` on Ubuntu/Debian)

#### Build Steps

1. Clone the repository:
```bash
git clone https://github.com/BirchKwok/NumPack.git
cd NumPack
```

2. Install maturin:
```bash
pip install maturin>=1.0,<2.0
```

3. Build and install:
```bash
# Install in development mode
maturin develop

# Or build wheel package
maturin build --release
pip install target/wheels/numpack-*.whl
```

## Usage

### Basic Operations

```python
import numpy as np
from numpack import NumPack

# Using context manager (Recommended)
with NumPack("data_directory") as npk:
    # Save arrays
    arrays = {
        'array1': np.random.rand(1000, 100).astype(np.float32),
        'array2': np.random.rand(500, 200).astype(np.float32)
    }
    npk.save(arrays)
    
    # Load arrays - Normal mode
    loaded = npk.load("array1")
    
    # Load arrays - Lazy mode
    lazy_array = npk.load("array1", lazy=True)
```

### Advanced Operations

```python
with NumPack("data_directory") as npk:
    # Replace specific rows
    replacement = np.random.rand(10, 100).astype(np.float32)
    npk.replace({'array1': replacement}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    # Append new data
    new_data = {'array1': np.random.rand(100, 100).astype(np.float32)}
    npk.append(new_data)
    
    # Drop arrays or specific rows
    npk.drop('array1')  # Drop entire array
    npk.drop('array2', [0, 1, 2])  # Drop specific rows
    
    # Random access operations
    data = npk.getitem('array1', [0, 1, 2])
    data = npk['array1']  # Dictionary-style access
    
    # Stream loading for large arrays
    for batch in npk.stream_load('array1', buffer_size=1000):
        process_batch(batch)
```

### Batch Processing Modes

NumPack provides two high-performance batch modes for scenarios with frequent modifications:

#### Batch Mode (21x speedup, 43% faster than before)

```python
with NumPack("data.npk") as npk:
    with npk.batch_mode():
        for i in range(1000):
            arr = npk.load('data')      # Load from cache
            arr[:10] *= 2.0
            npk.save({'data': arr})     # Save to cache
# All changes written to disk on exit
# ✨ Now with smart dirty tracking and zero-copy detection
```

#### Writable Batch Mode (89x speedup)

```python
with NumPack("data.npk") as npk:
    with npk.writable_batch_mode() as wb:
        for i in range(1000):
            arr = wb.load('data')   # Memory-mapped view
            arr[:10] *= 2.0         # Direct modification
            # No save needed - changes are automatic
```

## Performance

All benchmarks were conducted on macOS (Apple Silicon) using the Rust backend with precise timeit measurements.

### Performance Comparison (1M rows × 10 columns, Float32, 38.1MB)

| Operation | NumPack | NPY | NPZ | Zarr | HDF5 | NumPack Advantage |
|-----------|---------|-----|-----|------|------|-------------------|
| **Full Load** | **4.11ms** 🥇 | 6.17ms | 165.03ms | 34.37ms | 54.02ms | **1.50x vs NPY** ⬆️ |
| **Lazy Load** | **0.002ms** 🥇 | 0.091ms | N/A | 0.357ms | 0.102ms | **46x vs NPY** |
| **Replace 100 rows** | **0.029ms** 🥇 | 13.29ms | 1493ms | 7.82ms | 0.50ms | **458x vs NPY** 🔥 |
| **Append 100 rows** | **0.060ms** 🥇 | 20.58ms | 1499ms | 8.88ms | 0.21ms | **343x vs NPY** |
| **Random Access (1K)** | 0.047ms | **0.010ms** 🥇 | 165.83ms | 2.93ms | 5.05ms | - |
| **Save** | 11.76ms | **6.20ms** 🥇 | 1332ms | 69.11ms | 57.59ms | 1.9x slower |

### Performance Comparison (100K rows × 10 columns, Float32, 3.8MB)

| Operation | NumPack | NPY | NPZ | Zarr | HDF5 | NumPack Advantage |
|-----------|---------|-----|-----|------|------|-------------------|
| **Full Load** | **0.34ms** 🥇 | 0.41ms | 16.79ms | 5.03ms | 5.68ms | **1.2x vs NPY** |
| **Lazy Load** | **0.003ms** 🥇 | 0.088ms | N/A | 0.414ms | 0.079ms | **29x vs NPY** |
| **Replace 100 rows** | **0.035ms** 🥇 | 1.36ms | 150.41ms | 3.55ms | 0.19ms | **39x vs NPY** |
| **Append 100 rows** | **0.056ms** 🥇 | 1.66ms | 150.73ms | 3.53ms | 0.20ms | **30x vs NPY** |
| **Random Access (1K)** | 0.055ms | **0.010ms** 🥇 | 16.32ms | 1.58ms | 4.66ms | - |

### Batch Mode Performance (1M rows × 10 columns)

100 consecutive modify operations:

| Mode | Time | Improvement from v0.4.0 | Speedup |
|------|------|----------------------|---------|
| Normal Mode | **409ms** | 52% faster ✨ | - |
| **Batch Mode** | **19.5ms** | 43% faster ✨ | **21x faster** 🔥 |
| **Writable Batch Mode** | **4.6ms** | 6% faster | **89x faster** 🔥 |

💡 **Note:** All modes benefit from v0.4.2 I/O optimizations. Speedup ratios are calculated against the optimized Normal Mode baseline.

### Key Performance Highlights

1. **Data Modification - Exceptional Performance** 🏆
   - Replace operations: **458x faster** than NPY (improved from 397x) 🔥
   - Append operations: **343x faster** than NPY (large dataset)
   - Supports efficient in-place modification without full file rewrite
   - NumPack's core advantage - now even stronger

2. **Data Loading - Outstanding Improvements** ⭐ **Significantly Enhanced**
   - Full load: **1.50x faster** than NPY (improved from 1.3x)
   - Lazy load: **46x faster** than NPY mmap (0.002ms)
   - **50% faster** than previous version (8.27ms → 4.11ms)
   - Optimized with adaptive buffering and SIMD acceleration

3. **Batch Processing - Enhanced Performance** ⭐ **Improved**
   - Batch Mode: **21x speedup**, 43% faster than before
   - Writable Batch Mode: **89x speedup**, maintained excellence
   - System-wide I/O optimizations benefit all modes
   - Normal Mode also 52% faster from optimizations

4. **Storage Efficiency**
   - File size identical to NPY
   - ~10% smaller than Zarr/NPZ (compressed formats)

5. **New in v0.4.2** ✨
   - Adaptive buffer sizing (256KB/4MB/16MB based on data size)
   - Smart parallelization strategy
   - Fast overwrite path for same-shape arrays
   - SIMD-accelerated large file operations
   - Intelligent dirty tracking for Batch Mode

### When to Use NumPack

✅ **Strongly Recommended** (90% of use cases):
- Machine learning and deep learning pipelines
- Real-time data stream processing
- Data annotation and correction workflows
- Feature stores with dynamic updates
- Any scenario requiring frequent data modifications
- Fast data loading requirements

⚠️ **Consider Alternatives** (10% of use cases):
- Write-once, never modify → Use NPY (faster initial write)
- Frequent single-row access → Use NPY mmap
- Extreme compression requirements → Use NPZ (10% smaller, but 1000x slower)

## Best Practices

### 1. Use Writable Batch Mode for Frequent Modifications

```python
# 89x speedup for frequent modifications
with NumPack("data.npk") as npk:
    with npk.writable_batch_mode() as wb:
        for i in range(1000):
            arr = wb.load('data')
            arr[:10] *= 2.0
# Automatic persistence on exit
```

### 2. Use Batch Mode for Batch Operations

```python
# 21x speedup for batch processing (43% faster than before!)
with NumPack("data.npk") as npk:
    with npk.batch_mode():
        for i in range(1000):
            arr = npk.load('data')
            arr[:10] *= 2.0
            npk.save({'data': arr})
# Single write on exit with smart dirty tracking
```

### 3. Use Lazy Loading for Large Datasets

```python
with NumPack("large_data.npk") as npk:
    # Only 0.002ms to initialize
    lazy_array = npk.load("array", lazy=True)
    # Data loaded on demand
    subset = lazy_array[1000:2000]
```

### 4. Reuse NumPack Instances

```python
# ✅ Efficient: Reuse instance
with NumPack("data.npk") as npk:
    for i in range(100):
        data = npk.load('array')

# ❌ Inefficient: Create new instance each time
for i in range(100):
    with NumPack("data.npk") as npk:
        data = npk.load('array')
```

## Benchmark Methodology

All benchmarks use:
- `timeit` for precise timing
- Multiple repeats, best time selected
- Pure operation time (excluding file open/close overhead)
- Float32 arrays
- macOS Apple Silicon (results may vary by platform)

For complete benchmark code, see `comprehensive_format_benchmark.py`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License, Version 2.0 - see the LICENSE file for details.


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
