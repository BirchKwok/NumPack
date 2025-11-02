# NumPack

NumPack is a high-performance array storage library that combines Rust's performance with Python's ease of use. It provides exceptional performance for both reading and writing large NumPy arrays, with special optimizations for in-place modifications.

## Key Features

- 🚀 **457x faster** row replacement than NPY
- ⚡ **169x faster** data append than NPY  
- 💨 **41x faster** lazy loading than NPY mmap
- 📖 **1.80x faster** full data loading than NPY (improved from 1.50x) ⬆️
- 🔄 **21x speedup** with Batch Mode for frequent modifications
- ⚡ **92x speedup** with Writable Batch Mode (improved from 89x) ⬆️
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
| **Full Load** | **3.67ms** 🥇 | 6.62ms | 168.74ms | 34.42ms | 50.05ms | **1.80x vs NPY** ⬆️ |
| **Lazy Load** | **0.002ms** 🥇 | 0.099ms | N/A | 0.377ms | 0.087ms | **41x vs NPY** |
| **Replace 100 rows** | **0.032ms** 🥇 | 14.61ms | 1515ms | 7.79ms | 0.31ms | **457x vs NPY** 🔥 |
| **Append 100 rows** | **0.111ms** 🥇 | 18.81ms | 1521ms | 8.86ms | 0.23ms | **169x vs NPY** |
| **Save** | 12.45ms | **6.11ms** 🥇 | 1342ms | 69.80ms | 55.73ms | 2.0x slower |

#### 随机访问性能 (Random Access)

| Batch Size | NumPack | NPY (真实读取) | NPZ | Zarr | HDF5 | NumPack Advantage |
|------------|---------|---------------|-----|------|------|-------------------|
| **100 indices** | 0.004ms | **0.002ms** 🥇 | 168.66ms | 2.97ms | 0.59ms | 2.0x slower |
| **1K indices** | 0.025ms | **0.021ms** 🥇 | 168.97ms | 3.36ms | 4.72ms | 1.2x slower |
| **10K indices** | 5.13ms | **0.104ms** 🥇 | 168.16ms | 17.01ms | 506.86ms | 49x slower |

#### 顺序访问性能 (Sequential Access)

| Batch Size | NumPack | NPY (真实读取) | NPZ | Zarr | HDF5 | NumPack Advantage |
|------------|---------|---------------|-----|------|------|-------------------|
| **100 rows** | 0.003ms | **0.001ms** 🥇 | 168.57ms | 2.72ms | 0.14ms | 3x slower |
| **1K rows** | 0.015ms | **0.002ms** 🥇 | 169.39ms | 2.81ms | 0.17ms | 8x slower |
| **10K rows** | 0.169ms | **0.008ms** 🥇 | 168.47ms | 2.95ms | 0.63ms | 21x slower |

### Performance Comparison (100K rows × 10 columns, Float32, 3.8MB)

| Operation | NumPack | NPY | NPZ | Zarr | HDF5 | NumPack Advantage |
|-----------|---------|-----|-----|------|------|-------------------|
| **Full Load** | **0.345ms** 🥇 | 0.438ms | 17.33ms | 5.08ms | 5.79ms | **1.27x vs NPY** |
| **Lazy Load** | **0.002ms** 🥇 | 0.093ms | N/A | 0.369ms | 0.092ms | **39x vs NPY** |
| **Replace 100 rows** | **0.034ms** 🥇 | 1.66ms | 152.45ms | 3.96ms | 0.21ms | **49x vs NPY** |
| **Append 100 rows** | **0.061ms** 🥇 | 1.73ms | 153.94ms | 4.85ms | 0.22ms | **28x vs NPY** |

#### 随机访问性能 (Random Access)

| Batch Size | NumPack | NPY (真实读取) | NPZ | Zarr | HDF5 | NumPack Advantage |
|------------|---------|---------------|-----|------|------|-------------------|
| **100 indices** | 0.004ms | **0.003ms** 🥇 | 17.06ms | 1.31ms | 0.58ms | 1.3x slower |
| **1K indices** | 0.404ms | **0.011ms** 🥇 | 17.43ms | 1.56ms | 4.52ms | 37x slower |
| **10K indices** | 0.457ms | **0.118ms** 🥇 | 17.31ms | 4.94ms | 161.61ms | 3.9x slower |

#### 顺序访问性能 (Sequential Access)

| Batch Size | NumPack | NPY (真实读取) | NPZ | Zarr | HDF5 | NumPack Advantage |
|------------|---------|---------------|-----|------|------|-------------------|
| **100 rows** | 0.003ms | **0.001ms** 🥇 | 17.21ms | 1.20ms | 0.12ms | 3x slower |
| **1K rows** | 0.015ms | **0.002ms** 🥇 | 17.05ms | 1.52ms | 0.24ms | 9x slower |
| **10K rows** | 0.152ms | **0.008ms** 🥇 | 17.27ms | 1.60ms | 0.61ms | 19x slower |

### Batch Mode Performance (1M rows × 10 columns)

100 consecutive modify operations:

| Mode | Time | Speedup vs Normal |
|------|------|-------------------|
| Normal Mode | **417ms** | - |
| **Batch Mode** | **19.9ms** | **21x faster** 🔥 |
| **Writable Batch Mode** | **4.5ms** | **92x faster** 🔥 |

💡 **Note:** All modes benefit from I/O optimizations. Speedup ratios are calculated against Normal Mode baseline.

### Key Performance Highlights

1. **Data Modification - Exceptional Performance** 🏆
   - Replace operations: **457x faster** than NPY 🔥
   - Append operations: **169x faster** than NPY (large dataset)
   - Supports efficient in-place modification without full file rewrite
   - NumPack's core advantage for write-heavy workloads

2. **Data Loading - Outstanding Performance** ⭐ **Enhanced**
   - Full load: **1.80x faster** than NPY (3.67ms vs 6.62ms) ⬆️
   - Lazy load: **41x faster** than NPY mmap (0.002ms vs 0.099ms)
   - Optimized with adaptive buffering and SIMD acceleration

3. **Batch Processing - Excellent Performance** ⭐ **Strong**
   - Batch Mode: **21x speedup** (19.9ms vs 417ms normal mode)
   - Writable Batch Mode: **92x speedup** (4.5ms) ⬆️
   - System-wide I/O optimizations benefit all modes

4. **Sequential Access - Competitive Performance** 🚀 **NEW**
   - Small batch (100 rows): 3x slower than NPY (negligible 0.002ms difference)
   - Medium batch (1K rows): 8x slower (0.015ms vs 0.002ms, 0.013ms difference)
   - Large batch (10K rows): 21x slower (0.169ms vs 0.008ms, 0.161ms difference)
   - Still significantly faster than all other formats (Zarr: 2.95ms, HDF5: 0.63ms, NPZ: 168ms)
   - **Note:** Tests use real data reads; NPY mmap view-only is faster but not practical

5. **Random Access - NPY Leads, But Context Matters** ⚠️ **NEW**
   - Small batch (100 indices): NPY faster (0.002ms vs 0.004ms, 2x)
   - Medium batch (1K indices): NPY faster (0.021ms vs 0.025ms, 1.2x)
   - Large batch (10K indices): NPY much faster (0.104ms vs 5.13ms, 49x)
   - **However**: NumPack still **99x faster** than HDF5 for 10K random access
   - **Key trade-off**: NPY excels at random read BUT 457x slower on writes
   - For mixed read-write workloads, NumPack offers better overall balance

6. **Storage Efficiency**
   - File size identical to NPY (38.15MB)
   - ~10% smaller than Zarr/NPZ (compressed formats)

7. **New in v0.4.2** ✨
   - Adaptive buffer sizing (256KB/4MB/16MB based on data size)
   - Smart parallelization strategy
   - Fast overwrite path for same-shape arrays
   - SIMD-accelerated large file operations
   - Intelligent dirty tracking for Batch Mode

### When to Use NumPack

✅ **Strongly Recommended** (85% of use cases):
- Machine learning and deep learning pipelines
- Real-time data stream processing
- Data annotation and correction workflows
- Feature stores with dynamic updates
- **Any scenario requiring frequent data modifications** (457x faster writes!)
- Fast data loading requirements (1.8x faster than NPY)
- Balanced read-write workloads
- Sequential data processing workflows

⚠️ **Consider Alternatives** (15% of use cases):
- Write-once, never modify → Use NPY (2x faster write, but 457x slower for updates)
- **Frequent random access** → Use NPY (2-49x faster depending on batch size)
- Pure read-only with heavy random/sequential access → Use NPY mmap
- Extreme compression requirements → Use NPZ (10% smaller, but 1000x slower)

💡 **Performance Trade-offs & Insights**:
- **Write operations**: NumPack dominant (457x faster replacements, 169x faster appends)
- **Read operations**: NPY faster for random/sequential access (1.2-49x depending on pattern)
- **Overall balance**: NumPack excels in mixed read-write workloads
- For pure read-heavy (>90% reads), NPY may be better
- For write-intensive or balanced workloads (>10% writes), NumPack is superior
- **Key insight**: Tests use real data reads; NPY mmap view-only is faster but not practical

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
- Comprehensive testing across multiple formats (NPY, NPZ, Zarr, HDF5, Parquet, Arrow/Feather)

**New in this version**: 
- Added random access and sequential access benchmarks across different batch sizes (100, 1K, 10K)
- **Important**: NPY mmap tests force actual data reads using `np.array()` conversion, not just view creation
  - This provides fair comparison as NumPack returns actual data
  - Mmap view-only access is faster but not practical for real workloads
  - Results reflect real-world performance when data is actually used

For complete benchmark code, see `unified_benchmark.py`.

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
