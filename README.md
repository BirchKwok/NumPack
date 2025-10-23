# NumPack

NumPack is a high-performance array storage library that combines Rust's performance with Python's ease of use. It provides exceptional performance for both reading and writing large NumPy arrays, with special optimizations for in-place modifications.

## Key Features

- 🚀 **397x faster** row replacement than NPY
- ⚡ **405x faster** data append than NPY  
- 💨 **54x faster** lazy loading than NPY mmap
- 📖 **1.3x faster** full data loading than NPY
- 🔄 **174x speedup** with Writable Batch Mode for frequent modifications
- 💾 Zero-copy operations with minimal memory footprint
- 🛠 Seamless integration with existing NumPy workflows

## Features

- **High Performance**: Optimized for both reading and writing large numerical arrays
- **Lazy Loading Support**: Efficient memory usage through on-demand data loading
- **In-place Operations**: Support for in-place array modifications without full file rewrite
- **Batch Processing Modes**: 
  - Batch Mode: 25-37x speedup for batch operations
  - Writable Batch Mode: 174x speedup for frequent modifications
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

#### Batch Mode (25-37x speedup)

```python
with NumPack("data.npk") as npk:
    with npk.batch_mode():
        for i in range(1000):
            arr = npk.load('data')      # Load from cache
            arr[:10] *= 2.0
            npk.save({'data': arr})     # Save to cache
# All changes written to disk on exit
```

#### Writable Batch Mode (174x speedup)

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

| Operation | NumPack | NPY | NPZ | Zarr | HDF5 | Parquet | NumPack Advantage |
|-----------|---------|-----|-----|------|------|---------|-------------------|
| **Full Load** | **8.27ms** 🥇 | 10.51ms | 181.62ms | 41.40ms | 58.39ms | 23.74ms | 1.3x vs NPY |
| **Lazy Load** | **0.002ms** 🥇 | 0.107ms | N/A | 0.397ms | 0.080ms | N/A | 54x vs NPY |
| **Replace 100 rows** | **0.047ms** 🥇 | 18.51ms | 1574ms | 9.08ms | 0.299ms | 187.65ms | 397x vs NPY |
| **Append 100 rows** | **0.067ms** 🥇 | 27.09ms | 1582ms | 9.98ms | 0.212ms | 204.74ms | 405x vs NPY |
| **Random Access (1K)** | 0.051ms | **0.010ms** 🥇 | 183.16ms | 3.46ms | 4.91ms | 22.80ms | 26x vs NPZ |
| **Save** | 16.15ms | **7.19ms** 🥇 | 1378ms | 80.91ms | 55.66ms | 159.14ms | 2.2x slower |

### Performance Comparison (100K rows × 10 columns, Float32, 3.8MB)

| Operation | NumPack | NPY | NPZ | Zarr | HDF5 | NumPack Advantage |
|-----------|---------|-----|-----|------|------|-------------------|
| **Full Load** | 0.98ms | **0.66ms** 🥇 | 18.65ms | 6.24ms | 6.35ms | 1.5x slower |
| **Lazy Load** | **0.002ms** 🥇 | 0.103ms | N/A | 0.444ms | 0.085ms | 51x vs NPY |
| **Replace 100 rows** | **0.039ms** 🥇 | 2.13ms | 159.19ms | 4.39ms | 0.208ms | 55x vs NPY |
| **Append 100 rows** | **0.059ms** 🥇 | 3.29ms | 159.19ms | 4.59ms | 0.206ms | 56x vs NPY |
| **Random Access (1K)** | 0.116ms | **0.010ms** 🥇 | 18.73ms | 1.89ms | 4.82ms | 12x vs NPZ |

### Batch Mode Performance (1M rows × 10 columns)

100 consecutive modify operations:

| Mode | Time | Speedup |
|------|------|---------|
| Normal Mode | 856ms | 1.0x |
| **Batch Mode** | 34ms | **25x faster** 🔥 |
| **Writable Batch Mode** | 4.9ms | **174x faster** 🔥🔥 |

### Key Performance Highlights

1. **Data Modification - Exceptional Performance** 🏆
   - Replace operations: **397x faster** than NPY (large dataset)
   - Append operations: **405x faster** than NPY (large dataset)
   - Supports efficient in-place modification without full file rewrite
   - NumPack's core advantage

2. **Data Loading - Industry Leading**
   - Full load: **Fastest** for large datasets (8.27ms)
   - Lazy load: **54x faster** than NPY mmap (0.002ms)
   - Optimized batch data transfer with SIMD acceleration

3. **Batch Processing - Revolutionary Performance**
   - Batch Mode: **25-37x speedup** for batch operations
   - Writable Batch Mode: **174x speedup** for frequent modifications
   - Ideal for machine learning pipelines and data processing workflows

4. **Storage Efficiency**
   - File size identical to NPY
   - ~10% smaller than Zarr/NPZ (compressed formats)

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
# 174x speedup for frequent modifications
with NumPack("data.npk") as npk:
    with npk.writable_batch_mode() as wb:
        for i in range(1000):
            arr = wb.load('data')
            arr[:10] *= 2.0
# Automatic persistence on exit
```

### 2. Use Batch Mode for Batch Operations

```python
# 25-37x speedup for batch processing
with NumPack("data.npk") as npk:
    with npk.batch_mode():
        for i in range(1000):
            arr = npk.load('data')
            arr[:10] *= 2.0
            npk.save({'data': arr})
# Single write on exit
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
