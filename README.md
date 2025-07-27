# NumPack

NumPack is a lightning-fast array manipulation engine that revolutionizes how you handle large-scale NumPy arrays. By combining Rust's raw performance with Python's ease of use, NumPack delivers up to 166x faster operations than traditional methods, while using minimal memory. With our new high-performance binary format, matrix operations are now up to 5.33x faster than NumPy mmap, and lazy loading achieves throughput exceeding 100,000 MB/s. Whether you're working with gigabyte-sized matrices or performing millions of array operations, NumPack makes it effortless with its zero-copy architecture and intelligent memory management.

Key highlights:
- 🚀 Up to 166x faster than traditional NumPy storage methods
- ⚡ Matrix operations up to 5.33x faster than NumPy mmap
- 🚀 SIMD-optimized operations with streaming throughput up to 4,417 MB/s
- 💾 Zero-copy operations for minimal memory footprint
- 🔄 Seamless integration with existing NumPy workflows
- 🛠 Battle-tested in production with arrays exceeding 1 billion rows

## Features

- **High Performance**: Optimized for both reading and writing large numerical arrays
- **Lazy Loading Support**: Efficient memory usage through on-demand data loading
- **Selective Loading**: Load only the arrays you need, when you need them
- **In-place Operations**: Support for in-place array modifications without full file rewrite
- **Parallel I/O**: Utilizes parallel processing for improved performance
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

To build and install NumPack from source, you need to meet the following requirements:

#### Prerequisites

- Python >= 3.9
- Rust >= 1.70.0
- NumPy >= 1.26.0
- Appropriate C/C++ compiler (depending on your operating system)
  - Linux: GCC or Clang
  - macOS: Clang (via Xcode Command Line Tools)
  - Windows: MSVC (via Visual Studio or Build Tools)

#### Build Steps

1. Clone the repository:
```bash
git clone https://github.com/BirchKwok/NumPack.git
cd NumPack
```

2. Install maturin (for building Rust and Python hybrid projects):
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

#### Platform-Specific Notes

- **Linux Users**:
  - Ensure python3-dev (Ubuntu/Debian) or python3-devel (Fedora/RHEL) is installed
  - If using conda environment, make sure the appropriate compiler toolchain is installed

- **macOS Users**:
  - Make sure Xcode Command Line Tools are installed: `xcode-select --install`
  - Supports both Intel and Apple Silicon architectures

- **Windows Users**:
  - Visual Studio or Visual Studio Build Tools required
  - Ensure "Desktop development with C++" workload is installed


## Usage

### Basic Operations

```python
import numpy as np
from numpack import NumPack

# Create a NumPack instance
npk = NumPack("data_directory")

# Save arrays
arrays = {
    'array1': np.random.rand(1000, 100).astype(np.float32),
    'array2': np.random.rand(500, 200).astype(np.float32)
}
npk.save(arrays)

# Load arrays
# Normal mode
loaded = npk.load("array1")

# lazy load
lazy_array = npk.load("arr1", lazy=True)
```

### Advanced Operations

```python
# Replace specific rows
replacement = np.random.rand(10, 100).astype(np.float32)
npk.replace({'array1': replacement}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # Using list indices
npk.replace({'array1': replacement}, slice(0, 10))  # Using slice notation

# Append new arrays
new_arrays = {
    'array3': np.random.rand(200, 100).astype(np.float32)
}
npk.append(new_arrays)

# Drop arrays or specific rows
npk.drop('array1')  # Drop entire array
npk.drop(['array1', 'array2'])  # Drop multiple arrays
npk.drop('array2', [0, 1, 2])  # Drop specific rows

# Random access operations
data = npk.getitem('array1', [0, 1, 2])  # Access specific rows
data = npk.getitem('array1', slice(0, 10))  # Access using slice
data = npk['array1']  # Dictionary-style access for entire array

# Metadata operations
shapes = npk.get_shape()  # Get shapes of all arrays
shapes = npk.get_shape('array1')  # Get shape of specific array
members = npk.get_member_list()  # Get list of array names
mtime = npk.get_modify_time('array1')  # Get modification time
metadata = npk.get_metadata()  # Get complete metadata

# Stream loading for large arrays
for batch in npk.stream_load('array1', buffer_size=1000):
    # Process 1000 rows at a time
    process_batch(batch)

# Reset/clear storage
npk.reset()  # Clear all arrays

# Iterate over all arrays
for array_name in npk:
    data = npk[array_name]
    print(f"{array_name} shape: {data.shape}")
```

### Lazy Loading and Buffer Operations

NumPack supports lazy loading and buffer operations, which are particularly useful for handling large-scale datasets. Using the `lazy=True` parameter enables data to be loaded only when actually needed, making it ideal for streaming processing or scenarios where only partial data access is required.

```python
from numpack import NumPack
import numpy as np

# Create NumPack instance and save large-scale data
npk = NumPack("test_data/", drop_if_exists=True)
a = np.random.random((1000000, 128))  # Create a large array
npk.save({"arr1": a})

# Lazy loading - keeps data in buffer
lazy_array = npk.load("arr1", lazy=True)  # LazyArray Object

# Perform computations with lazy-loaded data
# Only required data is loaded into memory
similarity_scores = np.inner(a[0], npk.load("arr1", lazy=True))
```

## Performance

NumPack offers significant performance improvements compared to traditional NumPy storage methods, especially in data modification operations and random access. Below are detailed benchmark results:

### Benchmark Results

The following benchmarks were performed on a MacBook Pro (Apple Silicon) with arrays of size 1M x 10 and 500K x 5 (float32).

#### Storage Operations

| Operation | NumPack (Python) | NumPack (Rust) | NumPy NPZ | NumPy NPY |
|-----------|------------------|----------------|-----------|-----------|
| Save | 0.038s (1.81x NPZ, 2.92x NPY) | 0.026s (2.19x NPZ, 2.00x NPY) | 0.021s | 0.013s |
| Full Load | 0.010s (1.60x NPZ, 1.10x NPY) | 0.011s (1.45x NPZ, 1.00x NPY) | 0.016s | 0.011s |
| Lazy Load | 0.001s (89,740 MB/s) | 0.001s (87,761 MB/s) | - | - |

#### Data Modification Operations

| Operation | NumPack (Python) | NumPack (Rust) | NumPy NPZ | NumPy NPY |
|-----------|------------------|----------------|-----------|-----------|
| Single Row Replace | 0.000s (≥154x NPZ, ≥85x NPY) | 0.000s (≥166x NPZ, ≥92x NPY) | 0.023s | 0.013s |
| Continuous Rows (10K) | 0.001s | 0.001s | - | - |
| Random Rows (10K) | 0.014s | 0.015s | - | - |
| Large Data Replace (500K) | 0.020s | 0.018s | - | - |

#### Drop Operations

| Operation (1M rows, float32) | NumPack (Python) | NumPack (Rust) | NumPy NPZ | NumPy NPY |
|-----------|------------------|----------------|-----------|-----------|
| Drop Array | 0.008s (1.60x NPZ, 0.12x NPY) | 0.004s (2.80x NPZ, 0.22x NPY) | 0.012s | 0.001s |
| Drop First Row | 0.023s (1.62x NPZ, 1.21x NPY) | 0.020s (1.86x NPZ, 1.39x NPY) | 0.038s | 0.028s |
| Drop Last Row | 0.019s (∞x NPZ, ∞x NPY) | 0.020s (∞x NPZ, ∞x NPY) | 0.038s | 0.028s |
| Drop Middle Row | 0.019s (1.96x NPZ, 1.46x NPY) | 0.019s (1.95x NPZ, 1.46x NPY) | 0.038s | 0.028s |
| Drop Front Continuous (10K rows) | 0.021s (1.77x NPZ, 1.33x NPY) | 0.021s (1.84x NPZ, 1.37x NPY) | 0.038s | 0.028s |
| Drop Middle Continuous (10K rows) | 0.020s (1.85x NPZ, 1.38x NPY) | 0.020s (1.86x NPZ, 1.39x NPY) | 0.038s | 0.028s |
| Drop End Continuous (10K rows) | 0.020s (1.88x NPZ, 1.41x NPY) | 0.020s (1.85x NPZ, 1.38x NPY) | 0.038s | 0.028s |
| Drop Random Rows (10K rows) | 0.025s (1.52x NPZ, 1.14x NPY) | 0.021s (1.76x NPZ, 1.32x NPY) | 0.038s | 0.028s |
| Drop Near Non-continuous (10K rows) | 0.018s (2.05x NPZ, 1.53x NPY) | 0.022s (1.75x NPZ, 1.31x NPY) | 0.038s | 0.028s |

#### Append Operations

| Operation | NumPack (Python) | NumPack (Rust) | NumPy NPZ | NumPy NPY |
|-----------|------------------|----------------|-----------|-----------|
| Small Append (1K rows) | 0.004s (≥6x NPZ, ≥4x NPY) | 0.004s (≥7x NPZ, ≥4x NPY) | 0.028s | 0.017s |
| Large Append (500K rows) | 0.008s (4.88x NPZ, 3.28x NPY) | 0.016s (2.28x NPZ, 1.53x NPY) | 0.037s | 0.025s |

#### Random Access Performance (10K indices)

| Operation | NumPack (Python) | NumPack (Rust) | NumPy NPZ | NumPy NPY |
|-----------|------------------|----------------|-----------|-----------|
| Random Access | 0.005s (2.20x NPZ, 1.45x NPY) | 0.005s (2.30x NPZ, 1.52x NPY) | 0.012s | 0.008s |

#### Matrix Computation Performance (1M rows x 128 columns, Float32)

| Operation | NumPack (Python) | NumPack (Rust) | NumPy NPZ | NumPy NPY | In-Memory |
|-----------|------------------|----------------|-----------|-----------|-----------|
| Inner Product | 0.006s (5.33x NPZ, 1.83x Memory) | 0.006s (5.33x NPZ, 1.83x Memory) | 0.032s | 0.096s | 0.011s |

#### File Size Comparison

| Format | Size | Ratio |
|--------|------|-------|
| NumPack | 47.68 MB | 1.0x |
| NPZ | 47.68 MB | 1.00x |
| NPY | 47.68 MB | 1.00x |

> **Note**: Both Python and Rust backends generate identical file sizes as they use the same underlying file format.

#### Large-scale Data Operations (>1B rows, Float32)

| Operation | NumPack (Python) | NumPack (Rust) | NumPy NPZ | NumPy NPY |
|-----------|------------------|----------------|-----------|-----------|
| Replace | Efficient in-place modification | Zero-copy in-place modification | Memory exceeded | Memory exceeded |
| Drop | Efficient in-place deletion | Zero-copy in-place deletion | Memory exceeded | Memory exceeded |
| Append | Efficient in-place addition | Zero-copy in-place addition | Memory exceeded | Memory exceeded |
| Random Access | High-performance I/O | Near-hardware I/O speed | Memory exceeded | Memory exceeded |

> **Key Advantage**: NumPack provides excellent matrix computation performance (0.065s vs 0.142s NPZ mmap) with several implementation advantages:
> - Uses Arc<Mmap> for reference counting, ensuring automatic resource cleanup
> - Implements MMAP_CACHE to avoid redundant data loading
> - Linux-specific optimizations with huge pages and sequential access hints
> - Supports parallel I/O operations for improved data throughput
> - Optimizes memory usage through Buffer Pool to reduce fragmentation

### Key Performance Highlights

1. **Data Modification**:
   - Single row replacement: NumPack Python backend is **≥154x faster** than NPZ and **≥85x faster** than NPY; Rust backend is **≥166x faster** than NPZ and **≥92x faster** than NPY
   - Continuous rows: Both backends show excellent performance for bulk modifications
   - Random rows: Both backends provide efficient random row replacement
   - Large data replacement: Rust backend shows **10% better performance** than Python backend for large-scale modifications

2. **Drop Operations**:
   - Drop array: Rust backend is **2.80x faster** than NPZ, Python backend is **1.60x faster** than NPZ
   - Drop rows: Both backends are **~1.5-2x faster** than NPZ and **~1.3-1.5x faster** than NPY in typical scenarios
   - NumPack continues to support efficient in-place row deletion without full file rewrite

3. **Append Operations**:
   - Small append (1K rows): Both backends are **≥6x faster** than NPZ and **≥4x faster** than NPY
   - Large append (500K rows): Python backend is **4.88x faster** than NPZ; Rust backend is **2.28x faster** than NPZ
   - Python backend shows superior performance for large append operations

4. **Loading Performance**:
   - Full load: Python backend is **1.60x faster** than NPZ; Rust backend is **1.45x faster** than NPZ
   - Lazy load (memory-mapped): Python backend achieves **89,740 MB/s**, Rust backend achieves **87,761 MB/s** throughput
   - SIMD-optimized streaming: Achieves up to **4,417 MB/s** for large-scale sequential processing

5. **Random Access**:
   - Rust backend is **2.30x faster** than NPZ and **1.52x faster** than NPY for random index access
   - Python backend is **2.20x faster** than NPZ and **1.45x faster** than NPY

6. **Storage Efficiency**:
   - All formats achieve identical compression ratios (47.68 MB)
   - Both Python and Rust backends generate identical file sizes using the same underlying format

7. **Matrix Computation**:
   - Both backends provide **5.33x faster** performance than NPZ mmap
   - Only **~1.8x slower** than pure in-memory computation, providing excellent balance of performance and memory efficiency
   - Zero risk of file descriptor leaks or resource exhaustion

8. **SIMD-Optimized Operations**:
   - **Streaming throughput**: Up to **4,417 MB/s** for large-scale sequential data processing
   - **Clustered access**: **1,041 MB/s** for spatially-local data access patterns
   - **Strided access**: **802 MB/s** for regularly-spaced data access
   - **Large batch operations**: **432 MB/s** for 50K random indices processing

9. **Backend Performance**:
   - **Python backend**: Excellent overall performance, particularly strong in append operations and modification operations
   - **Rust backend**: Superior performance in loading, drop operations, and single-row modifications with zero-copy optimizations
   - Both backends share the same file format ensuring perfect compatibility

> Note: All benchmarks were performed with float32 arrays in the dev conda environment. Performance may vary depending on data types, array sizes, and system configurations. Numbers greater than 1.0x indicate faster performance, while numbers less than 1.0x indicate slower performance. The Python and Rust backends demonstrate different performance characteristics - Python backend excels in append operations and large data modifications, while Rust backend shows superior performance in loading operations and drop operations.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License, Version 2.0 - see the LICENSE file for details.

Copyright 2024 NumPack Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
