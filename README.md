# NumPack

NumPack is a lightning-fast array manipulation engine that revolutionizes how you handle large-scale NumPy arrays. By combining Rust's raw performance with Python's ease of use, NumPack delivers up to 20x faster operations than traditional methods, while using minimal memory. Whether you're working with gigabyte-sized matrices or performing millions of array operations, NumPack makes it effortless with its zero-copy architecture and intelligent memory management.

Key highlights:
- 🚀 Up to 20x faster than traditional NumPy storage methods
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
  - Floating point (32-bit and 64-bit)

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

The following benchmarks were performed on an MacBook Pro (Apple Silicon) with arrays of size 1M x 10 and 500K x 5 (float32).

#### Storage Operations

| Operation | NumPack (Best) | NumPack (Python) | NumPy NPZ | NumPy NPY |
|-----------|----------------|------------------|-----------|-----------|
| Save | 0.015s (0.96x NPZ, 0.55x NPY) | 0.015s (0.96x NPZ, 0.55x NPY) | 0.014s | 0.008s |
| Full Load | 0.007s (1.89x NPZ, 1.02x NPY) | 0.009s (1.56x NPZ, 0.85x NPY) | 0.014s | 0.008s |
| Selective Load | 0.006s (1.61x NPZ, -) | 0.006s (1.61x NPZ, -) | 0.010s | - |

#### Data Modification Operations

| Operation | NumPack (Best) | NumPack (Python) | NumPy NPZ | NumPy NPY |
|-----------|----------------|------------------|-----------|-----------|
| Single Row Replace | 0.000s (≥156x NPZ, ≥90x NPY) | 0.000s (≥156x NPZ, ≥90x NPY) | 0.024s | 0.014s |
| Continuous Rows (10K) | 0.001s (24.00x NPZ, 14.00x NPY) | 0.001s (24.00x NPZ, 14.00x NPY) | 0.024s | 0.014s |
| Random Rows (10K) | 0.014s (1.71x NPZ, 1.00x NPY) | 0.014s (1.71x NPZ, 1.00x NPY) | 0.024s | 0.014s |
| Large Data Replace (500K) | 0.018s (1.33x NPZ, 0.78x NPY) | 0.020s (1.20x NPZ, 0.70x NPY) | 0.024s | 0.014s |

#### Drop Operations

| Operation (1M rows, float32) | NumPack (Best) | NumPack (Python) | NumPy NPZ | NumPy NPY |
|-----------|----------------|------------------|-----------|-----------|
| Drop Array | 0.005s (2.60x NPZ, 0.16x NPY) | 0.007s (1.91x NPZ, 0.12x NPY) | 0.013s | 0.001s |
| Drop First Row | 0.019s (2.10x NPZ, 1.50x NPY) | 0.023s (1.67x NPZ, 1.20x NPY) | 0.039s | 0.028s |
| Drop Last Row | 0.019s (∞x NPZ, ∞x NPY) | 0.020s (∞x NPZ, ∞x NPY) | 0.039s | 0.028s |
| Drop Middle Row | 0.020s (2.00x NPZ, 1.43x NPY) | 0.020s (2.00x NPZ, 1.43x NPY) | 0.039s | 0.028s |
| Drop Front Continuous (10K rows) | 0.018s (2.11x NPZ, 1.52x NPY) | 0.021s (1.90x NPZ, 1.36x NPY) | 0.039s | 0.028s |
| Drop Middle Continuous (10K rows) | 0.019s (2.04x NPZ, 1.46x NPY) | 0.019s (2.04x NPZ, 1.46x NPY) | 0.039s | 0.028s |
| Drop End Continuous (10K rows) | 0.021s (1.87x NPZ, 1.34x NPY) | 0.021s (1.87x NPZ, 1.34x NPY) | 0.039s | 0.028s |
| Drop Random Rows (10K rows) | 0.022s (1.77x NPZ, 1.27x NPY) | 0.024s (1.63x NPZ, 1.17x NPY) | 0.039s | 0.028s |
| Drop Near Non-continuous (10K rows) | 0.021s (1.82x NPZ, 1.31x NPY) | 0.021s (1.82x NPZ, 1.31x NPY) | 0.039s | 0.028s |

#### Append Operations

| Operation | NumPack (Best) | NumPack (Python) | NumPy NPZ | NumPy NPY |
|-----------|----------------|------------------|-----------|-----------|
| Small Append (1K rows) | 0.004s (≥6x NPZ, ≥4x NPY) | 0.006s (≥5x NPZ, ≥3x NPY) | 0.028s | 0.018s |
| Large Append (500K rows) | 0.008s (4.86x NPZ, 3.06x NPY) | 0.008s (4.86x NPZ, 3.06x NPY) | 0.037s | 0.024s |

#### Random Access Performance (10K indices)

| Operation | NumPack (Best) | NumPack (Python) | NumPy NPZ | NumPy NPY |
|-----------|----------------|------------------|-----------|-----------|
| Random Access | 0.005s (2.13x NPZ, 1.54x NPY) | 0.005s (2.13x NPZ, 1.54x NPY) | 0.012s | 0.008s |

#### Matrix Computation Performance (1M rows x 128 columns, Float32)

| Operation | NumPack (Best) | NumPack (Python) | NumPy NPZ | NumPy NPY | In-Memory |
|-----------|----------------|------------------|-----------|-----------|-----------|
| Inner Product | 0.066s (2.18x NPZ, 6.23x Memory) | 0.066s (2.18x NPZ, 6.23x Memory) | 0.144s | 0.096s | 0.011s |

#### File Size Comparison

| Format | Size | Ratio |
|--------|------|-------|
| NumPack | 47.68 MB | 1.0x |
| NPZ | 47.68 MB | 1.0x |
| NPY | 47.68 MB | 1.0x |

> **Note**: Both Python and Rust backends generate identical file sizes as they use the same underlying file format.

#### Large-scale Data Operations (>1B rows, Float32)

| Operation | NumPack (Best) | NumPack (Python) | NumPy NPZ | NumPy NPY |
|-----------|----------------|------------------|-----------|-----------|
| Replace | Zero-copy in-place modification | Efficient in-place modification | Memory exceeded | Memory exceeded |
| Drop | Zero-copy in-place deletion | Efficient in-place deletion | Memory exceeded | Memory exceeded |
| Append | Zero-copy in-place addition | Efficient in-place addition | Memory exceeded | Memory exceeded |
| Random Access | Near-hardware I/O speed | High-performance I/O | Memory exceeded | Memory exceeded |

> **Key Advantage**: NumPack provides excellent matrix computation performance (0.066s vs 0.144s NPZ mmap) with several implementation advantages:
> - Uses Arc<Mmap> for reference counting, ensuring automatic resource cleanup
> - Implements MMAP_CACHE to avoid redundant data loading
> - Linux-specific optimizations with huge pages and sequential access hints
> - Supports parallel I/O operations for improved data throughput
> - Optimizes memory usage through Buffer Pool to reduce fragmentation

### Key Performance Highlights

1. **Data Modification**:
   - Single row replacement: NumPack Python backend is **≥156x faster** than NPZ and **≥90x faster** than NPY
   - Continuous rows: NumPack is **24x faster** than NPZ and **14x faster** than NPY
   - Random rows: NumPack is **1.71x faster** than NPZ and on par with NPY
   - Large data replacement: NumPack Python backend is **1.20x faster** than NPZ but **0.70x slower** than NPY

2. **Drop Operations**:
   - Drop array: NumPack Python backend is **1.91x faster** than NPZ
   - Drop rows: NumPack Python backend is **~2x faster** than NPZ and **~1.4x faster** than NPY in typical scenarios
   - NumPack continues to support efficient in-place row deletion without full file rewrite

3. **Append Operations**:
   - Small append (1K rows): NumPack Python backend is **≥5x faster** than NPZ and **≥3x faster** than NPY
   - Large append (500K rows): NumPack Python backend is **4.86x faster** than NPZ and **3.06x faster** than NPY
   - Performance improvements in append operations are attributed to optimized buffer management

4. **Loading Performance**:
   - Full load: NumPack Python backend is **1.56x faster** than NPZ and **0.85x slower** than NPY
   - Lazy load (memory-mapped): NumPack provides near-instantaneous loading
   - Selective load: NumPack Python backend is **1.61x faster** than NPZ

5. **Random Access**:
   - NumPack Python backend is **2.13x faster** than NPZ and **1.54x faster** than NPY for random index access

6. **Storage Efficiency**:
   - All formats achieve identical compression ratios (47.68 MB)
   - Both Python and Rust backends generate identical file sizes using the same underlying format

7. **Matrix Computation**:
   - NumPack Python backend provides **2.18x faster** performance than NPZ mmap
   - Only **6.23x slower** than pure in-memory computation, providing excellent balance of performance and memory efficiency
   - Zero risk of file descriptor leaks or resource exhaustion

8. **Backend Performance**:
   - **Python backend**: Excellent overall performance, particularly strong in modification operations
   - **Rust backend**: Optimized for specific use cases, with best-in-class performance for certain operations
   - Both backends share the same file format ensuring perfect compatibility

> Note: All benchmarks were performed with float32 arrays. Performance may vary depending on data types, array sizes, and system configurations. Numbers greater than 1.0x indicate faster performance, while numbers less than 1.0x indicate slower performance.

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
