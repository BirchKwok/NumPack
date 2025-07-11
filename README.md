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

| Operation | NumPack | NumPy NPZ | NumPy NPY |
|-----------|---------|-----------|-----------|
| Save | 0.015s (0.93x NPZ, 0.53x NPY) | 0.014s | 0.008s |
| Full Load | 0.008s (1.75x NPZ, 1.00x NPY) | 0.014s | 0.008s |
| Selective Load | 0.006s (1.67x NPZ, -) | 0.010s | - |

#### Data Modification Operations

| Operation | NumPack | NumPy NPZ | NumPy NPY |
|-----------|---------|-----------|-----------|
| Single Row Replace | 0.000s (≥40x NPZ, ≥30x NPY) | 0.022s | 0.015s |
| Continuous Rows (10K) | 0.001s (24.00x NPZ, 14.00x NPY) | 0.024s | 0.014s |
| Random Rows (10K) | 0.015s (1.53x NPZ, 0.93x NPY) | 0.023s | 0.014s |
| Large Data Replace (500K) | 0.020s (1.10x NPZ, 0.75x NPY) | 0.022s | 0.015s |

#### Drop Operations

| Operation (1M rows, float32) | NumPack | NumPy NPZ | NumPy NPY |
|-----------|---------|-----------|-----------|
| Drop Array | 0.001s (24.00x NPZ, 1.00x NPY) | 0.024s | 0.001s |
| Drop First Row | 0.014s (3.29x NPZ, 1.93x NPY) | 0.046s | 0.027s |
| Drop Last Row | 0.000s (∞x NPZ, ∞x NPY) | 0.046s | 0.027s |
| Drop Middle Row | 0.014s (3.29x NPZ, 1.93x NPY) | 0.046s | 0.027s |
| Drop Front Continuous (10K rows) | 0.015s (3.07x NPZ, 1.80x NPY) | 0.046s | 0.027s |
| Drop Middle Continuous (10K rows) | 0.015s (3.07x NPZ, 1.80x NPY) | 0.046s | 0.027s |
| Drop End Continuous (10K rows) | 0.001s (46.00x NPZ, 27.00x NPY) | 0.046s | 0.027s |
| Drop Random Rows (10K rows) | 0.018s (2.56x NPZ, 1.50x NPY) | 0.046s | 0.027s |
| Drop Near Non-continuous (10K rows) | 0.015s (3.07x NPZ, 1.80x NPY) | 0.046s | 0.027s |

#### Append Operations

| Operation | NumPack | NumPy NPZ | NumPy NPY |
|-----------|---------|-----------|-----------|
| Small Append (1K rows) | 0.000s (≥80x NPZ, ≥60x NPY) | 0.025s | 0.017s |
| Large Append (500K rows) | 0.003s (11.33x NPZ, 7.67x NPY) | 0.034s | 0.023s |

#### Random Access Performance (10K indices)

| Operation | NumPack | NumPy NPZ | NumPy NPY |
|-----------|---------|-----------|-----------|
| Random Access | 0.008s (2.38x NPZ, 1.50x NPY) | 0.019s | 0.012s |

#### Matrix Computation Performance (1M rows x 128 columns, Float32)

| Operation | NumPack | NumPy NPZ | NumPy NPY | In-Memory |
|-----------|---------|-----------|-----------|-----------|
| Inner Product | 0.021s (6.62x NPZ, 1.05x NPY) | 0.139s | 0.022s | 0.011s |

#### File Size Comparison

| Format | Size | Ratio |
|--------|------|-------|
| NumPack | 47.68 MB | 1.0x |
| NPZ | 47.68 MB | 1.0x |
| NPY | 47.68 MB | 1.0x |

#### Large-scale Data Operations (>1B rows, Float32)

| Operation | NumPack | NumPy NPZ | NumPy NPY |
|-----------|---------|-----------|-----------|
| Replace | Zero-copy in-place modification | Memory exceeded | Memory exceeded |
| Drop | Zero-copy in-place deletion | Memory exceeded | Memory exceeded |
| Append | Zero-copy in-place addition | Memory exceeded | Memory exceeded |
| Random Access | Near-hardware I/O speed | Memory exceeded | Memory exceeded |

> **Key Advantage**: NumPack achieves the same performance as NumPy's NPY mmap (0.021s) for matrix computations, with several implementation advantages:
> - Uses Arc<Mmap> for reference counting, ensuring automatic resource cleanup
> - Implements MMAP_CACHE to avoid redundant data loading
> - Linux-specific optimizations with huge pages and sequential access hints
> - Supports parallel I/O operations for improved data throughput
> - Optimizes memory usage through Buffer Pool to reduce fragmentation

### Key Performance Highlights

1. **Data Modification**:
   - Single row replacement: NumPack is **≥40x faster** than NPZ and **≥30x faster** than NPY
   - Continuous rows: NumPack is **24x faster** than NPZ and **14x faster** than NPY
   - Random rows: NumPack is **1.53x faster** than NPZ but **0.93x slower** than NPY
   - Large data replacement: NumPack is **1.10x faster** than NPZ but **0.75x slower** than NPY

2. **Drop Operations**:
   - Drop array: NumPack is **24x faster** than NPZ and comparable to NPY
   - Drop rows: NumPack is now **~3x faster** than NPZ and **~2x faster** than NPY in typical scenarios
   - NumPack continues to support efficient in-place row deletion without full file rewrite

3. **Append Operations**:
   - Small append (1K rows): NumPack is **≥80x faster** than NPZ and **≥60x faster** than NPY
   - Large append (500K rows): NumPack is **11x faster** than NPZ and **8x faster** than NPY
   - Performance improvements in append operations are attributed to optimized buffer management

4. **Loading Performance**:
   - Full load: NumPack is **1.75x faster** than NPZ and on par with NPY
   - Lazy load (memory-mapped): NumPack is **~2.0x faster** than NPZ mmap and close to NPY mmap
   - Selective load: NumPack is **1.67x faster** than NPZ

5. **Random Access**:
   - NumPack is **2.38x faster** than NPZ and **1.50x faster** than NPY for random index access

6. **Storage Efficiency**:
   - All formats achieve identical compression ratios (47.68 MB)
   - NumPack maintains high performance while keeping file sizes competitive

7. **Matrix Computation**:
   - NumPack remains on par with NPY mmap performance while providing better resource management
   - **6.62x faster** than NPZ mmap for matrix operations
   - Only 1.91x slower than pure in-memory computation
   - Zero risk of file descriptor leaks or resource exhaustion

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
