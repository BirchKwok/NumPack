# NumPack

NumPack is a high-performance array storage and manipulation library designed to efficiently handle large NumPy arrays. Built with Rust for performance and exposed to Python through PyO3, NumPack provides a seamless interface for storing, loading, and manipulating large numerical arrays with better performance compared to traditional NumPy storage methods.

## Features

- **High Performance**: Optimized for both reading and writing large numerical arrays
- **Memory Mapping Support**: Efficient memory usage through memory mapping capabilities
- **Selective Loading**: Load only the arrays you need, when you need them
- **In-place Operations**: Support for in-place array modifications without full file rewrite
- **Parallel I/O**: Utilizes parallel processing for improved performance
- **Multiple Data Types**: Supports various numerical data types including:
  - Boolean
  - Unsigned integers (8-bit to 64-bit)
  - Signed integers (8-bit to 64-bit)
  - Floating point (32-bit and 64-bit)

## Installation

```bash
pip install numpack
```

## Requirements

- Python >= 3.9
- NumPy

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

# Memory mapping mode for large arrays
with npk.mmap_mode() as mmap_npk:
   # Access specific arrays
   array1 = mmap_npk.load('array1')
   array2 = mmap_npk.load('array2')
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

### Memory Mapping Mode

For large arrays, memory mapping mode provides more efficient memory usage:

```python
# Using memory mapping mode
with npk.mmap_mode() as mmap_npk:
    # Access specific arrays
    array1 = mmap_npk.load('array1')  # Array is not fully loaded into memory
    array2 = mmap_npk.load('array2')
    
    # Perform operations on memory-mapped arrays
    result = array1[0:1000] + array2[0:1000]
```

## Performance

NumPack offers significant performance improvements compared to traditional NumPy storage methods, especially in data modification operations and random access. Below are detailed benchmark results:

### Benchmark Results

The following benchmarks were performed on an MacBook Pro (Apple Silicon) with arrays of size 1M x 10 and 500K x 5 (float32).

#### Storage Operations

| Operation | NumPack | NumPy NPZ | NumPy NPY |
|-----------|---------|-----------|-----------|
| Save | 0.014s (0.86x NPZ, 0.57x NPY) | 0.012s | 0.008s |
| Full Load | 0.008s (1.63x NPZ, 0.88x NPY) | 0.013s | 0.007s |
| Selective Load | 0.006s (1.50x NPZ, -) | 0.009s | - |
| Mmap Load | 0.006s (2.00x NPZ, 0.67x NPY) | 0.012s | 0.004s |

#### Data Modification Operations

| Operation | NumPack | NumPy NPZ | NumPy NPY |
|-----------|---------|-----------|-----------|
| Single Row Replace | 0.000s (19.00x NPZ, 12.00x NPY) | 0.019s | 0.012s |
| Continuous Rows (10K) | 0.001s (20.00x NPZ, 12.00x NPY) | 0.020s | 0.012s |
| Random Rows (10K) | 0.015s (1.33x NPZ, 0.87x NPY) | 0.020s | 0.013s |
| Large Data Replace (500K) | 0.019s (1.00x NPZ, 0.74x NPY) | 0.019s | 0.014s |

#### Drop Operations

| Operation (1M rows, float32) | NumPack | NumPy NPZ | NumPy NPY |
|-----------|---------|-----------|-----------|
| Drop Array | 0.001s (22.00x NPZ, 1.00x NPY) | 0.022s | 0.001s |
| Drop First Row | 0.014s (3.21x NPZ, 1.93x NPY) | 0.045s | 0.027s |
| Drop Last Row | 0.000s (∞x NPZ, ∞x NPY) | 0.045s | 0.027s |
| Drop Middle Row | 0.014s (3.21x NPZ, 1.93x NPY) | 0.045s | 0.027s |
| Drop Front Continuous (10K rows) | 0.016s (2.81x NPZ, 1.69x NPY) | 0.045s | 0.027s |
| Drop Middle Continuous (10K rows) | 0.016s (2.81x NPZ, 1.69x NPY) | 0.045s | 0.027s |
| Drop End Continuous (10K rows) | 0.001s (45.00x NPZ, 27.00x NPY) | 0.045s | 0.027s |
| Drop Random Rows (10K rows) | 0.018s (2.50x NPZ, 1.50x NPY) | 0.045s | 0.027s |
| Drop Near Non-continuous (10K rows) | 0.015s (3.00x NPZ, 1.80x NPY) | 0.045s | 0.027s |

#### Append Operations

| Operation | NumPack | NumPy NPZ | NumPy NPY |
|-----------|---------|-----------|-----------|
| Small Append (1K rows) | 0.000s (22.00x NPZ, 18.00x NPY) | 0.022s | 0.018s |
| Large Append (500K rows) | 0.003s (9.67x NPZ, 6.67x NPY) | 0.029s | 0.020s |

#### Random Access Performance (10K indices)

| Operation | NumPack | NumPy NPZ | NumPy NPY |
|-----------|---------|-----------|-----------|
| Random Access | 0.008s (2.00x NPZ, 1.38x NPY) | 0.016s | 0.011s |

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

#### Matrix Computation Performance (1M rows x 128 columns, Float32)

| Operation | NumPack | NumPy NPZ | NumPy NPY | In-Memory |
|-----------|---------|-----------|-----------|-----------|
| Inner Product | 0.019s (6.58x NPZ, 1.00x NPY) | 0.125s | 0.019s | 0.011s |
| Other calculations are similar to the above case | ... | ... | ... | ... |

> **Key Advantage**: NumPack achieves the same performance as NumPy's NPY mmap (0.019s) for matrix computations, with several implementation advantages:
> - Uses Arc<Mmap> for reference counting, ensuring automatic resource cleanup
> - Implements MMAP_CACHE to avoid redundant data loading
> - Linux-specific optimizations with huge pages and sequential access hints
> - Supports parallel I/O operations for improved data throughput
> - Optimizes memory usage through Buffer Pool to reduce fragmentation

### Key Performance Highlights

1. **Data Modification**:
   - Single row replacement: NumPack is **19x faster** than NPZ and **12x faster** than NPY
   - Continuous rows: NumPack is **20x faster** than NPZ and **12x faster** than NPY
   - Random rows: NumPack is **1.33x faster** than NPZ but **0.87x slower** than NPY
   - Large data replacement: NumPack is comparable to NPZ but **0.74x slower** than NPY

2. **Drop Operations**:
   - Drop array: NumPack is **22x faster** than NPZ and comparable to NPY
   - Drop rows: NumPack is currently **0.61x slower** than NPZ and **0.41x slower** than NPY
   - NumPack provides efficient in-place row deletion without full file rewrite

3. **Append Operations**:
   - Small append (1K rows): NumPack is **22x faster** than NPZ and **18x faster** than NPY
   - Large append (500K rows): NumPack is **9.67x faster** than NPZ and **6.67x faster** than NPY
   - NumPack excels at both small and large append operations

4. **Loading Performance**:
   - Full load: NumPack is **1.63x faster** than NPZ but **0.88x slower** than NPY
   - Memory-mapped load: NumPack is **2.00x faster** than NPZ but **0.67x slower** than NPY
   - Selective load: NumPack is **1.50x faster** than NPZ

5. **Random Access**:
   - NumPack is **2.00x faster** than NPZ and **1.38x faster** than NPY for random index access

6. **Storage Efficiency**:
   - All formats achieve identical compression ratios (47.68 MB)
   - NumPack maintains high performance while keeping file sizes competitive

7. **Matrix Computation**:
   - NumPack matches NPY mmap performance while providing better resource management
   - **6.58x faster** than NPZ mmap for matrix operations
   - Only 1.72x slower than pure in-memory computation
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
