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
loaded = npk.load(mmap_mode=False)

# Memory mapping mode for large arrays
lazy_loaded = npk.load(mmap_mode=True)

# Access specific arrays
array1 = loaded['array1']
array2 = loaded['array2']
```

### Advanced Operations

```python
# Replace specific rows
replacement = np.random.rand(10, 100).astype(np.float32)
npk.replace({'array1': replacement}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Append new arrays
new_arrays = {
    'array3': np.random.rand(200, 100).astype(np.float32)
}
npk.append(new_arrays)

# Drop arrays or specific rows
npk.drop('array1')  # Drop entire array
npk.drop('array2', [0, 1, 2])  # Drop specific rows

# Get metadata
shapes = npk.get_shape()  # Get shapes of all arrays
members = npk.get_member_list()  # Get list of array names
mtime = npk.get_modify_time('array1')  # Get modification time
```

## Performance

NumPack offers significant performance improvements compared to traditional NumPy storage methods, especially in data modification operations and random access. Below are detailed benchmark results:

### Benchmark Results

The following benchmarks were performed on an MacBook Pro (M1, 2020, 32GB Memory) with arrays of size 1M x 10 and 500K x 5 (float32).

#### Storage Operations

| Operation | NumPack | NumPy NPZ | NumPy NPY |
|-----------|---------|-----------|-----------|
| Save | 0.026s (0.50x NPZ, 0.31x NPY) | 0.013s | 0.008s |
| Full Load | 0.009s (1.56x NPZ, 0.89x NPY) | 0.014s | 0.008s |
| Selective Load | 0.006s (1.67x NPZ, -) | 0.010s | - |
| Mmap Load | 0.005s (2.60x NPZ, 0.00x NPY) | 0.013s | 0.000s |

#### Data Modification Operations

| Operation | NumPack | NumPy NPZ | NumPy NPY |
|-----------|---------|-----------|-----------|
| Single Row Replace | 0.000s (24.00x NPZ, 13.00x NPY) | 0.024s | 0.013s |
| Continuous Rows (10K) | 0.001s (23.00x NPZ, 14.00x NPY) | 0.023s | 0.014s |
| Random Rows (10K) | 0.015s (1.60x NPZ, 0.93x NPY) | 0.024s | 0.014s |
| Large Data Replace (500K) | 0.013s (1.85x NPZ, 1.08x NPY) | 0.024s | 0.014s |

#### Drop Operations

| Operation | NumPack | NumPy NPZ | NumPy NPY |
|-----------|---------|-----------|-----------|
| Drop Array | 0.000s (24.00x NPZ, 13.00x NPY) | 0.024s | 0.013s |
| Drop Rows (500K) | 0.015s (1.67x NPZ, 0.93x NPY) | 0.025s | 0.014s |

#### Append Operations

| Operation | NumPack | NumPy NPZ |
|-----------|---------|-----------|
| Append | 0.014s (1.14x NPZ) | 0.016s |

#### Random Access Performance (10K indices)

| Operation | NumPack | NumPy NPZ | NumPy NPY |
|-----------|---------|-----------|-----------|
| Random Access | 0.009s (1.89x NPZ, 1.11x NPY) | 0.017s | 0.010s |

#### File Size Comparison

| Format | Size | Ratio |
|--------|------|-------|
| NumPack | 47.68 MB | 1.0x |
| NPZ | 47.68 MB | 1.0x |
| NPY | 47.68 MB | 1.0x |

### Key Performance Highlights

1. **Data Modification**:
   - Single row replacement: NumPack is **24x faster** than NPZ and **13x faster** than NPY
   - Continuous rows: NumPack is **23x faster** than NPZ and **14x faster** than NPY
   - Random rows: NumPack is **1.6x faster** than NPZ but **0.93x slower** than NPY
   - Large data replacement: NumPack is **1.85x faster** than NPZ and **1.08x faster** than NPY

2. **Drop Operations**:
   - Drop array: NumPack is **24x faster** than NPZ and **13x faster** than NPY
   - Drop rows: NumPack is **1.67x faster** than NPZ but **0.93x slower** than NPY
   - NumPack provides efficient in-place row deletion without full file rewrite

3. **Loading Performance**:
   - Full load: NumPack is **1.56x faster** than NPZ but **0.89x slower** than NPY
   - Memory-mapped load: NumPack is **2.6x faster** than NPZ but slower than NPY
   - Selective load: NumPack is **1.67x faster** than NPZ (NPY不支持)

4. **Random Access**:
   - NumPack is **1.89x faster** than NPZ and **1.11x faster** than NPY for random index access

5. **Storage Efficiency**:
   - All formats achieve identical compression ratios (47.68 MB)
   - NumPack maintains high performance while keeping file sizes competitive

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
