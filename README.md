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

NumPack offers significant performance improvements compared to traditional NumPy storage methods:

- Faster saving and loading of large arrays
- Efficient memory usage through memory mapping
- Better performance for in-place modifications
- Optimized random access operations

### Benchmark Results

The following benchmarks were performed on a 2021 M1 Pro MacBook Pro (32GB Memory):

#### Test Configuration
- Array 1: (1,000,000 × 10) float32
- Array 2: (500,000 × 5) float32
- Total Data Size: ~47.68 MB

#### Operation Performance (Time in seconds)

| Operation | NumPack | NumPy (npz) | NumPy (npy) | Speedup vs npz | Speedup vs npy |
|-----------|---------|-------------|-------------|----------------|----------------|
| Save | 0.03 | 0.02 | 0.02 | 0.56x | 0.47x |
| Full Load | 0.00 | 0.02 | 0.02 | ∞ | ∞ |
| Selective Load | 0.02 | 0.01 | - | 0.82x | - |
| Mmap Load | 0.00 | 0.02 | 0.01 | ∞ | ∞ |
| Replace (Single Row) | 0.00 | 0.03 | 0.02 | 100x | 100x |
| Replace (Continuous Rows) | 0.00 | 0.03 | 0.02 | 25x | 16.7x |
| Replace (Random Rows) | 0.01 | 0.03 | 0.02 | 4.76x | 2.94x |
| Replace (Large Data) | 0.01 | 0.03 | 0.02 | 1.89x | 1.22x |
| Random Access | 0.01 | 0.02 | 0.02 | 1.45x | 1.25x |
| Append | 0.01 | 0.02 | - | 1.14x | - |

#### File Size Comparison

| Format | Size |
|--------|------|
| NumPack | 47.68 MB |
| NumPy (npz) | 47.68 MB |
| NumPy (npy) | 47.68 MB |

#### Key Findings

1. **Loading Performance**: NumPack shows exceptional performance in full load and memory-mapped load operations, with near-instantaneous loading times.
2. **Replace Operations**: NumPack significantly outperforms traditional NumPy formats in all replacement scenarios:
   - Up to 100x faster for single row replacements
   - 25x faster for continuous row replacements
   - 4.76x faster for random row replacements
3. **Storage Efficiency**: NumPack maintains the same file size as traditional NumPy formats while providing better performance characteristics.
4. **Random Access**: NumPack provides up to 1.45x faster random access compared to NumPy formats.

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
