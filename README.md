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

The following benchmarks were performed on an M1 Pro MacBook Pro (2021, 32GB RAM).

#### Storage Performance (Array Size: 1M x 10 and 500K x 5)

| Operation | NumPack vs NPZ | NumPack vs NPY |
|-----------|---------------|----------------|
| Save      | 1.50x slower  | 1.91x slower   |
| Full Load | 1.29x slower  | 1.97x slower   |
| Mmap Load | **0.73x faster** | 1.82x slower   |

#### Data Modification Performance

| Operation Type        | NumPack vs NPZ | NumPack vs NPY |
|----------------------|----------------|----------------|
| Single Row Replace   | **100x faster**    | **100x faster**    |
| Continuous Rows      | **50x faster**     | **33x faster**     |
| Random Rows          | **7.7x faster**    | **2.6x faster**    |
| Large Data Replace   | **1.8x faster**    | **1.03x faster**   |
| Append              | **1.5x faster**    | -              |

#### Random Access Performance

| Operation | NumPack vs NPZ | NumPack vs NPY |
|-----------|---------------|----------------|
| Random Access | **1.85x faster** | 0.98x faster |

#### File Size Comparison

| Format  | Size    | Ratio |
|---------|---------|-------|
| NumPack | 47.68MB | 1.0x  |
| NPZ     | 47.68MB | 1.0x  |
| NPY     | 47.68MB | 1.0x  |

> Note: All benchmarks were run with NumPy arrays containing float32 data type. Performance may vary depending on the specific use case and data types.

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
