# NumPack API Documentation

Welcome to the NumPack API documentation! NumPack is a high-performance array storage library that combines Rust's performance with Python's ease of use.

## Documentation Index

### Getting Started

- **[01. Getting Started Guide](./01_getting_started.md)**
  - Installation instructions
  - Quick start examples
  - Basic concepts and usage patterns
  - Context manager and file management
  - Supported data types

### API Reference

- **[API Reference (Detailed)](./api_reference/README.md)** ⭐ NEW
  - Complete function-level documentation
  - Parameters, return values, and examples
  - Organized by module (Core, IO, Utils)

- **[02. Core Operations](./02_core_operations.md)**
  - Complete API reference for all basic operations
  - `save()`, `load()`, `replace()`, `append()`, `drop()`
  - Random access with `getitem()`
  - Metadata operations
  - Stream loading
  - File management

- **[03. Batch Processing](./03_batch_processing.md)**
  - High-performance batch modes (25-174x speedup)
  - `batch_mode()`: Memory-cached processing
  - `writable_batch_mode()`: Zero-copy file mapping
  - Comparison and selection guide
  - Best practices and examples

- **[04. Advanced Features](./04_advanced_features.md)**
  - Lazy arrays and memory-mapped loading
  - Streaming operations for large datasets
  - Advanced indexing and slicing
  - In-place operations
  - Memory management strategies
  - Cross-platform considerations

### Optimization

- **[05. Performance Guide](./05_performance_guide.md)**
  - Comprehensive benchmark results
  - Optimization strategies
  - Common performance pitfalls
  - Platform-specific optimizations
  - Real-world use case examples

### Quick Reference

- **[06. Quick Reference](./06_quick_reference.md)**
  - API cheatsheet
  - Common patterns
  - Decision trees for choosing the right approach
  - Troubleshooting guide

### Format Conversion

- **[07. IO Conversion](./07_io_conversion.md)**
  - PyTorch tensor conversion
  - PyArrow/Feather/Parquet conversion
  - SafeTensors conversion
  - NumPy, HDF5, Zarr, CSV conversion
  - Memory ↔ .npk and File ↔ .npk patterns

---

## Quick Links

### By Use Case

| Use Case | Documentation | Key Features |
|----------|--------------|--------------|
| **First-time users** | [Getting Started](./01_getting_started.md) | Installation, basic usage |
| **API lookup** | [Core Operations](./02_core_operations.md) | Complete API reference |
| **Performance optimization** | [Batch Processing](./03_batch_processing.md), [Performance Guide](./05_performance_guide.md) | 25-174x speedup |
| **Large datasets** | [Advanced Features](./04_advanced_features.md) | Lazy loading, streaming |
| **Quick answers** | [Quick Reference](./06_quick_reference.md) | Cheatsheet, common patterns |
| **Format conversion** | [IO Conversion](./07_io_conversion.md) | PyTorch, Arrow, Parquet, SafeTensors |

### By Feature

| Feature | Documentation | Performance Gain |
|---------|--------------|------------------|
| **Batch modifications** | [Batch Processing](./03_batch_processing.md) | 25-174x faster |
| **Row replacement** | [Core Operations](./02_core_operations.md) | 397x faster than NPY |
| **Data append** | [Core Operations](./02_core_operations.md) | 405x faster than NPY |
| **Lazy loading** | [Advanced Features](./04_advanced_features.md) | 54x faster initialization |
| **Streaming** | [Advanced Features](./04_advanced_features.md) | Memory-efficient |

---

## Performance Highlights

NumPack excels in three key areas:

### 1. Data Modification (397-405x faster)
```python
# Replace 100 rows: 0.047ms (NPY: 18.51ms)
npk.replace({'features': new_data}, [0, 1, 2, ...])

# Append 100 rows: 0.067ms (NPY: 27.09ms)
npk.append({'features': new_data})
```

### 2. Lazy Loading (54x faster)
```python
# Initialization: 0.002ms (NPY mmap: 0.107ms)
lazy_arr = npk.load('features', lazy=True)
subset = lazy_arr[1000:2000]
```

### 3. Batch Processing (25-174x faster)
```python
# 100 modifications: 4.9ms (Normal: 856ms)
with npk.writable_batch_mode() as wb:
    arr = wb.load('features')
    for i in range(100):
        arr *= 1.1
```

---

## Common Tasks

### Save and Load Data
```python
from numpack import NumPack
import numpy as np

with NumPack("data.npk") as npk:
    # Save
    npk.save({'features': np.random.rand(1000, 100)})
    
    # Load
    features = npk.load('features')
```
[Full documentation](./02_core_operations.md#save-operations)

### Modify Existing Data
```python
with NumPack("data.npk") as npk:
    # Replace specific rows (397x faster than NPY)
    npk.replace({'features': new_data}, [0, 1, 2])
    
    # Append new rows (405x faster than NPY)
    npk.append({'features': more_data})
```
[Full documentation](./02_core_operations.md#replace-operations)

### High-Performance Batch Processing
```python
# For frequent modifications (174x speedup)
with NumPack("data.npk") as npk:
    with npk.writable_batch_mode() as wb:
        arr = wb.load('features')
        arr *= 2.0  # Direct file modification
```
[Full documentation](./03_batch_processing.md)

### Large Dataset Handling
```python
# Lazy loading (54x faster initialization)
with NumPack("large_data.npk") as npk:
    lazy_arr = npk.load('features', lazy=True)
    subset = lazy_arr[1000:2000]

# Streaming (memory-efficient)
with NumPack("large_data.npk") as npk:
    for batch in npk.stream_load('features', buffer_size=10000):
        process(batch)
```
[Full documentation](./04_advanced_features.md)

---

## Installation

### From PyPI (Recommended)
```bash
pip install numpack
```

**Requirements:**
- Python >= 3.9
- NumPy >= 1.26.0

### From Source
```bash
git clone https://github.com/BirchKwok/NumPack.git
cd NumPack
pip install maturin>=1.0,<2.0
maturin develop
```

**Additional requirements:**
- Rust >= 1.70.0
- Appropriate C/C++ compiler

[Full installation guide](./01_getting_started.md#installation)

---

## When to Use NumPack

### Strongly Recommended (90% of use cases)

- Machine learning and deep learning pipelines
- Real-time data stream processing
- Data annotation and correction workflows
- Feature stores with dynamic updates
- Any scenario requiring frequent data modifications
- Fast data loading requirements

### Consider Alternatives (10% of use cases)

- Write-once, never modify → Use NPY (2.2x faster initial write)
- Frequent single-row random access → Use NPY mmap
- Extreme compression requirements → Use NPZ (10% smaller, 1000x slower)

[Performance comparison](./05_performance_guide.md#performance-overview)

---

## API Overview

### Core Classes

```python
from numpack import NumPack, LazyArray

# Main class
npk = NumPack("data.npk")

# Lazy array (memory-mapped)
lazy_arr = npk.load('array', lazy=True)
```

### Key Methods

| Method | Purpose | Performance |
|--------|---------|-------------|
| `save(arrays)` | Save arrays | 2.2x slower than NPY |
| `load(name, lazy=False)` | Load array | 1.3x faster (eager), 54x faster (lazy) |
| `replace(arrays, indexes)` | Replace rows | **397x faster than NPY** |
| `append(arrays)` | Append rows | **405x faster than NPY** |
| `drop(name, indexes)` | Drop arrays/rows | Very fast (logical) |
| `getitem(name, indexes)` | Random access | Fast |
| `stream_load(name, buffer_size)` | Stream batches | Memory-efficient |
| `batch_mode()` | Batch processing | **25-37x speedup** |
| `writable_batch_mode()` | Zero-copy batch | **174x speedup** |

[Complete API reference](./02_core_operations.md)

---

## Learning Path

### Beginner
1. Read [Getting Started Guide](./01_getting_started.md)
2. Try basic examples from [Core Operations](./02_core_operations.md)
3. Learn about context managers and file management

### Intermediate
1. Explore [Batch Processing](./03_batch_processing.md) for performance gains
2. Learn when to use batch_mode vs writable_batch_mode
3. Study [Advanced Features](./04_advanced_features.md) for lazy loading

### Advanced
1. Master [Performance Guide](./05_performance_guide.md)
2. Optimize your specific use case
3. Understand platform-specific optimizations

---

## Troubleshooting

### Common Issues

**Q: File handle warning on Windows?**
- Use context manager: `with NumPack(...) as npk:`
- See [Getting Started](./01_getting_started.md#windows-file-handle-warning)

**Q: Out of memory errors?**
- Use lazy loading or streaming
- See [Advanced Features](./04_advanced_features.md#memory-management)

**Q: Slow performance?**
- Use appropriate batch mode
- See [Performance Guide](./05_performance_guide.md#common-performance-pitfalls)

**Q: Need to reclaim disk space after deletions?**
- Call `npk.update(array_name)` to compact
- See [Core Operations](./02_core_operations.md#update)

[Full troubleshooting guide](./06_quick_reference.md#troubleshooting)

---

## Examples Repository

All documentation includes practical examples. For complete working examples, see:

- `examples/inplace_operators_example.py`
- `examples/writable_batch_mode_example.py`
- `examples/drop_operations_example.py`

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the Apache License, Version 2.0.

Copyright 2025 NumPack Contributors

---

## 🔗 Additional Resources

- [GitHub Repository](https://github.com/BirchKwok/NumPack)
- [PyPI Package](https://pypi.org/project/numpack/)
- [Issue Tracker](https://github.com/BirchKwok/NumPack/issues)

---


