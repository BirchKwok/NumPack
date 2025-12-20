# Quick Reference Guide

This is a quick reference cheatsheet for NumPack. For detailed documentation, see the full API reference.

## 📑 Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [API Cheatsheet](#api-cheatsheet)
- [Decision Trees](#decision-trees)
- [Common Patterns](#common-patterns)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

---

## Installation

```bash
# From PyPI
pip install numpack

# From source
pip install maturin>=1.0,<2.0
git clone https://github.com/BirchKwok/NumPack.git
cd NumPack
maturin develop
```

**Requirements:**
- Python >= 3.9
- NumPy >= 1.26.0

---

## Basic Usage

### Create and Save

```python
import numpy as np
from numpack import NumPack

# Using context manager (recommended)
with NumPack("data.npk") as npk:
    npk.save({'array': np.random.rand(1000, 100)})
```

### Load Data

```python
# Eager loading (full array in memory)
with NumPack("data.npk") as npk:
    arr = npk.load('array')

# Lazy loading (memory-mapped, on-demand)
with NumPack("data.npk") as npk:
    lazy_arr = npk.load('array', lazy=True)
    subset = lazy_arr[100:200]
```

### Modify Data

```python
with NumPack("data.npk") as npk:
    # Replace rows (397x faster than NPY)
    npk.replace({'array': new_data}, [0, 1, 2])
    
    # Append rows (405x faster than NPY)
    npk.append({'array': more_data})
    
    # Drop rows
    npk.drop('array', [0, 1, 2])
```

---

## API Cheatsheet

### Initialization

```python
# Basic
npk = NumPack("file.npk")

# With options
npk = NumPack(
    "file.npk",
    drop_if_exists=True,        # Delete if exists
    strict_context_mode=True,   # Require 'with' statement
    force_gc_on_close=False     # GC on close (default: False)
)
```

### File Management

```python
# Context manager (recommended)
with NumPack("file.npk") as npk:
    # Operations...
    pass

# Manual management
npk = NumPack("file.npk")
npk.open()
# Operations...
npk.close()
```

### Save Operations

```python
# Save single array
npk.save({'array': data})

# Save multiple arrays
npk.save({
    'features': features,
    'labels': labels
})
```

### Load Operations

```python
# Eager load
arr = npk.load('array')

# Lazy load
lazy = npk.load('array', lazy=True)

# Dictionary-style
arr = npk['array']
```

### Modification Operations

```python
# Replace rows
npk.replace({'array': new_data}, [0, 1, 2])
npk.replace({'array': new_data}, slice(0, 10))

# Append rows
npk.append({'array': new_data})

# Drop entire array
npk.drop('array')

# Drop specific rows
npk.drop('array', [0, 1, 2])
```

### Random Access

```python
# Get specific rows
rows = npk.getitem('array', [0, 10, 20])

# Single row
row = npk.getitem('array', 0)
```

### Streaming

```python
# Stream in batches
for batch in npk.stream_load('array', buffer_size=1000):
    process(batch)
```

### Format Conversion

```python
from numpack.io import (
    # Memory ↔ .npk
    from_torch, to_torch,           # PyTorch
    from_arrow, to_arrow,           # PyArrow
    from_parquet_table, to_parquet_table,  # Parquet
    from_safetensors, to_safetensors,      # SafeTensors
    # File ↔ .npk (streaming)
    from_torch_file, to_torch_file,
    from_feather_file, to_feather_file,
    from_parquet_file, to_parquet_file,
    from_safetensors_file, to_safetensors_file,
)

# Memory → .npk
from_torch(tensor, 'output.npk', array_name='embeddings')
from_arrow(table, 'output.npk', array_name='data')

# .npk → Memory
tensor = to_torch('input.npk', array_name='embeddings')
table = to_arrow('input.npk', array_name='data')

# File → .npk (streaming)
from_torch_file('model.pt', 'output.npk')
from_parquet_file('data.parquet', 'output.npk')

# .npk → File (streaming)
to_torch_file('input.npk', 'output.pt')
to_parquet_file('input.npk', 'output.parquet')
```

### Metadata & Array Operations

```python
# Get shape
shape = npk.get_shape('array')

# List all arrays
arrays = npk.get_member_list()

# Check existence
exists = npk.has_array('array')

# Get modification time
timestamp = npk.get_modify_time('array')

# Clone an array
npk.clone('source_array', 'target_array')

# Reset (clear all)
npk.reset()

# Compact after deletions
npk.update('array')

# Get I/O statistics
stats = npk.get_io_stats()
```

### Batch Modes

```python
# Batch mode (25-37x speedup)
with npk.batch_mode():
    for i in range(100):
        arr = npk.load('array')
        arr *= 2
        npk.save({'array': arr})

# Writable batch mode (174x speedup)
with npk.writable_batch_mode() as wb:
    arr = wb.load('array')
    for i in range(100):
        arr *= 2  # Direct modification
```

### LazyArray Operations

```python
lazy = npk.load('array', lazy=True)

# Indexing
row = lazy[0]
rows = lazy[100:200]
selection = lazy[[0, 10, 20]]

# Shape info
shape = lazy.shape
dtype = lazy.dtype
size = lazy.size

# Convert to numpy
numpy_arr = np.asarray(lazy)
```

---

## Decision Trees

### Which Loading Mode?

```
Need to modify data frequently?
├─ Yes
│   ├─ Array > 100MB?
│   │   ├─ Yes → Use writable_batch_mode (174x speedup)
│   │   └─ No → Use batch_mode (25-37x speedup)
│   └─ No (single modification)
│       └─ Use normal replace/append (397-405x faster)
│
└─ No (read-only)
    ├─ Need full array?
    │   ├─ Yes → Use eager load (1.3x faster)
    │   └─ No → Use lazy load (54x faster init)
    └─ Sequential access?
        └─ Use stream_load (memory-efficient)
```

### Which Batch Mode?

```
Need to change array shape?
├─ Yes → Use batch_mode
└─ No
    ├─ Array size > available RAM * 0.5?
    │   ├─ Yes → Use writable_batch_mode
    │   └─ No → Continue
    ├─ Array size > 100MB?
    │   ├─ Yes → Use writable_batch_mode
    │   └─ No → Use batch_mode
```

### Save vs Replace vs Append?

```
What do you want to do?
├─ Create new array or replace entire array
│   └─ Use save()
├─ Modify specific rows
│   └─ Use replace() (397x faster)
├─ Add new rows
│   └─ Use append() (405x faster)
└─ Remove rows
    └─ Use drop()
```

---

## Common Patterns

### Pattern 1: Basic CRUD

```python
with NumPack("data.npk") as npk:
    # Create
    npk.save({'data': np.random.rand(100, 50)})
    
    # Read
    data = npk.load('data')
    
    # Update
    npk.replace({'data': new_data}, [0, 1, 2])
    
    # Delete
    npk.drop('data', [0, 1, 2])
```

### Pattern 2: High-Performance Modifications

```python
with NumPack("data.npk") as npk:
    with npk.writable_batch_mode() as wb:
        arr = wb.load('data')  # Load once
        
        # Multiple modifications (174x faster)
        arr *= 2.0
        arr += 1.0
        arr[arr < 0] = 0
```

### Pattern 3: Memory-Efficient Processing

```python
with NumPack("large.npk") as npk:
    total = 0
    
    # Process in batches
    for batch in npk.stream_load('data', buffer_size=10000):
        total += batch.sum()
    
    mean = total / npk.get_shape('data')[0]
```

### Pattern 4: Lazy Exploration

```python
with NumPack("data.npk") as npk:
    lazy = npk.load('features', lazy=True)
    
    # Quick stats on subset
    sample = lazy[:1000]
    print(f"Mean: {sample.mean()}")
    print(f"Std: {sample.std()}")
```

### Pattern 5: Data Pipeline

```python
with NumPack("input.npk") as src:
    with NumPack("output.npk", drop_if_exists=True) as dst:
        first = True
        
        for batch in src.stream_load('data', buffer_size=10000):
            processed = transform(batch)
            
            if first:
                dst.save({'data': processed})
                first = False
            else:
                dst.append({'data': processed})
```

### Pattern 6: ML Training Loop

```python
with NumPack("train.npk") as npk:
    features = npk.load('features', lazy=True)
    labels = npk.load('labels', lazy=True)
    
    for epoch in range(10):
        for i in range(0, len(features), batch_size):
            X = features[i:i+batch_size]
            y = labels[i:i+batch_size]
            train_step(X, y)
```

### Pattern 7: Interactive Annotation

```python
with NumPack("data.npk") as npk:
    with npk.writable_batch_mode() as wb:
        labels = wb.load('labels')
        samples = npk.load('samples', lazy=True)
        
        for i, sample in enumerate(samples):
            display(sample)
            labels[i] = get_user_input()
            # Auto-saved
```

### Pattern 8: Checkpoint Management

```python
def save_checkpoint(npk, epoch, weights, optimizer):
    with npk.batch_mode():
        for name, w in weights.items():
            npk.save({f'weight_{name}': w})
        for name, s in optimizer.items():
            npk.save({f'optimizer_{name}': s})
        npk.save({'epoch': np.array([epoch])})

def load_checkpoint(npk):
    weights = {}
    optimizer = {}
    for name in npk.get_member_list():
        if name.startswith('weight_'):
            weights[name[7:]] = npk.load(name)
        elif name.startswith('optimizer_'):
            optimizer[name[10:]] = npk.load(name)
    epoch = npk.load('epoch')[0]
    return epoch, weights, optimizer
```

---

## Performance Tips

### DO

```python
# Use context manager
with NumPack("data.npk") as npk:
    # Operations

# Load once, use multiple times
with NumPack("data.npk") as npk:
    arr = npk.load('data')
    for i in range(100):
        process(arr)

# Use batch mode for frequent modifications
with npk.writable_batch_mode() as wb:
    arr = wb.load('data')
    for i in range(100):
        arr *= 1.1

# Use lazy load for partial access
lazy = npk.load('data', lazy=True)
subset = lazy[1000:2000]

# Use streaming for large data
for batch in npk.stream_load('data', buffer_size=10000):
    process(batch)

# Use appropriate dtype
data = np.random.rand(100, 50).astype(np.float32)
```

### DON'T

```python
# Don't create instance in loop
for i in range(100):
    with NumPack("data.npk") as npk:
        data = npk.load('data')

# Don't load repeatedly
with NumPack("data.npk") as npk:
    for i in range(100):
        arr = npk.load('data')  # Reloads every time
        value = arr[i]

# Don't use normal mode for frequent modifications
for i in range(100):
    arr = npk.load('data')
    arr *= 2
    npk.save({'data': arr})  # Slow

# Don't eager load for partial access
arr = npk.load('large_data')  # Loads 10GB
subset = arr[:100]  # Only need 1MB

# Don't use float64 unnecessarily
data = np.random.rand(100, 50)  # float64, 2x size
```

---

## Performance Comparison

### Operation Speed

| Operation | NumPack | NPY | Advantage |
|-----------|---------|-----|-----------|
| Replace 100 rows | 0.047ms | 18.51ms | **397x faster** |
| Append 100 rows | 0.067ms | 27.09ms | **405x faster** |
| Lazy load init | 0.002ms | 0.107ms | **54x faster** |
| Full load | 8.27ms | 10.51ms | **1.3x faster** |
| Save (initial) | 16.15ms | 7.19ms | 2.2x slower |

### Batch Mode Speed

| Mode | Time (100 ops) | Speedup |
|------|----------------|---------|
| Normal | 856ms | 1x |
| Batch mode | 34ms | **25x** |
| Writable batch | 4.9ms | **174x** |

---

## Troubleshooting

### Error: "NumPack instance is not opened"

**Cause:** Trying to use NumPack before opening file

**Solution:**
```python
# Use context manager
with NumPack("data.npk") as npk:
    npk.save({'data': arr})

# Or manually open
npk = NumPack("data.npk")
npk.open()
npk.save({'data': arr})
npk.close()
```

### Error: "Strict context mode requires 'with' statement"

**Cause:** Using strict mode without context manager

**Solution:**
```python
npk = NumPack("data.npk", strict_context_mode=True)
with npk:
    npk.save({'data': arr})
```

### Warning: "Not using context manager" (Windows)

**Cause:** Windows file handle management

**Solution:**
```python
# Best: Use context manager
with NumPack("data.npk") as npk:
    # Operations

# Or suppress warning
npk = NumPack("data.npk", warn_no_context=False)
```

### Out of Memory

**Cause:** Loading large array eagerly

**Solutions:**
```python
# 1. Use lazy loading
lazy = npk.load('large', lazy=True)
subset = lazy[1000:2000]

# 2. Use streaming
for batch in npk.stream_load('large', buffer_size=1000):
    process(batch)

# 3. Use writable batch mode (zero memory)
with npk.writable_batch_mode() as wb:
    arr = wb.load('large')  # mmap, not in RAM
    arr *= 2
```

### Slow Performance

**Cause:** Not using batch mode

**Solution:**
```python
# Instead of normal mode
for i in range(100):
    arr = npk.load('data')
    arr *= 2
    npk.save({'data': arr})

# Use writable batch mode (174x faster)
with npk.writable_batch_mode() as wb:
    arr = wb.load('data')
    for i in range(100):
        arr *= 2
```

### File Size Not Reducing After Drop

**Cause:** Logical deletion, space not reclaimed

**Solution:**
```python
# After dropping rows
npk.drop('array', [0, 1, 2])

# Physically compact
npk.update('array')
```

### Array Not Found

**Cause:** Array doesn't exist in file

**Solution:**
```python
# Check before loading
if npk.has_array('array'):
    arr = npk.load('array')
else:
    print("Array not found")

# Or list all arrays
arrays = npk.get_member_list()
print(f"Available arrays: {arrays}")
```

### Shape Mismatch on Replace/Append

**Cause:** Column count doesn't match

**Solution:**
```python
# Check shape first
shape = npk.get_shape('array')
rows, cols = shape

# Ensure replacement has same column count
new_data = np.random.rand(10, cols)
npk.replace({'array': new_data}, list(range(10)))
```

---

## Platform-Specific Notes

### Windows

```python
# Always use context manager
with NumPack("data.npk") as npk:
    # Operations

# Force cleanup if needed
from numpack import force_cleanup_windows_handles
force_cleanup_windows_handles()
```

### macOS / Linux

```python
# Context manager recommended but not required
npk = NumPack("data.npk")
npk.open()
# Operations
npk.close()
```

---

## Getting Help

### Backend Information

```python
from numpack import get_backend_info

info = get_backend_info()
print(info)
# {
#     'backend_type': 'rust',
#     'platform': 'Darwin',
#     'is_windows': False,
#     'version': '0.4.1'
# }
```

### Documentation Links

- [Full API Reference](./02_core_operations.md)
- [Batch Processing Guide](./03_batch_processing.md)
- [Performance Guide](./05_performance_guide.md)
- [Advanced Features](./04_advanced_features.md)

### External Resources

- [GitHub](https://github.com/BirchKwok/NumPack)
- [PyPI](https://pypi.org/project/numpack/)
- [Issue Tracker](https://github.com/BirchKwok/NumPack/issues)

---

## Common Data Types

```python
# Supported NumPy dtypes
types = {
    'bool': np.bool_,
    'int8': np.int8,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
    'uint8': np.uint8,
    'uint16': np.uint16,
    'uint32': np.uint32,
    'uint64': np.uint64,
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
    'complex64': np.complex64,
    'complex128': np.complex128,
}

# Recommended for most cases
recommended = np.float32  # Good balance of precision and size
```

---

## Memory Usage Estimates

```python
def estimate_memory(rows, cols, dtype):
    """Estimate memory usage"""
    dtype_sizes = {
        np.float32: 4,
        np.float64: 8,
        np.int32: 4,
        np.int64: 8,
        np.uint8: 1,
    }
    bytes_total = rows * cols * dtype_sizes[dtype]
    mb = bytes_total / 1024 / 1024
    return mb

# Example
rows, cols = 1000000, 100
print(f"float32: {estimate_memory(rows, cols, np.float32):.1f} MB")  # 381 MB
print(f"float64: {estimate_memory(rows, cols, np.float64):.1f} MB")  # 762 MB
```

---


**For complete documentation, see [Documentation Index](./README.md)**

