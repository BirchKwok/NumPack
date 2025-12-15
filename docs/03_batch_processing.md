# Batch Processing API Reference

NumPack provides two high-performance batch processing modes for scenarios with frequent read/write operations: **Batch Mode** and **Writable Batch Mode**. These modes can achieve 25-174x speedup compared to normal operations.

## Table of Contents

- [Quick Comparison](#quick-comparison)
- [Batch Mode](#batch-mode)
- [Writable Batch Mode](#writable-batch-mode)
- [Performance Comparison](#performance-comparison)
- [Selection Guide](#selection-guide)
- [Best Practices](#best-practices)
- [Advanced Usage](#advanced-usage)

## Quick Comparison

| Feature | `batch_mode` | `writable_batch_mode` |
|---------|--------------|----------------------|
| **Storage** | Memory cache | File mmap mapping |
| **Memory usage** | Full array size | ~0 (virtual only) |
| **load() behavior** | Copy to memory | Create file view |
| **Modification** | Modify memory copy | Direct file modification |
| **save() behavior** | Update cache | No-op (optional) |
| **Shape changes** | Supported | Not supported |
| **Best for** | Small arrays (< 100MB) | Large arrays (> 100MB) |
| **Performance boost** | 25-37x | 174x |
| **Memory constraint** | Moderate | Very tight |

## When to Use Each Mode

```
Start
 │
 ├─ Need to change array shape?
 │   ├─ Yes → Use batch_mode
 │   └─ No → Continue
 │
 ├─ Array total size > 50% of available RAM?
 │   ├─ Yes → Use writable_batch_mode
 │   └─ No → Continue
 │
 ├─ Array size > 100MB?
 │   ├─ Yes → Use writable_batch_mode
 │   └─ No → Use batch_mode
```

---

## Batch Mode

### Overview

Batch mode caches arrays in memory and writes all changes to disk in a single batch operation when exiting the context. This reduces disk I/O overhead for frequent operations.

### How It Works

```
Disk File          Memory Cache         User Code
   │                    │                    │
   │ 1. load()          │                    │
   ├───────────────────>│                    │
   │  Read data         │                    │
   │                    │  2. Return copy    │
   │                    ├───────────────────>│
   │                    │                    │ 3. arr *= 2
   │                    │                    │
   │                    │  4. save()         │
   │                    │<───────────────────┤
   │                    │  Update cache      │
   │                    │                    │
   │ 5. Exit: Flush     │                    │
   │<───────────────────┤                    │
   │                    │                    │

Features:
• Stores modified copies in memory
• Reduces disk I/O (batch write on exit)
• Supports shape changes (append/reshape)
• Memory usage = Σ(modified array sizes)
```

### API Reference

#### `batch_mode(memory_limit=None)`

Create a batch processing context.

**Parameters:**
- `memory_limit` (Optional[int]): Memory limit in MB. If set, switches to streaming mode when exceeded.

**Returns:**
- `BatchModeContext`: Batch processing context manager

### Basic Example

```python
import numpy as np
from numpack import NumPack

# Create and save initial data
with NumPack("data.npk", drop_if_exists=True) as npk:
    npk.save({'features': np.random.rand(1000, 100).astype(np.float32)})

# Use batch mode for frequent modifications
with NumPack("data.npk") as npk:
    with npk.batch_mode():
        for i in range(100):
            # Load from cache after first time
            arr = npk.load('features')
            
            # Modify in memory
            arr *= 1.1
            
            # Update cache only (no disk I/O)
            npk.save({'features': arr})
        
        # All changes written to disk here
```

### Advanced Example: Shape Changes

Batch mode supports operations that change array shapes:

```python
with NumPack("data.npk") as npk:
    with npk.batch_mode():
        # Load existing data
        features = npk.load('features')
        
        # Modify values
        features *= 2.0
        npk.save({'features': features})
        
        # Append new data (shape change)
        new_data = np.random.rand(100, 100).astype(np.float32)
        npk.append({'features': new_data})  # Supported
        
        # Reshape (shape change)
        metadata = npk.load('metadata')
        metadata = metadata.reshape(-1, 10)
        npk.save({'metadata': metadata})  # Supported
```

### Memory Management

```python
with NumPack("data.npk") as npk:
    # Set memory limit (in MB)
    with npk.batch_mode(memory_limit=1000):
        # If cache exceeds 1GB, automatically switches to streaming
        for i in range(1000):
            arr = npk.load(f'array_{i}')
            arr *= 2.0
            npk.save({f'array_{i}': arr})
```

### Performance Example

```python
import time
import numpy as np
from numpack import NumPack

# Setup
with NumPack("benchmark.npk", drop_if_exists=True) as npk:
    npk.save({'data': np.random.rand(10000, 100).astype(np.float32)})

# Normal mode
with NumPack("benchmark.npk") as npk:
    start = time.time()
    for i in range(100):
        arr = npk.load('data')
        arr *= 1.1
        npk.save({'data': arr})
    normal_time = time.time() - start
    print(f"Normal mode: {normal_time:.3f}s")

# Batch mode
with NumPack("benchmark.npk") as npk:
    start = time.time()
    with npk.batch_mode():
        for i in range(100):
            arr = npk.load('data')
            arr *= 1.1
            npk.save({'data': arr})
    batch_time = time.time() - start
    print(f"Batch mode: {batch_time:.3f}s")
    print(f"Speedup: {normal_time/batch_time:.1f}x")

# Expected output:
# Normal mode: 0.856s
# Batch mode: 0.034s
# Speedup: 25.2x
```

---

## Writable Batch Mode

### Overview

Writable batch mode uses memory-mapped file I/O to provide zero-copy direct file modification. The returned NumPy arrays are views directly into the file-mapped memory, so modifications are automatically written to the file.

### How It Works

```
Disk File          mmap Mapping         User Code
   │                    │                    │
   │ 1. load()          │                    │
   ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ >│ (map to virtual)   │
   ║                    │                    │
   ║  Shared memory     │  2. Return view    │
   ║                    ├───────────────────>│
   ║                    │                    │ 3. arr *= 2
   ║  <══════════════════════════════════════  (direct modify)
   ║      OS auto-sync dirty pages           │
   ║                    │                    │
   ║                    │  4. save()         │
   ║                    │<───────────────────┤
   ║                    │  (no-op)           │
   ║                    │                    │
   │ 5. Exit: flush     │                    │
   │<───────────────────┤                    │
   │                    │                    │

Features:
• File directly mapped to virtual memory
• Modifications auto-sync (OS managed)
• Zero memory copy (virtual memory only)
• Shape changes not supported
• Memory usage ≈ 0
```

### API Reference

#### `writable_batch_mode()`

Create a writable batch processing context with zero-copy file mapping.

**Returns:**
- `WritableBatchMode`: Writable batch processing context manager

### Basic Example

```python
import numpy as np
from numpack import NumPack

# Create initial data
with NumPack("data.npk", drop_if_exists=True) as npk:
    npk.save({'features': np.random.rand(1000, 100).astype(np.float32)})

# Use writable batch mode
with NumPack("data.npk") as npk:
    with npk.writable_batch_mode() as wb:
        for i in range(100):
            # Get mmap view (zero-copy)
            arr = wb.load('features')
            
            # Modify directly on file
            arr *= 1.1
            
            # save() is optional (no-op)
            wb.save({'features': arr})
        
        # mmap flushed to disk here
```

### Zero-Copy Verification

```python
with NumPack("data.npk") as npk:
    with npk.writable_batch_mode() as wb:
        arr = wb.load('features')
        
        # Verify it's a view (not a copy)
        print(f"OWNDATA: {arr.flags['OWNDATA']}")  # False
        print(f"WRITEABLE: {arr.flags['WRITEABLE']}")  # True
        
        # Direct modification
        arr[0, 0] = 999.0
        
        # No need to call save() - already written to file
```

### Large Array Example

Writable batch mode can handle arrays larger than available RAM:

```python
import numpy as np
from numpack import NumPack

# Create large array (e.g., 10GB)
with NumPack("large.npk", drop_if_exists=True) as npk:
    # This creates the file structure
    large_array = np.random.rand(100000000, 10).astype(np.float32)  # ~4GB
    npk.save({'large_features': large_array})
    del large_array  # Free memory

# Modify the large array with minimal memory usage
with NumPack("large.npk") as npk:
    with npk.writable_batch_mode() as wb:
        # Memory usage: ~0 MB (only virtual mapping)
        features = wb.load('large_features')
        
        # Process in chunks to avoid loading all at once
        chunk_size = 10000
        for i in range(0, len(features), chunk_size):
            # Only this chunk is loaded into physical memory
            features[i:i+chunk_size] *= 2.0
        
        # OS handles dirty page sync automatically
```

### Cache Array Reference

For maximum performance, cache the array reference instead of loading repeatedly:

```python
# Efficient: Cache reference
with NumPack("data.npk") as npk:
    with npk.writable_batch_mode() as wb:
        arr = wb.load('features')  # Load once
        for i in range(1000):
            arr *= 1.1  # Reuse cached reference

# Inefficient: Load every time
with NumPack("data.npk") as npk:
    with npk.writable_batch_mode() as wb:
        for i in range(1000):
            arr = wb.load('features')  # Creates new mmap each time
            arr *= 1.1
```

### Performance Example

```python
import time
import numpy as np
from numpack import NumPack

# Setup
with NumPack("benchmark.npk", drop_if_exists=True) as npk:
    npk.save({'data': np.random.rand(100000, 100).astype(np.float32)})

# Normal mode
with NumPack("benchmark.npk") as npk:
    start = time.time()
    for i in range(100):
        arr = npk.load('data')
        arr *= 1.1
        npk.save({'data': arr})
    normal_time = time.time() - start
    print(f"Normal mode: {normal_time:.3f}s")

# Writable batch mode
with NumPack("benchmark.npk") as npk:
    start = time.time()
    with npk.writable_batch_mode() as wb:
        arr = wb.load('data')  # Load once
        for i in range(100):
            arr *= 1.1
    writable_time = time.time() - start
    print(f"Writable batch mode: {writable_time:.3f}s")
    print(f"Speedup: {normal_time/writable_time:.1f}x")

# Expected output:
# Normal mode: 8.520s
# Writable batch mode: 0.049s
# Speedup: 174.0x
```

---

## Performance Comparison

### Benchmark Results

| Scenario | Normal Mode | batch_mode | writable_batch_mode |
|----------|-------------|------------|---------------------|
| Small array (80KB), 100 ops | 100 ms | 3.8 ms (26x) | 5.1 ms (20x) |
| Medium array (8MB), 50 ops | 1800 ms | 48 ms (37x) | 27 ms (67x) |
| Large array (800MB), 10 ops | OOM | OOM | 120 ms |

### Memory Usage Comparison

| Scenario | batch_mode | writable_batch_mode |
|----------|------------|---------------------|
| Modify 1 x 8MB array | ~8 MB | ~0.01 MB |
| Modify 3 x 8MB arrays | ~24 MB | ~0.01 MB |
| Modify 100 x 8MB arrays | ~800 MB | ~0.01 MB |

### Performance by Array Size

| Array Size | Recommended Mode | Reason |
|------------|------------------|---------|
| < 10MB | batch_mode | Lower overhead, supports shape changes |
| 10-100MB | batch_mode | Good balance unless memory constrained |
| 100MB-1GB | writable_batch_mode | Better performance, lower memory |
| > 1GB | writable_batch_mode | Only viable option |

---

## Selection Guide

### Use `batch_mode` when:

You need to change array shapes (append/reshape/resize)  
Arrays are relatively small (< 100MB total)  
You want to create new arrays during processing  
Memory is not a constraint  
Maximum flexibility is needed  

### Use `writable_batch_mode` when:

Arrays are large (> 100MB)  
Only value modifications needed (no shape changes)  
Memory is constrained  
Processing TB-scale datasets  
Maximum performance is critical  
Filesystem supports mmap  

---

## Best Practices

### Recommended Patterns

#### 1. Cache Array Reference (writable_batch_mode)

```python
with npk.writable_batch_mode() as wb:
    arr = wb.load('data')  # Load once
    for i in range(100):
        arr *= 1.1  # Reuse cached reference
```

#### 2. Batch Operations (batch_mode)

```python
with npk.batch_mode():
    for name in ['array1', 'array2', 'array3']:
        arr = npk.load(name)
        arr *= 2.0
        npk.save({name: arr})
    # All written in one batch on exit
```

#### 3. Mixed Usage

```python
# Large arrays: writable_batch_mode
with npk.writable_batch_mode() as wb:
    large = wb.load('large_features')
    large *= 2.0

# Small arrays + shape changes: batch_mode
with npk.batch_mode():
    small = npk.load('metadata')
    small = small.reshape(-1, 10)
    npk.save({'metadata': small})
```

#### 4. Process Large Arrays in Chunks

```python
with npk.writable_batch_mode() as wb:
    features = wb.load('huge_features')
    
    # Process in chunks
    chunk_size = 10000
    for i in range(0, len(features), chunk_size):
        chunk = features[i:i+chunk_size]
        chunk *= 2.0  # Only this chunk in physical memory
```

### Avoid These Mistakes

#### 1. Changing Shape in writable_batch_mode

```python
# Wrong: Creates new array, loses mmap
with npk.writable_batch_mode() as wb:
    arr = wb.load('data')
    arr = arr.reshape(-1, 10)  # Creates copy
    wb.save({'data': arr})

# Correct: Use batch_mode for reshaping
with npk.batch_mode():
    arr = npk.load('data')
    arr = arr.reshape(-1, 10)
    npk.save({'data': arr})
```

#### 2. Repeated Loading

```python
# Inefficient: Creates new mmap each time
with npk.writable_batch_mode() as wb:
    for i in range(100):
        arr = wb.load('data')
        arr *= 1.1

# Efficient: Load once
with npk.writable_batch_mode() as wb:
    arr = wb.load('data')
    for i in range(100):
        arr *= 1.1
```

#### 3. Nested Contexts

```python
# Wrong: Nested batch contexts
with npk.batch_mode():
    with npk.writable_batch_mode() as wb:
        # Undefined behavior
        pass

# Correct: Separate sequential contexts
with npk.batch_mode():
    # Process small arrays
    pass

with npk.writable_batch_mode() as wb:
    # Process large arrays
    pass
```

---

## Advanced Usage

### Combining with Lazy Loading

```python
with NumPack("data.npk") as npk:
    # Lazy load first to check shape
    lazy_arr = npk.load('features', lazy=True)
    print(f"Shape: {lazy_arr.shape}")
    
    # Then use appropriate batch mode
    if lazy_arr.nbytes > 100_000_000:  # > 100MB
        with npk.writable_batch_mode() as wb:
            arr = wb.load('features')
            arr *= 2.0
    else:
        with npk.batch_mode():
            arr = npk.load('features')
            arr *= 2.0
            npk.save({'features': arr})
```

### Error Handling

Both modes ensure proper cleanup even when errors occur:

```python
try:
    with npk.batch_mode():
        arr = npk.load('data')
        arr *= 2.0
        npk.save({'data': arr})
        raise ValueError("Something went wrong")
except ValueError:
    pass
# Cache is still flushed properly

try:
    with npk.writable_batch_mode() as wb:
        arr = wb.load('data')
        arr *= 2.0
        raise ValueError("Something went wrong")
except ValueError:
    pass
# mmap is still flushed properly
```

### Multiple Arrays

#### batch_mode: Multiple Arrays

```python
with npk.batch_mode():
    # All arrays cached in memory
    features = npk.load('features')
    labels = npk.load('labels')
    weights = npk.load('weights')
    
    features *= 2.0
    labels += 1
    weights *= 0.5
    
    npk.save({
        'features': features,
        'labels': labels,
        'weights': weights
    })
# All written in one batch
```

#### writable_batch_mode: Multiple Arrays

```python
with npk.writable_batch_mode() as wb:
    # Each array is mmap'd
    features = wb.load('features')
    labels = wb.load('labels')
    weights = wb.load('weights')
    
    # All modifications are direct file writes
    features *= 2.0
    labels += 1
    weights *= 0.5
    
    # No save needed (optional for API consistency)
```

---

## Technical Details

### batch_mode Implementation

```python
class BatchModeContext:
    def __enter__(self):
        self.npk._cache_enabled = True  # Enable caching
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.npk._flush_cache()  # Flush to file
        self.npk._cache_enabled = False

# load() checks cache
def load(self, array_name):
    if self._cache_enabled:
        if array_name in self._memory_cache:
            return self._memory_cache[array_name]  # From cache
        else:
            arr = self._npk.load(array_name, lazy=False)
            self._memory_cache[array_name] = arr
            return arr
```

### writable_batch_mode Implementation

```python
class WritableBatchMode:
    def load(self, array_name):
        # Open file and create mmap
        file = open(file_path, 'r+b')
        mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE)
        
        # Create NumPy array view (zero-copy)
        arr = np.ndarray(shape=shape, dtype=dtype, buffer=mm)
        
        # Keep mmap reference (prevent GC)
        self.writable_arrays[array_name] = (file, mm)
        return arr
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for array_name, (file, mm) in self.writable_arrays.items():
            mm.flush()  # Ensure persistence
            mm.close()
            file.close()
```

**Key Points**:
- `buffer=mm` makes NumPy array use mmap's memory
- `arr.flags['OWNDATA'] == False` indicates it's a view
- Array modifications directly modify file-mapped memory

---

## FAQ

### Q1: Why do both return numpy.ndarray?

**A:** Both return `numpy.ndarray`, but they're fundamentally different:
- `batch_mode`: Returns an independent copy in memory
- `writable_batch_mode`: Returns a zero-copy view of file mmap

The key difference is in `arr.flags['OWNDATA']` and the underlying buffer source.

### Q2: Why isn't writable_batch_mode in Rust?

**A:** Python's `mmap` module is already a thin wrapper over system calls:
- Python mmap overhead < 1%
- Rust implementation wouldn't provide significant performance gain
- The bottleneck is in array computation and disk I/O, not mmap

### Q3: Can both modes be used simultaneously?

**A:** Not in the same context, but sequentially:

```python
# OK
with npk.writable_batch_mode() as wb:
    # Process large arrays
    pass

with npk.batch_mode():
    # Process small arrays
    pass

# Not recommended
with npk.batch_mode():
    with npk.writable_batch_mode() as wb:
        # Undefined behavior
        pass
```

### Q4: Does writable_batch_mode support all NumPy operations?

**A:** Most operations are supported, with limitations:

- In-place modifications: `arr *= 2.0`, `arr += 1.0`, `arr[0] = 5.0`
- Universal functions: `np.sin(arr)`, `np.exp(arr)` (if in-place)
- Shape changes: `arr.reshape()`, `arr.resize()`
- New arrays: `arr + arr2` (creates new array, loses mmap)

---

## Summary

Both APIs serve different use cases and should be kept:

| Use Case | Recommended API |
|----------|-----------------|
| Small arrays, frequent read/write | `batch_mode` |
| Large arrays, value-only changes | `writable_batch_mode` |
| Need shape changes | `batch_mode` |
| Memory constrained | `writable_batch_mode` |
| TB-scale data | `writable_batch_mode` |

**Remember the core difference:**
- `batch_mode`: Memory cache, flexible but uses RAM
- `writable_batch_mode`: Zero-copy view, saves RAM but has restrictions

