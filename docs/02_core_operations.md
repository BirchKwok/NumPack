# Core Operations API Reference

This document covers all core NumPack operations including save, load, replace, append, drop, and random access.

## Table of Contents

- [Initialization](#initialization)
- [Save Operations](#save-operations)
- [Load Operations](#load-operations)
- [Replace Operations](#replace-operations)
- [Append Operations](#append-operations)
- [Drop Operations](#drop-operations)
- [Random Access](#random-access)
- [Metadata Operations](#metadata-operations)
- [Array Operations](#array-operations)
- [Stream Loading](#stream-loading)
- [File Management](#file-management)

## Initialization

### `NumPack(filename, **options)`

Create a NumPack instance for array storage.

#### Parameters

- `filename` (str | Path): Path to the NumPack storage directory
- `drop_if_exists` (bool, optional): Delete existing file if it exists. Default: `False`
- `strict_context_mode` (bool, optional): Require usage within `with` statement. Default: `False`
- `warn_no_context` (bool, optional): Warn when not using context manager. Default: `True` on Windows, `False` elsewhere
- `force_gc_on_close` (bool, optional): Force garbage collection on close. Default: `False`

#### Example

```python
from numpack import NumPack

# Basic initialization
npk = NumPack("data.npk")

# With options
npk = NumPack(
    "data.npk",
    drop_if_exists=True,        # Delete if exists
    strict_context_mode=True,   # Require 'with' statement
    force_gc_on_close=False     # Don't force GC (better performance)
)
```

#### Performance Notes

- Set `force_gc_on_close=True` only if you have strict memory constraints
- Forcing GC can impact performance, especially with frequent open/close operations

---

## Save Operations

### `save(arrays)`

Save one or more arrays to the NumPack file.

#### Parameters

- `arrays` (Dict[str, np.ndarray]): Dictionary mapping array names to NumPy arrays

#### Example

```python
import numpy as np

with NumPack("data.npk") as npk:
    # Save single array
    npk.save({'features': np.random.rand(1000, 100)})
    
    # Save multiple arrays
    arrays = {
        'features': np.random.rand(1000, 100).astype(np.float32),
        'labels': np.random.randint(0, 10, size=1000).astype(np.int32),
        'weights': np.random.rand(1000).astype(np.float32)
    }
    npk.save(arrays)
```

#### Behavior

- If array name doesn't exist: Creates new array
- If array name exists: Replaces entire array

#### Performance Notes

- NumPack save is ~2.2x slower than NPY for initial write (but much faster for modifications)
- Use batch mode for frequent save operations (25-37x speedup)
- All arrays are validated for compatibility with NumPack format

---

## Load Operations

### `load(array_name, lazy=False)`

Load an array from the NumPack file.

#### Parameters

- `array_name` (str): Name of the array to load
- `lazy` (bool, optional): Whether to use lazy (memory-mapped) loading. Default: `False`

#### Returns

- `np.ndarray`: The loaded array (eager mode)
- `LazyArray`: Memory-mapped lazy array (lazy mode)

#### Example

```python
with NumPack("data.npk") as npk:
    # Eager loading (default) - loads entire array into memory
    features = npk.load('features')
    print(type(features))  # <class 'numpy.ndarray'>
    
    # Lazy loading - memory mapped, loaded on demand
    features_lazy = npk.load('features', lazy=True)
    print(type(features_lazy))  # <class 'LazyArray'>
    
    # LazyArray behaves like NumPy array
    subset = features_lazy[100:200]  # Only loads this portion
```

### `__getitem__(key)` (Dictionary-style access)

Shorthand for loading arrays using bracket notation.

#### Example

```python
with NumPack("data.npk") as npk:
    # These are equivalent
    features1 = npk.load('features')
    features2 = npk['features']
    
    # Dictionary-style is more concise
    labels = npk['labels']
```

#### Performance Notes

| Operation | Time (1M rows) | Use Case |
|-----------|---------------|----------|
| Eager load | 8.27ms | Small arrays, need entire dataset |
| Lazy load | 0.002ms | Large arrays, partial access |
| NPY mmap | 0.107ms | Comparison baseline |

**Recommendation**:
- Use **eager loading** when you need the full array and it fits in memory
- Use **lazy loading** for large arrays (> 100MB) or when you only need portions
- Lazy loading is 54x faster to initialize than NPY mmap

---

## Replace Operations

### `replace(arrays, indexes)`

Replace values at specific row indexes.

#### Parameters

- `arrays` (Dict[str, np.ndarray]): Dictionary mapping array names to replacement values
- `indexes` (int | List[int] | np.ndarray | slice): Row indexes to replace

#### Example

```python
with NumPack("data.npk") as npk:
    # Replace single row
    new_row = np.random.rand(100)
    npk.replace({'features': new_row}, 0)
    
    # Replace multiple specific rows
    new_rows = np.random.rand(5, 100)
    npk.replace({'features': new_rows}, [0, 10, 20, 30, 40])
    
    # Replace using numpy array of indexes
    indexes = np.array([1, 3, 5, 7, 9])
    new_rows = np.random.rand(5, 100)
    npk.replace({'features': new_rows}, indexes)
    
    # Replace using slice
    new_rows = np.random.rand(10, 100)
    npk.replace({'features': new_rows}, slice(0, 10))
    
    # Replace multiple arrays simultaneously
    npk.replace({
        'features': np.random.rand(3, 100),
        'labels': np.array([1, 2, 3])
    }, [0, 1, 2])
```

#### Performance Notes

| Format | Replace Time (100 rows) | NumPack Advantage |
|--------|------------------------|-------------------|
| NumPack | **0.047ms** | 1x (baseline) |
| NPY | 18.51ms | **397x faster** |
| NPZ | 1574ms | **33,489x faster** |
| HDF5 | 0.299ms | **6.4x faster** |

**Recommendation**:
- Replace is NumPack's strongest feature - 397x faster than NPY
- Use replace instead of load-modify-save pattern for maximum performance

---

## Append Operations

### `append(arrays)`

Append new rows to existing arrays.

#### Parameters

- `arrays` (Dict[str, np.ndarray]): Dictionary mapping array names to rows to append

#### Example

```python
with NumPack("data.npk") as npk:
    # Initial save
    npk.save({'features': np.random.rand(1000, 100)})
    
    # Append 100 new rows
    new_data = np.random.rand(100, 100)
    npk.append({'features': new_data})
    
    # Verify shape changed
    shape = npk.get_shape('features')
    print(shape)  # (1100, 100)
    
    # Append to multiple arrays
    npk.append({
        'features': np.random.rand(50, 100),
        'labels': np.random.randint(0, 10, size=50)
    })
```

#### Important Notes

- Column count must match existing array
- Appended data dtype should match existing array
- Efficient incremental growth without full file rewrite

#### Performance Notes

| Format | Append Time (100 rows) | NumPack Advantage |
|--------|------------------------|-------------------|
| NumPack | **0.067ms** | 1x (baseline) |
| NPY | 27.09ms | **405x faster** |
| NPZ | 1582ms | **23,612x faster** |
| HDF5 | 0.212ms | **3.2x faster** |

**Recommendation**:
- Append is 405x faster than NPY - ideal for incremental data collection
- Use batch mode for even better performance with multiple appends

---

## Drop Operations

### `drop(array_name, indexes=None)`

Drop entire arrays or specific rows.

#### Parameters

- `array_name` (str | List[str]): Name(s) of array(s) to drop
- `indexes` (Optional[int | List[int] | np.ndarray | slice]): Row indexes to drop. If `None`, drops entire array(s)

#### Example

```python
with NumPack("data.npk") as npk:
    # Save some test data
    npk.save({
        'array1': np.random.rand(100, 50),
        'array2': np.random.rand(100, 30),
        'array3': np.random.rand(100, 20)
    })
    
    # Drop entire array
    npk.drop('array1')
    
    # Drop multiple arrays
    npk.drop(['array2', 'array3'])
    
    # Drop specific rows from array
    npk.drop('array1', [0, 1, 2, 10, 20])
    
    # Drop using numpy array
    indexes = np.array([5, 15, 25])
    npk.drop('array1', indexes)
    
    # Drop using slice
    npk.drop('array1', slice(0, 10))  # Drop first 10 rows
```

#### Row Deletion Mechanism

NumPack uses **logical deletion** for efficiency:

1. Rows are marked as deleted in a bitmap (not physically removed)
2. Deleted rows are skipped during load/access operations
3. Disk space is not immediately reclaimed

To physically compact the file and reclaim space:

```python
# After dropping rows
npk.drop('array1', [0, 1, 2])

# Physically compact the array to reclaim disk space
npk.update('array1')
```

#### Performance Notes

- Drop operation is very fast (logical deletion via bitmap)
- Use `update()` to physically compact and reclaim disk space
- Compaction uses batched approach (100K rows per batch) for large arrays

---

## Random Access

### `getitem(array_name, indexes)`

Randomly access specific rows from an array.

#### Parameters

- `array_name` (str): Name of the array
- `indexes` (int | List[int] | np.ndarray): Row indexes to access

#### Returns

- `np.ndarray`: The requested row(s)

#### Example

```python
with NumPack("data.npk") as npk:
    # Save data
    npk.save({'features': np.random.rand(1000, 100)})
    
    # Access single row
    row = npk.getitem('features', 0)
    print(row.shape)  # (100,)
    
    # Access multiple specific rows
    rows = npk.getitem('features', [0, 10, 20, 30])
    print(rows.shape)  # (4, 100)
    
    # Access using numpy array of indexes
    indexes = np.array([1, 5, 10, 15, 20])
    rows = npk.getitem('features', indexes)
    print(rows.shape)  # (5, 100)
```

#### Performance Notes

| Format | Random Access (1K rows) | Performance |
|--------|------------------------|-------------|
| NPY | **0.010ms** | Fastest |
| NumPack | 0.051ms | 5x slower |
| HDF5 | 4.91ms | 491x slower |
| NPZ | 183.16ms | 18,316x slower |

**Recommendation**:
- For frequent random access of small subsets, consider using NPY mmap
- NumPack random access is still very fast and much better than compressed formats
- For large batch access, lazy loading is more efficient

---

## Metadata Operations

### `get_shape(array_name)`

Get the shape of an array.

#### Parameters

- `array_name` (str): Name of the array

#### Returns

- `Tuple[int, int]`: Shape as (rows, columns)

#### Example

```python
with NumPack("data.npk") as npk:
    shape = npk.get_shape('features')
    rows, cols = shape
    print(f"Shape: {rows} rows, {cols} columns")
```

### `get_member_list()`

Get list of all array names in the file.

#### Returns

- `List[str]`: List of array names

#### Example

```python
with NumPack("data.npk") as npk:
    arrays = npk.get_member_list()
    print(f"Arrays: {arrays}")
    
    # Iterate over all arrays
    for name in arrays:
        shape = npk.get_shape(name)
        print(f"{name}: {shape}")
```

### `has_array(array_name)`

Check if an array exists in the file.

#### Parameters

- `array_name` (str): Name to check

#### Returns

- `bool`: `True` if exists, `False` otherwise

#### Example

```python
with NumPack("data.npk") as npk:
    if npk.has_array('features'):
        data = npk.load('features')
    else:
        print("Array not found")
```

### `get_modify_time(array_name)`

Get last modification timestamp of an array.

#### Parameters

- `array_name` (str): Name of the array

#### Returns

- `Optional[int]`: Unix timestamp, or `None` if array doesn't exist

#### Example

```python
from datetime import datetime

with NumPack("data.npk") as npk:
    timestamp = npk.get_modify_time('features')
    if timestamp:
        dt = datetime.fromtimestamp(timestamp)
        print(f"Last modified: {dt}")
```

### `get_metadata()`

Get complete metadata dictionary.

#### Returns

- `Dict[str, Any]`: Metadata information

#### Example

```python
with NumPack("data.npk") as npk:
    metadata = npk.get_metadata()
    print(metadata)
```

### `reset()`

Clear all arrays from the file.

#### Example

```python
with NumPack("data.npk") as npk:
    # Clear everything
    npk.reset()
    
    # File is now empty
    print(npk.get_member_list())  # []
```

### `update(array_name)`

Physically compact an array by removing logically deleted rows.

#### Parameters

- `array_name` (str): Name of the array to compact

#### Example

```python
with NumPack("data.npk") as npk:
    # Delete some rows (logical deletion)
    npk.drop('features', [0, 1, 2, 100, 200, 300])
    
    # Array still occupies full disk space
    # Physically compact to reclaim space
    npk.update('features')
```

#### Notes

- Compaction creates a new file with only non-deleted rows
- Original file is replaced after compaction
- Uses batched approach (100K rows per batch) for large arrays
- If no rows were deleted, operation is a no-op

---

## Array Operations

### `clone(source_name, target_name)`

Clone an existing array to a new array name.

#### Parameters

- `source_name` (str): Name of the source array to clone
- `target_name` (str): Name for the cloned array

#### Example

```python
with NumPack("data.npk") as npk:
    # Save original data
    npk.save({'original': np.random.rand(100, 50)})
    
    # Clone to new array
    npk.clone('original', 'backup')
    
    # Modify the clone independently
    backup = npk.load('backup')
    backup *= 2.0
    npk.save({'backup': backup})
    
    # Original is unchanged
    original = npk.load('original')
```

#### Notes

- The cloned array is independent of the original
- Raises `KeyError` if source array doesn't exist
- Raises `ValueError` if target array already exists

### `get_io_stats()`

Get I/O statistics for the NumPack instance.

#### Returns

- `Dict[str, Any]`: Dictionary containing backend statistics

#### Example

```python
with NumPack("data.npk") as npk:
    stats = npk.get_io_stats()
    print(f"Backend: {stats['backend_type']}")
```

#### Notes

- Currently returns basic backend information
- Detailed per-call statistics may be added in future versions

---

## Stream Loading

### `stream_load(array_name, buffer_size=None)`

Stream array data in batches for memory-efficient processing.

#### Parameters

- `array_name` (str): Name of the array to stream
- `buffer_size` (Optional[int]): Rows per batch. If `None`, loads one row at a time

#### Returns

- `Iterator[np.ndarray]`: Iterator yielding batches

#### Example

```python
with NumPack("data.npk") as npk:
    # Save large array
    npk.save({'large_array': np.random.rand(1000000, 100)})
    
    # Stream in batches of 10,000 rows
    for batch in npk.stream_load('large_array', buffer_size=10000):
        print(f"Processing batch of shape: {batch.shape}")
        # Process batch...
        result = process_batch(batch)
    
    # Stream one row at a time (memory-efficient but slower)
    for row in npk.stream_load('large_array'):
        print(f"Row shape: {row.shape}")
```

#### Use Cases

- Processing arrays too large to fit in memory
- ETL pipelines with large datasets
- Online learning with incremental updates

#### Performance Notes

- Larger `buffer_size` = fewer iterations, more memory
- Smaller `buffer_size` = more iterations, less memory
- Typical good value: 10,000 - 100,000 rows depending on row size

---

## File Management

### `open()`

Manually open the NumPack file.

#### Example

```python
npk = NumPack("data.npk")
npk.open()  # Manually open
npk.save({'data': np.random.rand(100, 50)})
npk.close()
```

### `close(force_gc=None)`

Explicitly close the file and release resources.

#### Parameters

- `force_gc` (Optional[bool]): Force garbage collection. If `None`, uses instance default

#### Example

```python
npk = NumPack("data.npk")
npk.open()
npk.save({'data': np.random.rand(100, 50)})
npk.close()

# Force GC on close
npk.close(force_gc=True)
```

#### Notes

- Multiple calls to `close()` are safe (idempotent)
- File can be reopened after close by calling `open()`
- Metadata is flushed automatically

### Properties

#### `is_opened`

Check if file is currently opened.

```python
npk = NumPack("data.npk")
print(npk.is_opened)  # False

npk.open()
print(npk.is_opened)  # True

npk.close()
print(npk.is_opened)  # False
```

#### `is_closed`

Check if file is currently closed.

```python
npk = NumPack("data.npk")
print(npk.is_closed)  # True

npk.open()
print(npk.is_closed)  # False
```

#### `backend_type`

Get the backend type (always `'rust'`).

```python
npk = NumPack("data.npk")
print(npk.backend_type)  # 'rust'
```

### Iterator Protocol

NumPack supports iteration over array names:

```python
with NumPack("data.npk") as npk:
    npk.save({
        'array1': np.random.rand(100, 50),
        'array2': np.random.rand(200, 30),
        'array3': np.random.rand(150, 40)
    })
    
    # Iterate over array names
    for name in npk:
        shape = npk.get_shape(name)
        print(f"{name}: {shape}")
```

---

## Complete Example

Here's a comprehensive example using multiple operations:

```python
import numpy as np
from numpack import NumPack

# Initialize with options
with NumPack("example.npk", drop_if_exists=True) as npk:
    # Save initial data
    npk.save({
        'features': np.random.rand(1000, 100).astype(np.float32),
        'labels': np.random.randint(0, 10, size=1000).astype(np.int32)
    })
    
    # Get metadata
    arrays = npk.get_member_list()
    print(f"Arrays: {arrays}")
    
    # Load and inspect
    features = npk.load('features')
    print(f"Features shape: {features.shape}")
    
    # Replace some rows
    new_rows = np.random.rand(10, 100).astype(np.float32)
    npk.replace({'features': new_rows}, list(range(10)))
    
    # Append more data
    npk.append({
        'features': np.random.rand(100, 100).astype(np.float32),
        'labels': np.random.randint(0, 10, size=100).astype(np.int32)
    })
    
    # New shape after append
    print(f"New shape: {npk.get_shape('features')}")  # (1100, 100)
    
    # Random access
    sample = npk.getitem('features', [0, 100, 200, 300, 400])
    print(f"Sample shape: {sample.shape}")  # (5, 100)
    
    # Drop some rows
    npk.drop('features', [0, 1, 2])
    
    # Compact to reclaim space
    npk.update('features')
    
    # Stream processing
    for batch in npk.stream_load('features', buffer_size=100):
        print(f"Processing batch: {batch.shape}")
        # Process...

print("NumPack file closed automatically")
```

