# Batch Modes API Reference

NumPack provides two batch processing modes for high-performance repeated operations.

## Overview

| Mode | Class | Memory Usage | Speed | Shape Changes |
|------|-------|--------------|-------|---------------|
| `batch_mode()` | `BatchModeContext` | Higher | 25-37x faster | ✅ Supported |
| `writable_batch_mode()` | `WritableBatchMode` | Near zero | 174x faster | ❌ Not supported |

---

# BatchModeContext

In-memory caching mode that batches disk writes.

## Class Definition

```python
class BatchModeContext:
    def __init__(self, numpack_instance: NumPack, memory_limit=None)
```

## Usage

```python
from numpack import NumPack

with NumPack('data.npk') as npk:
    with npk.batch_mode(memory_limit=1024) as batch:
        # All operations use memory cache
        for i in range(100):
            arr = npk.load('array')
            arr *= 2
            npk.save({'array': arr})
        # Changes flushed on exit
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `numpack_instance` | `NumPack` | *required* | Parent NumPack instance |
| `memory_limit` | `int` or `None` | `None` | Soft memory limit in MB |

## How It Works

1. **On Enter**: Enables in-memory caching
2. **During Operations**: 
   - `load()` returns cached arrays (or loads and caches)
   - `save()` updates the cache (no disk write)
3. **On Exit**: Flushes all dirty arrays to disk

## Methods

### `__enter__()`

Enter batch mode context.

#### Returns

- `BatchModeContext`: The context manager instance

---

### `__exit__(exc_type, exc_val, exc_tb)`

Exit batch mode and flush changes.

#### Returns

- `bool`: Always `False` (doesn't suppress exceptions)

---

## Example: Iterative Updates

```python
with NumPack('model.npk') as npk:
    with npk.batch_mode():
        weights = npk.load('weights')
        
        for epoch in range(100):
            # Modify weights
            weights += learning_rate * gradients
            
            # This only updates cache
            npk.save({'weights': weights})
        
        # Disk write happens here (once)
```

## Performance

- **25-37x faster** than individual save/load cycles
- Best for repeated read-modify-write patterns
- Memory usage scales with total cached array size

---

# WritableBatchMode

Zero-copy mode using memory-mapped files.

## Class Definition

```python
class WritableBatchMode:
    def __init__(self, numpack_instance: NumPack)
```

## Usage

```python
from numpack import NumPack

with NumPack('data.npk') as npk:
    with npk.writable_batch_mode() as wb:
        arr = wb.load('array')
        
        # Direct in-place modification
        arr *= 2
        arr[0:100] = new_values
        
        # Optional: wb.save() is a no-op
        wb.save({'array': arr})
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `numpack_instance` | `NumPack` | *required* | Parent NumPack instance |

## Methods

### `load(array_name)`

Load a writable memory-mapped array.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array_name` | `str` | *required* | Name of the array to load |

#### Returns

- `np.ndarray`: A writable view backed by the file

#### Raises

| Exception | Condition |
|-----------|-----------|
| `KeyError` | If array doesn't exist |
| `ValueError` | If dtype is unsupported |

#### Example

```python
with npk.writable_batch_mode() as wb:
    arr = wb.load('features')
    arr *= 2  # Directly modifies file
```

---

### `save(arrays_dict)`

No-op method for API symmetry.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arrays_dict` | `Dict[str, np.ndarray]` | *required* | Ignored |

#### Returns

- `None`

#### Notes

In writable batch mode, modifications are applied directly to the memory-mapped files. This method exists only for API compatibility with `batch_mode()`.

---

### `__enter__()`

Enter writable batch mode.

#### Returns

- `WritableBatchMode`: The context manager instance

---

### `__exit__(exc_type, exc_val, exc_tb)`

Flush and close all memory maps.

#### Returns

- `bool`: Always `False`

---

## How It Works

1. **On `load()`**: Opens file with `mmap` in write mode
2. **During Operations**: NumPy array directly references the mapped memory
3. **On Exit**: Flushes `mmap` buffers and closes file handles

## Example: Large-Scale Updates

```python
with NumPack('huge_data.npk') as npk:
    with npk.writable_batch_mode() as wb:
        # Near-zero memory usage regardless of array size
        arr = wb.load('huge_array')  # 10GB array
        
        # Update in batches
        for i in range(0, len(arr), 10000):
            arr[i:i+10000] *= scale_factor
        
        # Changes automatically persisted
```

## Performance

- **174x faster** than standard load/save cycles
- Near-zero RAM usage (OS manages page cache)
- Best for in-place value updates

## Limitations

| Operation | Supported |
|-----------|-----------|
| In-place value changes | ✅ |
| Append rows | ❌ |
| Change shape | ❌ |
| Change dtype | ❌ |

---

# Comparison

## When to Use Each Mode

| Scenario | Recommended Mode |
|----------|-----------------|
| Iterative training updates | `batch_mode()` |
| Large array value scaling | `writable_batch_mode()` |
| Appending data in batches | `batch_mode()` |
| Memory-constrained environment | `writable_batch_mode()` |
| Small arrays, many iterations | `batch_mode()` |
| Huge arrays, simple updates | `writable_batch_mode()` |

## Code Comparison

```python
# batch_mode - flexible, uses RAM
with npk.batch_mode():
    arr = npk.load('array')
    arr = np.concatenate([arr, new_rows])  # Shape change OK
    npk.save({'array': arr})

# writable_batch_mode - zero-copy, in-place only
with npk.writable_batch_mode() as wb:
    arr = wb.load('array')
    arr *= 2  # In-place only
    # Cannot change shape
```
