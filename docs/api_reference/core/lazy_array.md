# LazyArray Class API Reference

`LazyArray` provides memory-mapped lazy loading for efficient access to large arrays.

## Class Definition

```python
class LazyArray:
    """Memory-mapped array with lazy loading support."""
```

## Overview

`LazyArray` is returned when calling `NumPack.load()` with `lazy=True`. It provides:

- **Zero-copy access**: Data is memory-mapped directly from disk
- **On-demand loading**: Only accessed portions are loaded into memory
- **NumPy compatibility**: Supports standard indexing operations

## Getting a LazyArray

```python
from numpack import NumPack

with NumPack('data.npk') as npk:
    # Load as LazyArray
    lazy = npk.load('features', lazy=True)
```

---

## Properties

### `shape`

Shape of the array.

- **Type**: `Tuple[int, ...]`

```python
lazy = npk.load('features', lazy=True)
print(lazy.shape)  # (1000, 128)
```

---

### `dtype`

Data type of the array.

- **Type**: `np.dtype`

```python
print(lazy.dtype)  # float32
```

---

### `ndim`

Number of dimensions.

- **Type**: `int`

```python
print(lazy.ndim)  # 2
```

---

### `size`

Total number of elements.

- **Type**: `int`

```python
print(lazy.size)  # 128000
```

---

## Indexing Operations

### `__getitem__(key)`

Access array elements using standard NumPy indexing.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `int`, `slice`, `List[int]`, `np.ndarray` | Index specification |

#### Returns

- `np.ndarray`: The selected elements

#### Examples

```python
lazy = npk.load('features', lazy=True)

# Single row
row = lazy[0]

# Slice
rows = lazy[100:200]

# Fancy indexing
selected = lazy[[0, 10, 20, 30]]

# Boolean indexing
mask = np.array([True, False, True, ...])
filtered = lazy[mask]
```

---

## Conversion Methods

### `__array__()`

Convert to NumPy array (implements array protocol).

#### Returns

- `np.ndarray`: Full array loaded into memory

#### Example

```python
lazy = npk.load('features', lazy=True)

# Explicit conversion
arr = np.asarray(lazy)

# Implicit conversion in NumPy operations
result = np.sum(lazy)  # Triggers conversion
```

---

## Use Cases

### Memory-Efficient Processing

```python
with NumPack('large_data.npk') as npk:
    lazy = npk.load('huge_array', lazy=True)
    
    # Only loads required rows
    for i in range(0, lazy.shape[0], 1000):
        batch = lazy[i:i+1000]
        process(batch)
```

### Random Sampling

```python
lazy = npk.load('features', lazy=True)

# Efficiently sample random rows
indices = np.random.choice(lazy.shape[0], size=100, replace=False)
sample = lazy[indices]
```

### Conditional Access

```python
# Load metadata first
metadata = npk.load('metadata')

# Find relevant indices
relevant_idx = np.where(metadata['category'] == 'A')[0]

# Load only relevant rows
lazy = npk.load('features', lazy=True)
relevant_features = lazy[relevant_idx]
```

---

## Performance Notes

| Operation | Performance |
|-----------|-------------|
| Sequential access | Excellent (prefetching) |
| Random access | Good (OS page caching) |
| Full load | Same as eager loading |
| Partial access | Much faster than full load |

### Best Practices

1. **Use for large arrays**: Most beneficial when array > available RAM
2. **Batch access patterns**: Access contiguous rows when possible
3. **Avoid repeated small accesses**: Batch your index requests
4. **Let OS manage caching**: Repeated access to same regions is fast

---

## Comparison with Eager Loading

| Aspect | Lazy (`lazy=True`) | Eager (`lazy=False`) |
|--------|-------------------|---------------------|
| Initial load time | Near instant | Proportional to size |
| Memory usage | On-demand | Full array in RAM |
| Random access | Slightly slower | Fastest |
| Best for | Large arrays, partial access | Small arrays, full access |
