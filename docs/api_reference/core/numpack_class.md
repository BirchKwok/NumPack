# NumPack Class API Reference

The `NumPack` class is the main interface for high-performance array storage.

## Class Definition

```python
class NumPack:
    def __init__(
        self,
        filename: Union[str, Path],
        drop_if_exists: bool = False,
        strict_context_mode: bool = False,
        warn_no_context: bool = None,
        force_gc_on_close: bool = False
    )
```

## Constructor

### `NumPack(filename, **options)`

Create a NumPack instance for array storage.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | `str` or `Path` | *required* | NumPack directory path |
| `drop_if_exists` | `bool` | `False` | Delete existing directory before creating |
| `strict_context_mode` | `bool` | `False` | Require context manager usage |
| `warn_no_context` | `bool` | `None` | Warn if not using context manager (defaults to `True` on Windows) |
| `force_gc_on_close` | `bool` | `False` | Force garbage collection on close |

#### Returns

- `NumPack`: A new NumPack instance (not yet opened)

#### Example

```python
from numpack import NumPack

# Basic usage
npk = NumPack('data.npk')

# With options
npk = NumPack(
    'data.npk',
    drop_if_exists=True,
    strict_context_mode=True
)
```

#### Notes

- The file is **not** opened automatically. Use `open()` or a context manager.
- On Windows, `warn_no_context` defaults to `True` for reliability.

---

## Instance Methods

### `open()`

Open the NumPack file for operations.

#### Parameters

None

#### Returns

- `None`

#### Example

```python
npk = NumPack('data.npk')
npk.open()
# ... operations ...
npk.close()
```

#### Notes

- Idempotent: calling `open()` on an already-open instance is a no-op.
- Creates the directory if it doesn't exist.

---

### `close(force_gc=None)`

Close the NumPack instance and release resources.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force_gc` | `bool` or `None` | `None` | Force garbage collection after close |

#### Returns

- `None`

#### Example

```python
npk = NumPack('data.npk')
npk.open()
npk.save({'array': data})
npk.close()
```

#### Notes

- Idempotent: calling `close()` on an already-closed instance is a no-op.
- Flushes any cached data before closing.

---

### `save(arrays)`

Save arrays to the NumPack file.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arrays` | `Dict[str, np.ndarray]` | *required* | Dictionary mapping array names to NumPy arrays |

#### Returns

- `None`

#### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | If `arrays` is not a dictionary |
| `RuntimeError` | If instance is not opened or closed |

#### Example

```python
import numpy as np
from numpack import NumPack

with NumPack('data.npk', drop_if_exists=True) as npk:
    npk.save({
        'features': np.random.rand(1000, 128),
        'labels': np.array([0, 1, 0, 1, ...])
    })
```

#### Notes

- Arrays are saved atomically.
- Existing arrays with the same name are overwritten.

---

### `load(array_name, lazy=False)`

Load an array from the NumPack file.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array_name` | `str` | *required* | Name of the array to load |
| `lazy` | `bool` | `False` | Return `LazyArray` instead of loading into memory |

#### Returns

- `np.ndarray`: The loaded array (if `lazy=False`)
- `LazyArray`: A lazy array reference (if `lazy=True`)

#### Raises

| Exception | Condition |
|-----------|-----------|
| `KeyError` | If `array_name` does not exist |
| `RuntimeError` | If instance is not opened or closed |

#### Example

```python
with NumPack('data.npk') as npk:
    # Eager loading
    features = npk.load('features')
    
    # Lazy loading (memory-mapped)
    lazy_features = npk.load('features', lazy=True)
```

---

### `replace(arrays, indexes)`

Replace values at specific row indexes.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arrays` | `Dict[str, np.ndarray]` | *required* | Dictionary of replacement values |
| `indexes` | `int`, `List[int]`, `np.ndarray`, or `slice` | *required* | Row indexes to replace |

#### Returns

- `None`

#### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | If `arrays` is not a dict or `indexes` type is invalid |
| `RuntimeError` | If instance is not opened or closed |

#### Example

```python
with NumPack('data.npk') as npk:
    # Replace single row
    npk.replace({'features': new_row}, 0)
    
    # Replace multiple rows
    npk.replace({'features': new_rows}, [0, 5, 10])
    
    # Replace range
    npk.replace({'features': new_rows}, slice(0, 10))
```

---

### `append(arrays)`

Append rows to existing arrays.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arrays` | `Dict[str, np.ndarray]` | *required* | Dictionary of arrays to append |

#### Returns

- `None`

#### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | If `arrays` is not a dictionary |
| `RuntimeError` | If instance is not opened or closed |

#### Example

```python
with NumPack('data.npk') as npk:
    # Append new rows
    new_data = np.random.rand(100, 128)
    npk.append({'features': new_data})
```

#### Notes

- The appended array must have the same shape (except for the first dimension).

---

### `drop(array_name, indexes=None)`

Drop arrays or specific rows from arrays.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array_name` | `str` or `List[str]` | *required* | Array name(s) to drop |
| `indexes` | `int`, `List[int]`, `np.ndarray`, or `None` | `None` | Row indexes to drop. If `None`, drops entire array(s) |

#### Returns

- `None`

#### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | If `indexes` type is invalid |
| `RuntimeError` | If instance is not opened or closed |

#### Example

```python
with NumPack('data.npk') as npk:
    # Drop entire array
    npk.drop('temp_array')
    
    # Drop multiple arrays
    npk.drop(['array1', 'array2'])
    
    # Drop specific rows
    npk.drop('features', [0, 1, 2])
```

#### Notes

- Row deletion is logical (bitmap-based) for performance.
- Use `update()` to physically compact after many deletions.

---

### `getitem(array_name, indexes)`

Random access to specific rows.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array_name` | `str` | *required* | Name of the array |
| `indexes` | `int`, `List[int]`, `np.ndarray`, or `slice` | *required* | Row indexes to retrieve |

#### Returns

- `np.ndarray`: The selected rows

#### Example

```python
with NumPack('data.npk') as npk:
    # Get single row
    row = npk.getitem('features', 0)
    
    # Get multiple rows
    rows = npk.getitem('features', [0, 10, 20])
    
    # Get range
    rows = npk.getitem('features', slice(0, 100))
```

---

### `stream_load(array_name, buffer_size=None)`

Stream array data in batches for memory-efficient processing.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array_name` | `str` | *required* | Name of the array to stream |
| `buffer_size` | `int` or `None` | `None` | Rows per batch. If `None`, yields one row at a time |

#### Returns

- `Iterator[np.ndarray]`: Iterator yielding batches

#### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | If `buffer_size` is not `None` and <= 0 |

#### Example

```python
with NumPack('data.npk') as npk:
    for batch in npk.stream_load('features', buffer_size=10000):
        process(batch)
```

---

### `get_shape(array_name)`

Get the shape of an array.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array_name` | `str` | *required* | Name of the array |

#### Returns

- `Tuple[int, ...]`: Shape of the array

#### Example

```python
with NumPack('data.npk') as npk:
    shape = npk.get_shape('features')
    print(f"Shape: {shape}")  # e.g., (1000, 128)
```

---

### `get_member_list()`

Get list of all array names in the file.

#### Parameters

None

#### Returns

- `List[str]`: List of array names

#### Example

```python
with NumPack('data.npk') as npk:
    arrays = npk.get_member_list()
    print(arrays)  # ['features', 'labels']
```

---

### `has_array(array_name)`

Check if an array exists.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array_name` | `str` | *required* | Name to check |

#### Returns

- `bool`: `True` if array exists

#### Example

```python
with NumPack('data.npk') as npk:
    if npk.has_array('features'):
        data = npk.load('features')
```

---

### `get_modify_time(array_name)`

Get the last modification timestamp of an array.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array_name` | `str` | *required* | Name of the array |

#### Returns

- `int` or `None`: Unix timestamp, or `None` if array doesn't exist

---

### `get_metadata()`

Get file metadata.

#### Parameters

None

#### Returns

- `Dict[str, Any]`: Metadata dictionary containing array information

#### Example

```python
with NumPack('data.npk') as npk:
    metadata = npk.get_metadata()
    print(metadata['arrays'])
```

---

### `clone(source_name, target_name)`

Clone an existing array to a new name.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_name` | `str` | *required* | Source array name |
| `target_name` | `str` | *required* | Target array name |

#### Returns

- `None`

#### Raises

| Exception | Condition |
|-----------|-----------|
| `KeyError` | If `source_name` does not exist |
| `ValueError` | If `target_name` already exists |

#### Example

```python
with NumPack('data.npk') as npk:
    npk.clone('original', 'backup')
```

---

### `reset()`

Remove all arrays from the file.

#### Parameters

None

#### Returns

- `None`

#### Example

```python
with NumPack('data.npk') as npk:
    npk.reset()
    print(npk.get_member_list())  # []
```

---

### `update(array_name)`

Physically compact an array by removing logically deleted rows.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array_name` | `str` | *required* | Name of the array to compact |

#### Returns

- `None`

#### Example

```python
with NumPack('data.npk') as npk:
    npk.drop('features', [0, 1, 2, 100, 200])
    npk.update('features')  # Reclaim disk space
```

---

### `get_io_stats()`

Get I/O statistics.

#### Parameters

None

#### Returns

- `Dict[str, Any]`: Backend statistics

#### Example

```python
with NumPack('data.npk') as npk:
    stats = npk.get_io_stats()
    print(stats['backend_type'])  # 'rust'
```

---

### `batch_mode(memory_limit=None)`

Enable in-memory caching for batch operations.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_limit` | `int` or `None` | `None` | Soft memory limit in MB |

#### Returns

- `BatchModeContext`: Context manager for batch operations

#### Example

```python
with NumPack('data.npk') as npk:
    with npk.batch_mode():
        for i in range(100):
            arr = npk.load('array')
            arr *= 2
            npk.save({'array': arr})
```

See [batch_modes.md](./batch_modes.md) for details.

---

### `writable_batch_mode()`

Enable zero-copy in-place updates via memory mapping.

#### Parameters

None

#### Returns

- `WritableBatchMode`: Context manager for writable batch operations

#### Example

```python
with NumPack('data.npk') as npk:
    with npk.writable_batch_mode() as wb:
        arr = wb.load('array')
        arr *= 2  # Direct modification
```

See [batch_modes.md](./batch_modes.md) for details.

---

## Properties

### `backend_type`

Backend identifier string.

- **Type**: `str`
- **Value**: Always `"rust"`

### `is_opened`

Whether the instance is currently opened.

- **Type**: `bool`

### `is_closed`

Whether the instance is currently closed.

- **Type**: `bool`

---

## Special Methods

### `__getitem__(key)`

Dictionary-style access to arrays.

```python
arr = npk['features']  # Equivalent to npk.load('features')
```

### `__iter__()`

Iterate over array names.

```python
for name in npk:
    print(name)
```

### `__enter__()` / `__exit__()`

Context manager support.

```python
with NumPack('data.npk') as npk:
    # Auto-opens on enter, auto-closes on exit
    pass
```

### `__repr__()`

String representation.

```python
print(npk)  # NumPack(data.npk, arrays=2, backend=rust)
```
