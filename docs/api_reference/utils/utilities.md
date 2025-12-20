# Utility Functions API Reference

Miscellaneous utility functions provided by NumPack.

## Backend Information

### `get_backend_info()`

Get information about the active NumPack backend.

#### Import

```python
from numpack import get_backend_info
```

#### Parameters

None

#### Returns

- `Dict[str, Any]`: Backend information

#### Return Value Structure

```python
{
    'backend_type': str,    # Always 'rust'
    'platform': str,        # 'Darwin', 'Linux', 'Windows'
    'is_windows': bool,     # True on Windows
    'version': str,         # NumPack version
}
```

#### Example

```python
from numpack import get_backend_info

info = get_backend_info()
print(f"Backend: {info['backend_type']}")
print(f"Platform: {info['platform']}")
print(f"Version: {info['version']}")
```

---

### `force_cleanup_windows_handles()`

Force cleanup of Windows file handles.

#### Import

```python
from numpack import force_cleanup_windows_handles
```

#### Parameters

None

#### Returns

- `bool`: Always `True`

#### Notes

- Kept for backward compatibility
- With the Rust backend, most cleanup is automatic
- Triggers a garbage collection pass

#### Example

```python
from numpack import force_cleanup_windows_handles

# After closing many NumPack instances
force_cleanup_windows_handles()
```

---

## IO Utility Functions

### `get_file_size(path)`

Get file or directory size in bytes.

#### Import

```python
from numpack.io import get_file_size
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` or `Path` | *required* | File or directory path |

#### Returns

- `int`: Size in bytes

#### Notes

- For files: Returns file size
- For directories: Returns total size of all files (recursive)

#### Example

```python
from numpack.io import get_file_size

size = get_file_size('data.npk')
print(f"Size: {size / 1e6:.1f} MB")
```

---

### `is_large_file(path, threshold=LARGE_FILE_THRESHOLD)`

Check if a file exceeds the large file threshold.

#### Import

```python
from numpack.io import is_large_file, LARGE_FILE_THRESHOLD
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` or `Path` | *required* | File path |
| `threshold` | `int` | `1GB` | Threshold in bytes |

#### Returns

- `bool`: `True` if file exceeds threshold

#### Example

```python
from numpack.io import is_large_file

if is_large_file('data.npk'):
    print("Using streaming mode")
else:
    print("Loading directly")
```

---

### `estimate_chunk_rows(shape, dtype, target_chunk_bytes=DEFAULT_CHUNK_SIZE)`

Estimate optimal number of rows per chunk for streaming.

#### Import

```python
from numpack.io import estimate_chunk_rows
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `Tuple[int, ...]` | *required* | Array shape |
| `dtype` | `np.dtype` | *required* | Array dtype |
| `target_chunk_bytes` | `int` | `100MB` | Target chunk size |

#### Returns

- `int`: Recommended rows per chunk

#### Example

```python
from numpack.io import estimate_chunk_rows
import numpy as np

shape = (1000000, 128)
dtype = np.float32

rows = estimate_chunk_rows(shape, dtype)
print(f"Recommended chunk: {rows} rows")
```

---

## Constants

### `LARGE_FILE_THRESHOLD`

Default threshold for considering a file "large" (triggers streaming).

- **Value**: `1 * 1024 * 1024 * 1024` (1 GB)

### `DEFAULT_CHUNK_SIZE`

Default chunk size for streaming operations.

- **Value**: `100 * 1024 * 1024` (100 MB)

### `DEFAULT_BATCH_ROWS`

Default number of rows per batch in streaming operations.

- **Value**: `100000`

#### Import

```python
from numpack.io import (
    LARGE_FILE_THRESHOLD,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_BATCH_ROWS,
)
```

---

## Exception Classes

### `DependencyError`

Raised when an optional dependency is not installed.

#### Import

```python
from numpack.io import DependencyError
```

#### Example

```python
from numpack.io import DependencyError, from_torch

try:
    from_torch(tensor, 'output.npk')
except DependencyError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install torch")
```
