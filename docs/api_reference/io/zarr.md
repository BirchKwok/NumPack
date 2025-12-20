# Zarr Conversion API Reference

Functions for converting between Zarr arrays and NumPack.

## Dependencies

```bash
pip install zarr
```

---

## Functions

### `from_zarr(input_path, output_path, array_names=None, drop_if_exists=False, chunk_size=DEFAULT_CHUNK_SIZE)`

Import Zarr arrays into NumPack.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input Zarr directory path |
| `output_path` | `str` or `Path` | *required* | Output NumPack directory path |
| `array_names` | `List[str]` or `None` | `None` | Arrays to import (all if `None`) |
| `drop_if_exists` | `bool` | `False` | Overwrite if exists |
| `chunk_size` | `int` | `DEFAULT_CHUNK_SIZE` | Chunk size for streaming |

#### Returns

- `None`

#### Raises

| Exception | Condition |
|-----------|-----------|
| `DependencyError` | If zarr is not installed |

#### Example

```python
from numpack.io import from_zarr

# Import all arrays
from_zarr('data.zarr', 'output.npk')

# Import specific arrays
from_zarr('data.zarr', 'output.npk', array_names=['features', 'labels'])
```

---

### `to_zarr(input_path, output_path, array_names=None, chunks=None, compressor=None, chunk_size=DEFAULT_CHUNK_SIZE)`

Export NumPack arrays to Zarr format.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input NumPack directory path |
| `output_path` | `str` or `Path` | *required* | Output Zarr directory path |
| `array_names` | `List[str]` or `None` | `None` | Arrays to export (all if `None`) |
| `chunks` | `Tuple[int, ...]` or `None` | `None` | Zarr chunk shape (auto if `None`) |
| `compressor` | Zarr compressor or `None` | `None` | Compression algorithm |
| `chunk_size` | `int` | `DEFAULT_CHUNK_SIZE` | Streaming chunk size |

#### Returns

- `None`

#### Raises

| Exception | Condition |
|-----------|-----------|
| `DependencyError` | If zarr is not installed |

#### Example

```python
from numpack.io import to_zarr
import zarr

# Export with defaults
to_zarr('input.npk', 'output.zarr')

# Export with custom compression
compressor = zarr.Blosc(cname='zstd', clevel=3)
to_zarr('input.npk', 'output.zarr', compressor=compressor)

# Export with specific chunks
to_zarr('input.npk', 'output.zarr', chunks=(1000, 128))
```

---

## Usage Examples

### Cloud-Native Workflow

```python
from numpack.io import from_zarr, to_zarr
from numpack import NumPack

# Zarr is commonly used for cloud storage
from_zarr('s3://bucket/data.zarr', 'local.npk')

# Process locally with NumPack
with NumPack('local.npk') as npk:
    data = npk.load('features')
    processed = transform(data)
    npk.save({'features': processed})

# Export back to Zarr
to_zarr('local.npk', 'output.zarr')
```

### Chunked Array Processing

```python
import zarr
from numpack.io import to_zarr

# Export with optimized chunking for your access pattern
to_zarr(
    'input.npk',
    'output.zarr',
    chunks=(10000, 64),  # Optimize for row-wise access
    compressor=zarr.Blosc(cname='lz4', clevel=5)
)
```
