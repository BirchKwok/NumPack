# NumPy Conversion API Reference

Functions for converting between NumPy `.npy`/`.npz` files and NumPack.

## Dependencies

None (NumPy is a core dependency of NumPack).

---

## Functions

### `from_npy(input_path, output_path, array_name=None, drop_if_exists=False, chunk_size=DEFAULT_CHUNK_SIZE)`

Import a NumPy `.npy` or `.npz` file into NumPack.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input `.npy` or `.npz` file path |
| `output_path` | `str` or `Path` | *required* | Output NumPack directory path |
| `array_name` | `str` or `None` | `None` | Array name for `.npy` files (default: filename stem) |
| `drop_if_exists` | `bool` | `False` | Overwrite if exists |
| `chunk_size` | `int` | `DEFAULT_CHUNK_SIZE` | Chunk size for streaming large files |

#### Returns

- `None`

#### Notes

- For `.npy` files: Saves as single array
- For `.npz` files: Saves each contained array separately
- Large files (>1GB) are streamed using memory mapping

#### Example

```python
from numpack.io import from_npy

# Import .npy file
from_npy('data.npy', 'output.npk')
from_npy('data.npy', 'output.npk', array_name='features')

# Import .npz file (all arrays)
from_npy('data.npz', 'output.npk')
```

---

### `to_npy(input_path, output_path, array_names=None, single_file=None, chunk_size=DEFAULT_CHUNK_SIZE)`

Export NumPack arrays to NumPy `.npy` or `.npz` format.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input NumPack directory path |
| `output_path` | `str` or `Path` | *required* | Output `.npy` or `.npz` file path |
| `array_names` | `List[str]` or `None` | `None` | Arrays to export (all if `None`) |
| `single_file` | `bool` or `None` | `None` | Force `.npy` (single array) or `.npz` (multiple). Auto-detected from extension if `None` |
| `chunk_size` | `int` | `DEFAULT_CHUNK_SIZE` | Chunk size for streaming |

#### Returns

- `None`

#### Notes

- `.npy` output: Only works with single array
- `.npz` output: Can contain multiple arrays
- Extension determines format if `single_file` is `None`

#### Example

```python
from numpack.io import to_npy

# Export single array to .npy
to_npy('input.npk', 'output.npy', array_names=['features'])

# Export multiple arrays to .npz
to_npy('input.npk', 'output.npz')

# Export all arrays to .npz
to_npy('input.npk', 'all_data.npz')
```

---

## Deprecated Aliases

The following functions are deprecated and will be removed in version 0.6.0:

- `from_numpy` -> Use `from_npy` instead
- `to_numpy` -> Use `to_npy` instead

---

## Usage Examples

### Migration from NumPy

```python
import numpy as np
from numpack.io import from_npy, to_npy

# Save original NumPy data
np.save('legacy_data.npy', large_array)

# Convert to NumPack for better performance
from_npy('legacy_data.npy', 'data.npk')

# Work with NumPack...
# Later, export back if needed
to_npy('data.npk', 'exported.npy')
```

### Batch Conversion

```python
from pathlib import Path
from numpack.io import from_npy

# Convert all .npy files in a directory
for npy_file in Path('npy_files').glob('*.npy'):
    output = f'npk_files/{npy_file.stem}.npk'
    from_npy(npy_file, output)
```

### NPZ File Handling

```python
from numpack.io import from_npy, to_npy

# Import NPZ with multiple arrays
from_npy('model_weights.npz', 'model.npk')
# Creates: model.npk with arrays named by NPZ keys

# Export back to NPZ
to_npy('model.npk', 'exported_weights.npz')
```
