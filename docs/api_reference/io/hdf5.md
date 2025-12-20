# HDF5 Conversion API Reference

Functions for converting between HDF5 files and NumPack.

## Dependencies

```bash
pip install h5py
```

---

## Functions

### `from_hdf5(input_path, output_path, dataset_names=None, drop_if_exists=False, chunk_size=DEFAULT_CHUNK_SIZE)`

Import datasets from an HDF5 file into NumPack.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input HDF5 file path |
| `output_path` | `str` or `Path` | *required* | Output NumPack directory path |
| `dataset_names` | `List[str]` or `None` | `None` | Datasets to import (all if `None`) |
| `drop_if_exists` | `bool` | `False` | Overwrite if exists |
| `chunk_size` | `int` | `DEFAULT_CHUNK_SIZE` | Chunk size for streaming |

#### Returns

- `None`

#### Raises

| Exception | Condition |
|-----------|-----------|
| `DependencyError` | If h5py is not installed |

#### Notes

- Nested HDF5 groups are flattened with `/` separators
- Large datasets are streamed to minimize memory usage

#### Example

```python
from numpack.io import from_hdf5

# Import all datasets
from_hdf5('data.h5', 'output.npk')

# Import specific datasets
from_hdf5('data.h5', 'output.npk', dataset_names=['train/features', 'train/labels'])
```

---

### `to_hdf5(input_path, output_path, array_names=None, compression='gzip', chunk_size=DEFAULT_CHUNK_SIZE)`

Export NumPack arrays to an HDF5 file.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input NumPack directory path |
| `output_path` | `str` or `Path` | *required* | Output HDF5 file path |
| `array_names` | `List[str]` or `None` | `None` | Arrays to export (all if `None`) |
| `compression` | `str` or `None` | `'gzip'` | HDF5 compression filter |
| `chunk_size` | `int` | `DEFAULT_CHUNK_SIZE` | Chunk size for streaming |

#### Returns

- `None`

#### Raises

| Exception | Condition |
|-----------|-----------|
| `DependencyError` | If h5py is not installed |

#### Example

```python
from numpack.io import to_hdf5

# Export all arrays
to_hdf5('input.npk', 'output.h5')

# Export with specific compression
to_hdf5('input.npk', 'output.h5', compression='lzf')

# Export specific arrays
to_hdf5('input.npk', 'output.h5', array_names=['features'])
```

---

## Usage Examples

### Scientific Data Pipeline

```python
from numpack.io import from_hdf5, to_hdf5
from numpack import NumPack

# Import from HDF5 (common in scientific computing)
from_hdf5('experiment_data.h5', 'data.npk')

# Process with NumPack
with NumPack('data.npk') as npk:
    measurements = npk.load('measurements')
    processed = apply_calibration(measurements)
    npk.save({'processed': processed})

# Export back to HDF5 for sharing
to_hdf5('data.npk', 'results.h5')
```

### Nested HDF5 Structure

```python
from numpack.io import from_hdf5

# HDF5 with nested groups:
# /train/features
# /train/labels
# /test/features
# /test/labels

from_hdf5('dataset.h5', 'data.npk')

# NumPack arrays will be named:
# 'train/features', 'train/labels', 'test/features', 'test/labels'
```
