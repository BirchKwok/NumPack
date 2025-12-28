# Arrow/Feather Conversion API Reference

Functions for converting between PyArrow tables/arrays and NumPack files.

## Dependencies

```bash
pip install pyarrow
```

---

## Memory Functions (Zero-Copy)

### `from_table(table_or_array, output_path, array_name=None, drop_if_exists=False)`

Save a PyArrow Table or Array to NumPack.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `table_or_array` | `pa.Table` or `pa.Array` | *required* | PyArrow data to save |
| `output_path` | `str` or `Path` | *required* | Output NumPack directory path |
| `array_name` | `str` or `None` | `None` | Array name (default: `'data'` or column names) |
| `drop_if_exists` | `bool` | `False` | Overwrite if exists |

#### Returns

- `None`

#### Raises

| Exception | Condition |
|-----------|-----------|
| `DependencyError` | If PyArrow is not installed |

#### Example

```python
import pyarrow as pa
from numpack.io import from_table

# From Table
table = pa.table({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
from_table(table, 'output.npk')

# From Array
arr = pa.array([1.0, 2.0, 3.0, 4.0])
from_table(arr, 'output.npk', array_name='values')
```

---

### `to_table(input_path, array_name=None)`

Load a NumPack array as a PyArrow Table.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input NumPack directory path |
| `array_name` | `str` or `None` | `None` | Array name to load (inferred if single) |

#### Returns

- `pa.Table`: The loaded PyArrow table

#### Raises

| Exception | Condition |
|-----------|-----------|
| `DependencyError` | If PyArrow is not installed |
| `ValueError` | If multiple arrays exist and `array_name` not specified |

#### Example

```python
from numpack.io import to_table

arrow_table = to_table('input.npk', array_name='values')
print(arrow_table.column_names)
```

---

## File Functions (Streaming)

### `from_feather(input_path, output_path, columns=None, drop_if_exists=False, chunk_size=DEFAULT_CHUNK_SIZE)`

Convert a Feather file to NumPack.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input Feather file path |
| `output_path` | `str` or `Path` | *required* | Output NumPack directory path |
| `columns` | `List[str]` or `None` | `None` | Columns to import (all if `None`) |
| `drop_if_exists` | `bool` | `False` | Overwrite if exists |
| `chunk_size` | `int` | `DEFAULT_CHUNK_SIZE` | Chunk size for streaming |

#### Returns

- `None`

#### Example

```python
from numpack.io import from_feather

# Import all columns
from_feather('data.feather', 'output.npk')

# Import specific columns
from_feather('data.feather', 'output.npk', columns=['col1', 'col2'])
```

---

### `to_feather(input_path, output_path, array_name=None, chunk_size=DEFAULT_CHUNK_SIZE)`

Export NumPack arrays to a Feather file.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input NumPack directory path |
| `output_path` | `str` or `Path` | *required* | Output Feather file path |
| `array_name` | `str` or `None` | `None` | Array to export (all if `None`) |
| `chunk_size` | `int` | `DEFAULT_CHUNK_SIZE` | Chunk size for streaming |

#### Returns

- `None`

#### Example

```python
from numpack.io import to_feather

to_feather('input.npk', 'output.feather')
```

---

## Deprecated Aliases

The following functions are deprecated and will be removed in version 0.6.0:

- `from_arrow` -> Use `from_table` instead (memory conversion)
- `to_arrow` -> Use `to_table` instead (memory conversion)

Aliases for clarity (not deprecated):

- `from_feather_file` = `from_feather`
- `to_feather_file` = `to_feather`

---

## Usage Examples

### Zero-Copy Workflow

```python
import pyarrow as pa
from numpack.io import from_table, to_table

# Create Arrow table
table = pa.table({
    'features': pa.array(np.random.rand(1000).astype(np.float32)),
    'labels': pa.array(np.random.randint(0, 10, 1000))
})

# Save to NumPack (uses zero-copy where possible)
from_table(table, 'data.npk')

# Load back
features_table = to_table('data.npk', array_name='features')
```

### Feather File Conversion

```python
from numpack.io import from_feather, to_feather

# Import from Feather
from_feather('dataset.feather', 'dataset.npk')

# Export back to Feather
to_feather('dataset.npk', 'exported.feather')
```
