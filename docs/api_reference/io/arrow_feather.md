# Arrow/Feather Conversion API Reference

Functions for converting between PyArrow tables/arrays and NumPack files.

## Dependencies

```bash
pip install pyarrow
```

---

## Memory Functions (Zero-Copy)

### `from_arrow(table_or_array, output_path, array_name=None, drop_if_exists=False)`

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
from numpack.io import from_arrow

# From Table
table = pa.table({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
from_arrow(table, 'output.npk')

# From Array
arr = pa.array([1.0, 2.0, 3.0, 4.0])
from_arrow(arr, 'output.npk', array_name='values')
```

---

### `to_arrow(input_path, array_name=None)`

Load a NumPack array as a PyArrow Array.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input NumPack directory path |
| `array_name` | `str` or `None` | `None` | Array name to load (inferred if single) |

#### Returns

- `pa.Array`: The loaded PyArrow array

#### Raises

| Exception | Condition |
|-----------|-----------|
| `DependencyError` | If PyArrow is not installed |
| `ValueError` | If multiple arrays exist and `array_name` not specified |

#### Example

```python
from numpack.io import to_arrow

arrow_arr = to_arrow('input.npk', array_name='values')
print(arrow_arr.type)
```

---

## File Functions (Streaming)

### `from_feather_file(input_path, output_path, columns=None, drop_if_exists=False, chunk_size=DEFAULT_CHUNK_SIZE)`

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
from numpack.io import from_feather_file

# Import all columns
from_feather_file('data.feather', 'output.npk')

# Import specific columns
from_feather_file('data.feather', 'output.npk', columns=['col1', 'col2'])
```

---

### `to_feather_file(input_path, output_path, array_name=None, chunk_size=DEFAULT_CHUNK_SIZE)`

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
from numpack.io import to_feather_file

to_feather_file('input.npk', 'output.feather')
```

---

## Legacy Aliases

```python
from_feather = from_feather_file  # Alias
to_feather = to_feather_file      # Alias
```

---

## Usage Examples

### Zero-Copy Workflow

```python
import pyarrow as pa
from numpack.io import from_arrow, to_arrow

# Create Arrow table
table = pa.table({
    'features': pa.array(np.random.rand(1000).astype(np.float32)),
    'labels': pa.array(np.random.randint(0, 10, 1000))
})

# Save to NumPack (uses zero-copy where possible)
from_arrow(table, 'data.npk')

# Load back
features = to_arrow('data.npk', array_name='features')
```

### Feather File Conversion

```python
from numpack.io import from_feather_file, to_feather_file

# Import from Feather
from_feather_file('dataset.feather', 'dataset.npk')

# Export back to Feather
to_feather_file('dataset.npk', 'exported.feather')
```
