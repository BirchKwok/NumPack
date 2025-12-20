# Parquet Conversion API Reference

Functions for converting between Parquet files/tables and NumPack.

## Dependencies

```bash
pip install pyarrow
```

---

## Memory Functions

### `from_parquet_table(table, output_path, array_name=None, drop_if_exists=False)`

Save a PyArrow Table to NumPack.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `table` | `pa.Table` | *required* | PyArrow Table to save |
| `output_path` | `str` or `Path` | *required* | Output NumPack directory path |
| `array_name` | `str` or `None` | `None` | Array name (default: uses column names) |
| `drop_if_exists` | `bool` | `False` | Overwrite if exists |

#### Returns

- `None`

#### Example

```python
import pyarrow as pa
from numpack.io import from_parquet_table

table = pa.table({'col1': [1, 2, 3], 'col2': [4.0, 5.0, 6.0]})
from_parquet_table(table, 'output.npk')
```

---

### `to_parquet_table(input_path, array_name=None)`

Load NumPack arrays as a PyArrow Table.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input NumPack directory path |
| `array_name` | `str` or `None` | `None` | Array to load (all arrays if `None`) |

#### Returns

- `pa.Table`: PyArrow Table

#### Example

```python
from numpack.io import to_parquet_table

table = to_parquet_table('input.npk')
print(table.schema)
```

---

## File Functions (Streaming)

### `from_parquet_file(input_path, output_path, columns=None, drop_if_exists=False, chunk_size=DEFAULT_CHUNK_SIZE)`

Convert a Parquet file to NumPack (streaming).

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input Parquet file path |
| `output_path` | `str` or `Path` | *required* | Output NumPack directory path |
| `columns` | `List[str]` or `None` | `None` | Columns to import (all if `None`) |
| `drop_if_exists` | `bool` | `False` | Overwrite if exists |
| `chunk_size` | `int` | `DEFAULT_CHUNK_SIZE` | Chunk size for streaming |

#### Returns

- `None`

#### Example

```python
from numpack.io import from_parquet_file

from_parquet_file('data.parquet', 'output.npk')
from_parquet_file('data.parquet', 'output.npk', columns=['feature1', 'feature2'])
```

---

### `to_parquet_file(input_path, output_path, array_name=None, compression='snappy', chunk_size=DEFAULT_CHUNK_SIZE)`

Export NumPack arrays to a Parquet file.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input NumPack directory path |
| `output_path` | `str` or `Path` | *required* | Output Parquet file path |
| `array_name` | `str` or `None` | `None` | Array to export |
| `compression` | `str` | `'snappy'` | Compression codec |
| `chunk_size` | `int` | `DEFAULT_CHUNK_SIZE` | Chunk size for streaming |

#### Returns

- `None`

#### Example

```python
from numpack.io import to_parquet_file

to_parquet_file('input.npk', 'output.parquet')
to_parquet_file('input.npk', 'output.parquet', compression='zstd')
```

---

## Legacy Aliases

```python
from_parquet = from_parquet_file
to_parquet = to_parquet_file
```

---

## Usage Examples

### Data Pipeline

```python
from numpack.io import from_parquet_file, to_parquet_file
from numpack import NumPack

# Import Parquet data
from_parquet_file('raw_data.parquet', 'processed.npk')

# Process with NumPack
with NumPack('processed.npk') as npk:
    data = npk.load('data')
    data = normalize(data)
    npk.save({'data': data})

# Export back to Parquet
to_parquet_file('processed.npk', 'output.parquet')
```
