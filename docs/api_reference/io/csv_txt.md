# CSV/TXT Conversion API Reference

Functions for converting between CSV/TXT text files and NumPack.

## Dependencies

None (uses NumPy's built-in text loading).

---

## CSV Functions

### `from_csv(input_path, output_path, array_name=None, delimiter=',', skip_header=0, dtype=None, drop_if_exists=False, chunk_size=DEFAULT_CHUNK_SIZE)`

Import a CSV file into NumPack.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input CSV file path |
| `output_path` | `str` or `Path` | *required* | Output NumPack directory path |
| `array_name` | `str` or `None` | `None` | Array name (default: filename stem) |
| `delimiter` | `str` | `','` | Field delimiter |
| `skip_header` | `int` | `0` | Number of header rows to skip |
| `dtype` | `np.dtype` or `None` | `None` | Target dtype (auto-detected if `None`) |
| `drop_if_exists` | `bool` | `False` | Overwrite if exists |
| `chunk_size` | `int` | `DEFAULT_CHUNK_SIZE` | Chunk size for streaming |

#### Returns

- `None`

#### Example

```python
from numpack.io import from_csv

# Basic import
from_csv('data.csv', 'output.npk')

# With options
from_csv(
    'data.csv',
    'output.npk',
    array_name='features',
    delimiter=',',
    skip_header=1,
    dtype=np.float32
)
```

---

### `to_csv(input_path, output_path, array_name=None, delimiter=',', header=None, fmt='%.6g', chunk_size=DEFAULT_CHUNK_SIZE)`

Export a NumPack array to CSV format.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input NumPack directory path |
| `output_path` | `str` or `Path` | *required* | Output CSV file path |
| `array_name` | `str` or `None` | `None` | Array to export |
| `delimiter` | `str` | `','` | Field delimiter |
| `header` | `str` or `None` | `None` | Header line to write |
| `fmt` | `str` | `'%.6g'` | Number format string |
| `chunk_size` | `int` | `DEFAULT_CHUNK_SIZE` | Chunk size for streaming |

#### Returns

- `None`

#### Example

```python
from numpack.io import to_csv

# Basic export
to_csv('input.npk', 'output.csv')

# With custom format
to_csv(
    'input.npk',
    'output.csv',
    array_name='features',
    delimiter='\t',
    header='col1,col2,col3',
    fmt='%.4f'
)
```

---

## TXT Functions

### `from_txt(input_path, output_path, array_name=None, delimiter=None, skip_header=0, dtype=None, drop_if_exists=False, chunk_size=DEFAULT_CHUNK_SIZE)`

Import a text file into NumPack.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input text file path |
| `output_path` | `str` or `Path` | *required* | Output NumPack directory path |
| `array_name` | `str` or `None` | `None` | Array name (default: filename stem) |
| `delimiter` | `str` or `None` | `None` | Field delimiter (whitespace if `None`) |
| `skip_header` | `int` | `0` | Number of header rows to skip |
| `dtype` | `np.dtype` or `None` | `None` | Target dtype |
| `drop_if_exists` | `bool` | `False` | Overwrite if exists |
| `chunk_size` | `int` | `DEFAULT_CHUNK_SIZE` | Chunk size for streaming |

#### Returns

- `None`

#### Example

```python
from numpack.io import from_txt

# Whitespace-delimited file
from_txt('data.txt', 'output.npk')

# Tab-delimited file
from_txt('data.txt', 'output.npk', delimiter='\t')
```

---

### `to_txt(input_path, output_path, array_name=None, delimiter='\t', header=None, fmt='%.6g', chunk_size=DEFAULT_CHUNK_SIZE)`

Export a NumPack array to text format.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input NumPack directory path |
| `output_path` | `str` or `Path` | *required* | Output text file path |
| `array_name` | `str` or `None` | `None` | Array to export |
| `delimiter` | `str` | `'\t'` | Field delimiter |
| `header` | `str` or `None` | `None` | Header line |
| `fmt` | `str` | `'%.6g'` | Number format |
| `chunk_size` | `int` | `DEFAULT_CHUNK_SIZE` | Chunk size for streaming |

#### Returns

- `None`

#### Example

```python
from numpack.io import to_txt

to_txt('input.npk', 'output.txt', delimiter=' ')
```

---

## Usage Examples

### Data Import Pipeline

```python
from numpack.io import from_csv
from numpack import NumPack

# Import CSV with header
from_csv('raw_data.csv', 'data.npk', skip_header=1, dtype=np.float32)

# Process
with NumPack('data.npk') as npk:
    data = npk.load('raw_data')
    normalized = (data - data.mean()) / data.std()
    npk.save({'normalized': normalized})
```

### Large File Streaming

```python
from numpack.io import from_csv, to_csv

# Large CSV files are automatically streamed
from_csv('huge_dataset.csv', 'data.npk')  # Memory-efficient

# Export in chunks
to_csv('data.npk', 'output.csv')  # Also streamed
```
