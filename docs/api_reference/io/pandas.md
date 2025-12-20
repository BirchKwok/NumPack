# Pandas Conversion API Reference

Functions for converting between Pandas DataFrames and NumPack.

## Dependencies

```bash
pip install pandas
```

---

## Functions

### `from_pandas(df, output_path, array_name='data', drop_if_exists=False, chunk_size=DEFAULT_CHUNK_SIZE)`

Import a Pandas DataFrame into NumPack.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | *required* | Input DataFrame |
| `output_path` | `str` or `Path` | *required* | Output NumPack directory path |
| `array_name` | `str` | `'data'` | Array name in NumPack |
| `drop_if_exists` | `bool` | `False` | Overwrite if exists |
| `chunk_size` | `int` | `DEFAULT_CHUNK_SIZE` | Chunk size for large DataFrames |

#### Returns

- `None`

#### Raises

| Exception | Condition |
|-----------|-----------|
| `DependencyError` | If pandas is not installed |

#### Notes

- DataFrame is converted to a NumPy array using `df.values`
- All columns must be numeric or convertible to numeric
- Large DataFrames (>1GB) are streamed in chunks

#### Example

```python
import pandas as pd
from numpack.io import from_pandas

df = pd.DataFrame({
    'feature1': [1.0, 2.0, 3.0],
    'feature2': [4.0, 5.0, 6.0],
    'label': [0, 1, 0]
})

from_pandas(df, 'data.npk', array_name='dataset')
```

---

### `to_pandas(input_path, array_name=None, columns=None)`

Export a NumPack array as a Pandas DataFrame.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input NumPack directory path |
| `array_name` | `str` or `None` | `None` | Array to export (inferred if single array) |
| `columns` | `List[str]` or `None` | `None` | Column names (auto-generated if `None`) |

#### Returns

- `pd.DataFrame`: The exported DataFrame

#### Raises

| Exception | Condition |
|-----------|-----------|
| `DependencyError` | If pandas is not installed |
| `ValueError` | If multiple arrays exist and `array_name` not specified |

#### Notes

- If `columns` is `None` and array is 2D, columns are named `col0`, `col1`, etc.

#### Example

```python
from numpack.io import to_pandas

# Load as DataFrame
df = to_pandas('data.npk', array_name='dataset')
print(df.head())

# With custom column names
df = to_pandas('data.npk', columns=['feature1', 'feature2', 'label'])
```

---

## Usage Examples

### DataFrame Processing Pipeline

```python
import pandas as pd
from numpack.io import from_pandas, to_pandas
from numpack import NumPack

# Start with a DataFrame
df = pd.read_csv('raw_data.csv')

# Convert to NumPack for efficient processing
from_pandas(df, 'data.npk')

# Use NumPack's efficient operations
with NumPack('data.npk') as npk:
    with npk.batch_mode():
        data = npk.load('data')
        # Fast batch processing
        data = normalize(data)
        npk.save({'data': data})

# Convert back to DataFrame for analysis
result_df = to_pandas('data.npk', columns=df.columns.tolist())
print(result_df.describe())
```

### Selective Column Export

```python
from numpack.io import to_pandas

# Export with meaningful column names
df = to_pandas(
    'model_features.npk',
    array_name='features',
    columns=['age', 'income', 'score', 'category']
)

# Use pandas functionality
print(df.groupby('category').mean())
```

### Integration with ML Workflows

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from numpack.io import from_pandas, to_pandas
from numpack import NumPack

# Load and split data
df = pd.read_csv('dataset.csv')
train_df, test_df = train_test_split(df, test_size=0.2)

# Store efficiently
from_pandas(train_df, 'train.npk', array_name='data')
from_pandas(test_df, 'test.npk', array_name='data')

# Later, load for training
train_data = to_pandas('train.npk')
```
