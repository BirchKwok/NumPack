# IO Format Conversion Guide

NumPack provides comprehensive format conversion utilities for seamless integration with popular data frameworks.

## Table of Contents

- [Overview](#overview)
- [PyTorch Conversion](#pytorch-conversion)
- [PyArrow/Feather Conversion](#pyarrowfeather-conversion)
- [Parquet Conversion](#parquet-conversion)
- [SafeTensors Conversion](#safetensors-conversion)
- [Other Formats](#other-formats)
- [Text File Conversion](#text-file-conversion)
- [Pandas Conversion](#pandas-conversion)
- [S3 Cloud Storage](#s3-cloud-storage)
- [Zero-Copy Utilities](#zero-copy-utilities)
- [Supported Formats Summary](#supported-formats-summary)
- [Best Practices](#best-practices)

---

## Overview

NumPack supports two types of conversions:

| Type | Functions | Description |
|------|-----------|-------------|
| **Memory ↔ .npk** | `from_xx()` / `to_xx()` | Convert in-memory data to/from .npk files |
| **File ↔ .npk** | `from_xx_file()` / `to_xx_file()` | Convert external files to/from .npk files (streaming) |

### API Design

```
from_xx(data, npk_path)     # Memory → .npk file
to_xx(npk_path)             # .npk file → Memory

from_xx_file(xx_path, npk_path)  # .xx file → .npk file
to_xx_file(npk_path, xx_path)    # .npk file → .xx file
```

---

## PyTorch Conversion

### Memory ↔ .npk

```python
from numpack.io import from_torch, to_torch
import torch

# Save PyTorch tensor to .npk
tensor = torch.randn(1000, 128)
from_torch(tensor, 'output.npk', array_name='embeddings')

# Load from .npk as PyTorch tensor
tensor = to_torch('input.npk', array_name='embeddings')
tensor = to_torch('input.npk', array_name='embeddings', device='cuda')  # GPU
```

### File ↔ .npk (Streaming)

```python
from numpack.io import from_torch_file, to_torch_file

# Convert .pt file to .npk (streaming for large files)
from_torch_file('model.pt', 'output.npk')
from_torch_file('model.pt', 'output.npk', key='weights')  # specific key

# Convert .npk to .pt file
to_torch_file('input.npk', 'output.pt')
to_torch_file('input.npk', 'output.pt', array_names=['layer1', 'layer2'])
```

### API Reference

#### `from_torch(tensor, output_path, array_name=None, drop_if_exists=False)`

Save a PyTorch tensor to a NumPack file.

**Parameters:**
- `tensor`: PyTorch tensor to save
- `output_path`: Output .npk path
- `array_name`: Name in NumPack file (default: 'data')
- `drop_if_exists`: Overwrite if exists

#### `to_torch(input_path, array_name=None, device=None, dtype=None)`

Load from NumPack and return as PyTorch tensor.

**Parameters:**
- `input_path`: Input .npk path
- `array_name`: Array to load (auto-detected if single array)
- `device`: Target device ('cpu', 'cuda', etc.)
- `dtype`: Target dtype

**Returns:** `torch.Tensor`

---

## PyArrow/Feather Conversion

### Memory ↔ .npk

```python
from numpack.io import from_arrow, to_arrow
import pyarrow as pa

# Save PyArrow Table to .npk
table = pa.table({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
from_arrow(table, 'output.npk', array_name='my_table')

# Load from .npk as PyArrow Table
table = to_arrow('input.npk', array_name='my_table')
print(table.column_names)  # ['col0', 'col1', ...]
```

### File ↔ .npk

```python
from numpack.io import from_feather_file, to_feather_file

# Convert .feather to .npk
from_feather_file('data.feather', 'output.npk')

# Convert .npk to .feather
to_feather_file('input.npk', 'output.feather', compression='zstd')
```

### API Reference

#### `from_arrow(table, output_path, array_name=None, columns=None, drop_if_exists=False)`

Save a PyArrow Table to a NumPack file.

**Parameters:**
- `table`: PyArrow Table to save
- `output_path`: Output .npk path
- `array_name`: Name in NumPack file (default: 'data')
- `columns`: Specific columns to save
- `drop_if_exists`: Overwrite if exists

#### `to_arrow(input_path, array_name=None, column_names=None)`

Load from NumPack and return as PyArrow Table.

**Parameters:**
- `input_path`: Input .npk path
- `array_name`: Array to load
- `column_names`: Names for table columns

**Returns:** `pyarrow.Table`

---

## Parquet Conversion

### Memory ↔ .npk

```python
from numpack.io import from_parquet_table, to_parquet_table
import pyarrow.parquet as pq

# Load Parquet into memory, save to .npk
table = pq.read_table('data.parquet')
from_parquet_table(table, 'output.npk', array_name='data')

# Load from .npk as PyArrow Table (for Parquet writing)
table = to_parquet_table('input.npk', array_name='data')
pq.write_table(table, 'output.parquet')
```

### File ↔ .npk (Streaming)

```python
from numpack.io import from_parquet_file, to_parquet_file

# Convert .parquet to .npk (streaming for large files)
from_parquet_file('data.parquet', 'output.npk')
from_parquet_file('data.parquet', 'output.npk', columns=['col1', 'col2'])

# Convert .npk to .parquet
to_parquet_file('input.npk', 'output.parquet', compression='snappy')
```

### API Reference

#### `from_parquet_file(input_path, output_path, array_name=None, columns=None, drop_if_exists=False, chunk_size=100MB)`

Convert a Parquet file to NumPack format with streaming.

**Parameters:**
- `input_path`: Input .parquet path
- `output_path`: Output .npk path
- `array_name`: Name in NumPack file
- `columns`: Specific columns to convert
- `chunk_size`: Chunk size for streaming (large files)

#### `to_parquet_file(input_path, output_path, array_name=None, compression='snappy', row_group_size=100000)`

Convert a NumPack file to Parquet format.

**Parameters:**
- `input_path`: Input .npk path
- `output_path`: Output .parquet path
- `compression`: 'snappy', 'gzip', 'brotli', 'zstd', 'lz4', 'none'
- `row_group_size`: Parquet row group size

---

## SafeTensors Conversion

SafeTensors is a safe, fast tensor format by Hugging Face, commonly used for ML model weights.

### Memory ↔ .npk

```python
from numpack.io import from_safetensors, to_safetensors
from safetensors.numpy import load_file, save_file

# Load SafeTensors into memory, save to .npk
tensors = load_file('model.safetensors')
from_safetensors(tensors, 'output.npk')

# Load from .npk as dict (SafeTensors-compatible)
arrays = to_safetensors('input.npk')
save_file(arrays, 'output.safetensors')
```

### File ↔ .npk

```python
from numpack.io import from_safetensors_file, to_safetensors_file

# Convert .safetensors to .npk
from_safetensors_file('model.safetensors', 'output.npk')
from_safetensors_file('model.safetensors', 'output.npk', keys=['embeddings'])

# Convert .npk to .safetensors
to_safetensors_file('input.npk', 'output.safetensors')
to_safetensors_file('input.npk', 'output.safetensors', 
                    metadata={'format': 'numpack'})
```

### Utility Functions

```python
from numpack.io import get_safetensors_metadata, iter_safetensors

# Get metadata without loading tensors
info = get_safetensors_metadata('model.safetensors')
print(info['tensors'].keys())  # tensor names
print(info['metadata'])        # custom metadata

# Iterate tensors one at a time (memory-efficient)
for name, tensor in iter_safetensors('model.safetensors'):
    print(f"{name}: {tensor.shape}")
```

---

## Other Formats

### NumPy (.npy)

```python
from numpack.io import from_numpy, to_numpy

# .npy → .npk
from_numpy('data.npy', 'output.npk')

# .npk → .npy
to_numpy('input.npk', 'output.npy', array_name='data')
```

### HDF5 (.h5)

```python
from numpack.io import from_hdf5, to_hdf5

# .h5 → .npk
from_hdf5('data.h5', 'output.npk', dataset='features')

# .npk → .h5
to_hdf5('input.npk', 'output.h5')
```

### Zarr

```python
from numpack.io import from_zarr, to_zarr

# Zarr → .npk
from_zarr('data.zarr', 'output.npk')

# .npk → Zarr
to_zarr('input.npk', 'output.zarr')
```

### CSV

```python
from numpack.io import from_csv, to_csv

# .csv → .npk
from_csv('data.csv', 'output.npk')

# .npk → .csv
to_csv('input.npk', 'output.csv', array_name='data')
```

---

## Complete Example: Model Conversion Pipeline

```python
import torch
import numpy as np
from numpack import NumPack
from numpack.io import (
    from_torch, to_torch,
    from_safetensors_file, to_safetensors_file,
)

# Scenario: Convert model weights between formats

# 1. Load PyTorch model weights
model = torch.nn.Linear(128, 64)
state_dict = model.state_dict()

# 2. Save each tensor to NumPack
with NumPack('model.npk', drop_if_exists=True) as npk:
    for name, tensor in state_dict.items():
        from_torch(tensor, 'model.npk', array_name=name)

# 3. Load back as PyTorch tensors
loaded_state = {}
with NumPack('model.npk') as npk:
    for name in npk.get_member_list():
        loaded_state[name] = to_torch('model.npk', array_name=name)

# 4. Export to SafeTensors for sharing
to_safetensors_file('model.npk', 'model.safetensors',
                    metadata={'framework': 'pytorch', 'version': '1.0'})

# 5. Convert SafeTensors back to NumPack
from_safetensors_file('model.safetensors', 'restored.npk')

print("Model conversion pipeline complete!")
```

---

## Text File Conversion

### TXT Files

```python
from numpack.io import from_txt, to_txt

# .txt → .npk (whitespace-delimited)
from_txt('data.txt', 'output.npk', array_name='data', delimiter=None)

# .npk → .txt
to_txt('input.npk', 'output.txt', array_name='data', delimiter='\t')
```

**Parameters:**
- `delimiter`: Field separator (default: whitespace)
- `skip_header`: Number of header rows to skip
- `dtype`: Target data type

---

## Pandas Conversion

### DataFrame ↔ .npk

```python
from numpack.io import from_pandas, to_pandas
import pandas as pd

# DataFrame → .npk
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
from_pandas(df, 'output.npk', array_name='dataframe')

# .npk → DataFrame
df = to_pandas('input.npk', array_name='dataframe')
print(df.columns)
```

**Notes:**
- Numeric columns are converted to NumPy arrays
- String columns may require special handling

---

## S3 Cloud Storage

NumPack supports direct reading and writing to Amazon S3.

### S3 ↔ .npk

```python
from numpack.io import from_s3, to_s3

# Download from S3 and convert to .npk (uses default AWS credentials)
from_s3('s3://my-bucket/data.npy', 'output.npk')

# Public bucket access
from_s3('s3://public-bucket/data.csv', 'output.npk', anon=True)

# Upload .npk to S3
to_s3('input.npk', 's3://my-bucket/output.parquet')

# Specify output format
to_s3('input.npk', 's3://my-bucket/output.csv', format='csv')
```

**Parameters:**
- `s3_path`: S3 URI in the form `s3://bucket/path/to/file`
- `format`: Input/output format (`'auto'`, `'numpy'`, `'csv'`, `'txt'`, `'parquet'`, `'feather'`, `'hdf5'`)
- `**s3_kwargs`: Keyword arguments forwarded to `s3fs.S3FileSystem` (e.g., `anon=True` for public buckets)

**Dependencies:** `s3fs`

---

## Zero-Copy Utilities

NumPack provides zero-copy utilities for efficient data exchange with other libraries.

### DLPack Protocol

```python
from numpack.io import to_dlpack, from_dlpack

# NumPy → DLPack capsule
arr = np.random.rand(100, 50)
capsule = to_dlpack(arr)

# DLPack capsule → NumPy
arr_restored = from_dlpack(capsule)
```

### Arrow Zero-Copy

```python
from numpack.io import numpy_to_arrow_zero_copy, arrow_to_numpy_zero_copy

# NumPy → Arrow (zero-copy)
arr = np.random.rand(100, 50).astype(np.float32)
arrow_arr = numpy_to_arrow_zero_copy(arr)

# Arrow → NumPy (zero-copy)
numpy_arr = arrow_to_numpy_zero_copy(arrow_arr)
```

### PyTorch Zero-Copy

```python
from numpack.io import numpy_to_torch_zero_copy, torch_to_numpy_zero_copy

# NumPy → PyTorch (shared memory)
arr = np.random.rand(100, 50).astype(np.float32)
tensor = numpy_to_torch_zero_copy(arr)

# PyTorch → NumPy (shared memory)
numpy_arr = torch_to_numpy_zero_copy(tensor)
```

### ZeroCopyArray Wrapper

```python
from numpack.io import ZeroCopyArray, wrap_for_zero_copy

# Wrap array for zero-copy operations
arr = np.random.rand(100, 50)
zc_arr = wrap_for_zero_copy(arr)

# Access as different formats
torch_tensor = zc_arr.to_torch()
arrow_array = zc_arr.to_arrow()
```

---

## Supported Formats Summary

| Format | Import | Export | Dependencies |
|--------|--------|--------|--------------|
| PyTorch (.pt) | ✅ | ✅ | `torch` |
| SafeTensors | ✅ | ✅ | `safetensors` |
| Feather | ✅ | ✅ | `pyarrow` |
| Parquet | ✅ | ✅ | `pyarrow` |
| NumPy (.npy) | ✅ | ✅ | - |
| HDF5 (.h5) | ✅ | ✅ | `h5py` |
| Zarr | ✅ | ✅ | `zarr` |
| CSV | ✅ | ✅ | - |
| TXT | ✅ | ✅ | - |
| Pandas | ✅ | ✅ | `pandas` |
| S3 | ✅ | ✅ | `boto3`, `s3fs` |

---

## Best Practices

### 1. Use Memory Functions for Small Data

```python
# Good: Small tensor, use memory function
tensor = torch.randn(1000, 128)
from_torch(tensor, 'output.npk')
```

### 2. Use File Functions for Large Files

```python
# Good: Large file, use streaming file function
from_torch_file('large_model.pt', 'output.npk')  # Streams in chunks
```

### 3. Specify Array Names

```python
# Good: Explicit array names
from_torch(tensor, 'output.npk', array_name='embeddings')
loaded = to_torch('output.npk', array_name='embeddings')
```

### 4. Handle Multiple Arrays

```python
# Save multiple tensors
with NumPack('model.npk', drop_if_exists=True) as npk:
    from_torch(weights, 'model.npk', array_name='weights')
    from_torch(biases, 'model.npk', array_name='biases')

# Load specific array
weights = to_torch('model.npk', array_name='weights')
```
