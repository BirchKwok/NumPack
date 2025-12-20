# IO Module Overview

The `numpack.io` module provides format conversion utilities for seamless integration with popular data frameworks.

## Import

```python
from numpack.io import (
    # PyTorch
    from_torch, to_torch,
    from_torch_file, to_torch_file,
    
    # Arrow/Feather
    from_arrow, to_arrow,
    from_feather_file, to_feather_file,
    
    # Parquet
    from_parquet_table, to_parquet_table,
    from_parquet_file, to_parquet_file,
    
    # SafeTensors
    from_safetensors, to_safetensors,
    from_safetensors_file, to_safetensors_file,
    
    # NumPy
    from_numpy, to_numpy,
    
    # HDF5
    from_hdf5, to_hdf5,
    
    # Zarr
    from_zarr, to_zarr,
    
    # CSV/TXT
    from_csv, to_csv,
    from_txt, to_txt,
    
    # Pandas
    from_pandas, to_pandas,
    
    # S3
    from_s3, to_s3,
    
    # Zero-copy utilities
    to_dlpack, from_dlpack,
    numpy_to_torch_zero_copy, torch_to_numpy_zero_copy,
    numpy_to_arrow_zero_copy, arrow_to_numpy_zero_copy,
)
```

## API Design Pattern

All conversion functions follow a consistent naming pattern:

| Pattern | Description | Example |
|---------|-------------|---------|
| `from_xx(data, npk_path)` | Memory object → .npk file | `from_torch(tensor, 'out.npk')` |
| `to_xx(npk_path)` | .npk file → Memory object | `to_torch('in.npk')` |
| `from_xx_file(xx_path, npk_path)` | .xx file → .npk file | `from_torch_file('model.pt', 'out.npk')` |
| `to_xx_file(npk_path, xx_path)` | .npk file → .xx file | `to_torch_file('in.npk', 'model.pt')` |

## Streaming vs Memory

| Type | Use Case | Memory Usage |
|------|----------|--------------|
| Memory functions (`from_xx`, `to_xx`) | Small data | Full data in RAM |
| File functions (`from_xx_file`, `to_xx_file`) | Large files | Streaming (low RAM) |

## Dependencies

| Format | Required Package | Install |
|--------|-----------------|---------|
| PyTorch | `torch` | `pip install torch` |
| Arrow/Feather | `pyarrow` | `pip install pyarrow` |
| Parquet | `pyarrow` | `pip install pyarrow` |
| SafeTensors | `safetensors` | `pip install safetensors` |
| HDF5 | `h5py` | `pip install h5py` |
| Zarr | `zarr` | `pip install zarr` |
| Pandas | `pandas` | `pip install pandas` |
| S3 | `s3fs` | `pip install s3fs` |

## Documentation

- [PyTorch Conversion](./pytorch.md)
- [Arrow/Feather Conversion](./arrow_feather.md)
- [Parquet Conversion](./parquet.md)
- [SafeTensors Conversion](./safetensors.md)
- [NumPy Conversion](./numpy.md)
- [HDF5 Conversion](./hdf5.md)
- [Zarr Conversion](./zarr.md)
- [CSV/TXT Conversion](./csv_txt.md)
- [Pandas Conversion](./pandas.md)
- [S3 Cloud Storage](./s3.md)
- [Zero-Copy Utilities](./zero_copy.md)

## Constants

```python
from numpack.io import (
    LARGE_FILE_THRESHOLD,  # 1GB - threshold for streaming mode
    DEFAULT_CHUNK_SIZE,    # 100MB - default chunk size
    DEFAULT_BATCH_ROWS,    # 100000 - default batch rows
)
```

## Utility Functions

```python
from numpack.io import (
    get_file_size,        # Get file size in bytes
    is_large_file,        # Check if file exceeds threshold
    estimate_chunk_rows,  # Estimate rows per chunk
)
```
