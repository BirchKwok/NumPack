# IO Module Overview

The `numpack.io` module provides format conversion utilities for seamless integration with popular data frameworks.

## Import

```python
from numpack.io import (
    # PyTorch (memory conversion)
    from_tensor, to_tensor,
    # PyTorch (file conversion)
    from_pt, to_pt,
    
    # Arrow/Feather (memory conversion)
    from_table, to_table,
    # Feather (file conversion)
    from_feather, to_feather,
    
    # Parquet (memory conversion)
    from_parquet_table, to_parquet_table,
    # Parquet (file conversion)
    from_parquet, to_parquet,
    
    # SafeTensors (memory conversion)
    from_tensor_dict, to_tensor_dict,
    # SafeTensors (file conversion)
    from_safetensors_file, to_safetensors_file,
    
    # NumPy (file conversion)
    from_npy, to_npy,
    
    # HDF5 (file conversion)
    from_hdf5, to_hdf5,
    
    # Zarr (file conversion)
    from_zarr, to_zarr,
    
    # CSV/TXT (file conversion)
    from_csv, to_csv,
    from_txt, to_txt,
    
    # Pandas (memory conversion)
    from_dataframe, to_dataframe,
    
    # S3 (file conversion)
    from_s3, to_s3,
    
    # Zero-copy utilities
    to_dlpack, from_dlpack,
    numpy_to_torch_zero_copy, torch_to_numpy_zero_copy,
    numpy_to_arrow_zero_copy, arrow_to_numpy_zero_copy,
)
```

## API Design Pattern

All conversion functions follow a consistent naming pattern:

| Type | Pattern | Description | Example |
|------|---------|-------------|---------|
| File | `from_ext(file_path, npk_path)` | .ext file -> .npk file | `from_pt('model.pt', 'out.npk')` |
| File | `to_ext(npk_path, file_path)` | .npk file -> .ext file | `to_pt('in.npk', 'model.pt')` |
| Memory | `from_object(data, npk_path)` | Memory object -> .npk file | `from_tensor(tensor, 'out.npk')` |
| Memory | `to_object(npk_path)` | .npk file -> Memory object | `to_tensor('in.npk')` |

> **Note:** Some legacy function names (`from_torch`, `to_torch`, `from_arrow`, `to_arrow`, `from_pandas`, `to_pandas`, `from_numpy`, `to_numpy`, `from_safetensors`, `to_safetensors`) are deprecated and will be removed in version 0.6.0.

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
