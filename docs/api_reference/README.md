# NumPack API Reference

Complete API documentation for all NumPack modules and functions.

## Documentation Structure

```
api_reference/
├── README.md                 # This file
├── core/
│   ├── numpack_class.md      # NumPack class
│   ├── lazy_array.md         # LazyArray class
│   └── batch_modes.md        # BatchModeContext & WritableBatchMode
├── io/
│   ├── README.md             # IO module overview
│   ├── pytorch.md            # PyTorch conversion
│   ├── arrow_feather.md      # PyArrow/Feather conversion
│   ├── parquet.md            # Parquet conversion
│   ├── safetensors.md        # SafeTensors conversion
│   ├── numpy.md              # NumPy conversion
│   ├── hdf5.md               # HDF5 conversion
│   ├── zarr.md               # Zarr conversion
│   ├── csv_txt.md            # CSV/TXT conversion
│   ├── pandas.md             # Pandas conversion
│   ├── s3.md                 # S3 cloud storage
│   └── zero_copy.md          # Zero-copy utilities
└── utils/
    ├── package_io.md         # pack/unpack functions
    └── utilities.md          # Utility functions
```

## Quick Navigation

### Core Classes

| Class | Description | Documentation |
|-------|-------------|---------------|
| `NumPack` | Main array storage class | [numpack_class.md](./core/numpack_class.md) |
| `LazyArray` | Memory-mapped lazy loading | [lazy_array.md](./core/lazy_array.md) |
| `BatchModeContext` | In-memory batch caching | [batch_modes.md](./core/batch_modes.md) |
| `WritableBatchMode` | Zero-copy writable batch | [batch_modes.md](./core/batch_modes.md) |

### IO Conversion Functions

| Format | Import | Export | Documentation |
|--------|--------|--------|---------------|
| PyTorch | `from_torch`, `from_torch_file` | `to_torch`, `to_torch_file` | [pytorch.md](./io/pytorch.md) |
| Arrow/Feather | `from_arrow`, `from_feather_file` | `to_arrow`, `to_feather_file` | [arrow_feather.md](./io/arrow_feather.md) |
| Parquet | `from_parquet_table`, `from_parquet_file` | `to_parquet_table`, `to_parquet_file` | [parquet.md](./io/parquet.md) |
| SafeTensors | `from_safetensors`, `from_safetensors_file` | `to_safetensors`, `to_safetensors_file` | [safetensors.md](./io/safetensors.md) |
| NumPy | `from_numpy` | `to_numpy` | [numpy.md](./io/numpy.md) |
| HDF5 | `from_hdf5` | `to_hdf5` | [hdf5.md](./io/hdf5.md) |
| Zarr | `from_zarr` | `to_zarr` | [zarr.md](./io/zarr.md) |
| CSV/TXT | `from_csv`, `from_txt` | `to_csv`, `to_txt` | [csv_txt.md](./io/csv_txt.md) |
| Pandas | `from_pandas` | `to_pandas` | [pandas.md](./io/pandas.md) |
| S3 | `from_s3` | `to_s3` | [s3.md](./io/s3.md) |

### Utility Functions

| Function | Description | Documentation |
|----------|-------------|---------------|
| `pack` | Package NumPack directory | [package_io.md](./utils/package_io.md) |
| `unpack` | Extract NumPack package | [package_io.md](./utils/package_io.md) |
| `get_package_info` | Get package metadata | [package_io.md](./utils/package_io.md) |
| `get_backend_info` | Get backend information | [utilities.md](./utils/utilities.md) |

## Import Patterns

```python
# Core class
from numpack import NumPack, LazyArray

# IO conversion functions
from numpack.io import from_torch, to_torch
from numpack.io import from_numpy, to_numpy

# Package operations
from numpack import pack, unpack, get_package_info

# Backend info
from numpack import get_backend_info
```

## Version

This documentation is for NumPack version **0.5.0**.
