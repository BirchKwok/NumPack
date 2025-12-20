# Zero-Copy Utilities API Reference

Functions for efficient zero-copy data exchange between frameworks.

## Overview

Zero-copy operations share memory between different array representations without copying data, providing maximum performance for inter-framework data exchange.

## Dependencies

| Function | Required Package |
|----------|-----------------|
| `to_dlpack`, `from_dlpack` | None (NumPy 1.22+) |
| `numpy_to_torch_zero_copy`, `torch_to_numpy_zero_copy` | `torch` |
| `numpy_to_arrow_zero_copy`, `arrow_to_numpy_zero_copy` | `pyarrow` |

---

## DLPack Protocol

### `to_dlpack(array)`

Convert a NumPy array to a DLPack capsule.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array` | `np.ndarray` | *required* | Input NumPy array |

#### Returns

- DLPack capsule object

#### Example

```python
from numpack.io import to_dlpack
import numpy as np

arr = np.random.rand(100, 50).astype(np.float32)
capsule = to_dlpack(arr)

# Use with frameworks supporting DLPack
import torch
tensor = torch.from_dlpack(capsule)
```

---

### `from_dlpack(capsule)`

Convert a DLPack capsule to a NumPy array.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `capsule` | DLPack capsule | *required* | DLPack capsule object |

#### Returns

- `np.ndarray`: NumPy array (shares memory with source)

#### Example

```python
from numpack.io import from_dlpack
import torch

tensor = torch.randn(100, 50)
capsule = torch.to_dlpack(tensor)
arr = from_dlpack(capsule)
```

---

## PyTorch Zero-Copy

### `numpy_to_torch_zero_copy(array)`

Convert NumPy array to PyTorch tensor without copying.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array` | `np.ndarray` | *required* | Input NumPy array |

#### Returns

- `torch.Tensor`: Tensor sharing memory with input

#### Notes

- Array must be contiguous
- Modifying one affects the other

#### Example

```python
from numpack.io import numpy_to_torch_zero_copy
import numpy as np

arr = np.random.rand(1000, 128).astype(np.float32)
tensor = numpy_to_torch_zero_copy(arr)

# Shares memory
arr[0, 0] = 999
print(tensor[0, 0])  # 999.0
```

---

### `torch_to_numpy_zero_copy(tensor)`

Convert PyTorch tensor to NumPy array without copying.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor` | `torch.Tensor` | *required* | Input PyTorch tensor |

#### Returns

- `np.ndarray`: Array sharing memory with input

#### Notes

- Tensor must be on CPU
- Tensor must be contiguous

#### Example

```python
from numpack.io import torch_to_numpy_zero_copy
import torch

tensor = torch.randn(1000, 128)
arr = torch_to_numpy_zero_copy(tensor)
```

---

## Arrow Zero-Copy

### `numpy_to_arrow_zero_copy(array)`

Convert NumPy array to PyArrow Array without copying.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array` | `np.ndarray` | *required* | Input 1D NumPy array |

#### Returns

- `pa.Array`: Arrow array sharing memory

#### Notes

- Best for 1D arrays
- Array must be contiguous

#### Example

```python
from numpack.io import numpy_to_arrow_zero_copy
import numpy as np

arr = np.random.rand(10000).astype(np.float32)
arrow_arr = numpy_to_arrow_zero_copy(arr)
```

---

### `arrow_to_numpy_zero_copy(arrow_array)`

Convert PyArrow Array to NumPy array without copying.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arrow_array` | `pa.Array` | *required* | Input PyArrow array |

#### Returns

- `np.ndarray`: NumPy array sharing memory

#### Example

```python
from numpack.io import arrow_to_numpy_zero_copy
import pyarrow as pa

arrow_arr = pa.array([1.0, 2.0, 3.0, 4.0])
numpy_arr = arrow_to_numpy_zero_copy(arrow_arr)
```

---

### `table_to_numpy_zero_copy(table, column)`

Extract a column from PyArrow Table as NumPy array without copying.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `table` | `pa.Table` | *required* | Input PyArrow table |
| `column` | `str` | *required* | Column name to extract |

#### Returns

- `np.ndarray`: NumPy array sharing memory

#### Example

```python
from numpack.io import table_to_numpy_zero_copy
import pyarrow as pa

table = pa.table({'values': [1.0, 2.0, 3.0]})
arr = table_to_numpy_zero_copy(table, 'values')
```

---

## ZeroCopyArray Class

### `ZeroCopyArray`

Wrapper class providing unified zero-copy access to multiple formats.

#### Constructor

```python
class ZeroCopyArray:
    def __init__(self, array: np.ndarray)
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `to_numpy()` | `np.ndarray` | Get as NumPy array |
| `to_torch()` | `torch.Tensor` | Convert to PyTorch tensor |
| `to_arrow()` | `pa.Array` | Convert to Arrow array |
| `to_dlpack()` | DLPack capsule | Get DLPack capsule |

#### Example

```python
from numpack.io import ZeroCopyArray
import numpy as np

arr = np.random.rand(1000, 128).astype(np.float32)
zc = ZeroCopyArray(arr)

# Access in different formats (all share memory)
numpy_view = zc.to_numpy()
torch_view = zc.to_torch()
arrow_view = zc.to_arrow()
```

---

### `wrap_for_zero_copy(array)`

Convenience function to wrap an array for zero-copy access.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array` | `np.ndarray` | *required* | Input NumPy array |

#### Returns

- `ZeroCopyArray`: Wrapper object

#### Example

```python
from numpack.io import wrap_for_zero_copy

arr = np.random.rand(1000, 128).astype(np.float32)
zc = wrap_for_zero_copy(arr)

tensor = zc.to_torch()
```

---

## Usage Examples

### ML Pipeline with Zero-Copy

```python
from numpack import NumPack
from numpack.io import numpy_to_torch_zero_copy
import torch

with NumPack('features.npk') as npk:
    # Load as NumPy
    features = npk.load('features')
    
    # Zero-copy to PyTorch
    tensor = numpy_to_torch_zero_copy(features)
    
    # Use in model
    with torch.no_grad():
        output = model(tensor)
```

### Cross-Framework Data Sharing

```python
from numpack.io import ZeroCopyArray
import numpy as np

# Create data once
data = np.random.rand(10000, 256).astype(np.float32)
zc = ZeroCopyArray(data)

# Use in different frameworks without copying
pytorch_model(zc.to_torch())
arrow_table = pa.table({'features': zc.to_arrow()})
```

---

## Performance Notes

| Operation | Memory Copy | Speed |
|-----------|-------------|-------|
| Regular conversion | Full copy | Slower |
| Zero-copy | No copy | Instant |

### Requirements for Zero-Copy

1. **Contiguous memory**: Array must be C-contiguous
2. **Compatible dtype**: Some dtypes may require conversion
3. **Same device**: For PyTorch, tensor must be on CPU

### When Zero-Copy Fails

If requirements aren't met, functions fall back to copying:

```python
# Non-contiguous array
arr = np.random.rand(100, 100)[:, ::2]  # Non-contiguous
tensor = numpy_to_torch_zero_copy(arr)  # May copy
```
