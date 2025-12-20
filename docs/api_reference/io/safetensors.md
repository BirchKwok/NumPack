# SafeTensors Conversion API Reference

Functions for converting between SafeTensors format and NumPack.

## Dependencies

```bash
pip install safetensors
```

---

## Memory Functions

### `from_safetensors(tensors, output_path, drop_if_exists=False)`

Save a dictionary of tensors to NumPack.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensors` | `Dict[str, np.ndarray]` | *required* | Dictionary of tensors |
| `output_path` | `str` or `Path` | *required* | Output NumPack directory path |
| `drop_if_exists` | `bool` | `False` | Overwrite if exists |

#### Returns

- `None`

#### Example

```python
import numpy as np
from numpack.io import from_safetensors

tensors = {
    'weight': np.random.rand(768, 768).astype(np.float32),
    'bias': np.random.rand(768).astype(np.float32)
}
from_safetensors(tensors, 'model.npk')
```

---

### `to_safetensors(input_path, array_names=None)`

Load NumPack arrays as a dictionary of tensors.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input NumPack directory path |
| `array_names` | `List[str]` or `None` | `None` | Arrays to load (all if `None`) |

#### Returns

- `Dict[str, np.ndarray]`: Dictionary of arrays

#### Example

```python
from numpack.io import to_safetensors

tensors = to_safetensors('model.npk')
print(tensors.keys())
```

---

## File Functions (Streaming)

### `from_safetensors_file(input_path, output_path, keys=None, drop_if_exists=False, chunk_size=DEFAULT_CHUNK_SIZE)`

Convert a SafeTensors file to NumPack.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input `.safetensors` file path |
| `output_path` | `str` or `Path` | *required* | Output NumPack directory path |
| `keys` | `List[str]` or `None` | `None` | Keys to import (all if `None`) |
| `drop_if_exists` | `bool` | `False` | Overwrite if exists |
| `chunk_size` | `int` | `DEFAULT_CHUNK_SIZE` | Chunk size for streaming |

#### Returns

- `None`

#### Example

```python
from numpack.io import from_safetensors_file

from_safetensors_file('model.safetensors', 'model.npk')
from_safetensors_file('model.safetensors', 'model.npk', keys=['layer1.weight'])
```

---

### `to_safetensors_file(input_path, output_path, array_names=None, metadata=None, chunk_size=DEFAULT_CHUNK_SIZE)`

Export NumPack arrays to a SafeTensors file.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input NumPack directory path |
| `output_path` | `str` or `Path` | *required* | Output `.safetensors` file path |
| `array_names` | `List[str]` or `None` | `None` | Arrays to export (all if `None`) |
| `metadata` | `Dict[str, str]` or `None` | `None` | Optional metadata to include |
| `chunk_size` | `int` | `DEFAULT_CHUNK_SIZE` | Chunk size for streaming |

#### Returns

- `None`

#### Example

```python
from numpack.io import to_safetensors_file

to_safetensors_file('model.npk', 'model.safetensors')
to_safetensors_file(
    'model.npk', 
    'model.safetensors',
    metadata={'framework': 'pytorch', 'version': '1.0'}
)
```

---

## Utility Functions

### `get_safetensors_metadata(path)`

Get metadata from a SafeTensors file without loading tensors.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` or `Path` | *required* | SafeTensors file path |

#### Returns

- `Dict[str, Any]`: Metadata dictionary including tensor shapes and dtypes

#### Example

```python
from numpack.io import get_safetensors_metadata

meta = get_safetensors_metadata('model.safetensors')
print(meta['tensors'])  # {'weight': {'shape': [768, 768], 'dtype': 'F32'}, ...}
```

---

### `iter_safetensors(path, keys=None)`

Iterate over tensors in a SafeTensors file.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` or `Path` | *required* | SafeTensors file path |
| `keys` | `List[str]` or `None` | `None` | Keys to iterate (all if `None`) |

#### Returns

- `Iterator[Tuple[str, np.ndarray]]`: Iterator yielding (name, tensor) pairs

#### Example

```python
from numpack.io import iter_safetensors

for name, tensor in iter_safetensors('model.safetensors'):
    print(f"{name}: {tensor.shape}")
```

---

## Usage Examples

### Model Conversion Pipeline

```python
from numpack.io import (
    from_safetensors_file,
    to_safetensors_file,
    get_safetensors_metadata
)
from numpack import NumPack

# Check model structure
meta = get_safetensors_metadata('original.safetensors')
print(f"Tensors: {list(meta['tensors'].keys())}")

# Convert to NumPack
from_safetensors_file('original.safetensors', 'model.npk')

# Process with NumPack
with NumPack('model.npk') as npk:
    weight = npk.load('encoder.weight')
    weight = quantize(weight)
    npk.save({'encoder.weight': weight})

# Export back to SafeTensors
to_safetensors_file(
    'model.npk',
    'quantized.safetensors',
    metadata={'quantization': 'int8'}
)
```
