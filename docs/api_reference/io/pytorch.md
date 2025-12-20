# PyTorch Conversion API Reference

Functions for converting between PyTorch tensors and NumPack files.

## Dependencies

```bash
pip install torch
```

---

## Memory Functions

### `from_torch(tensor, output_path, array_name=None, drop_if_exists=False)`

Save a PyTorch tensor to a NumPack file.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor` | `torch.Tensor` | *required* | PyTorch tensor to save |
| `output_path` | `str` or `Path` | *required* | Output NumPack directory path |
| `array_name` | `str` or `None` | `None` | Array name in NumPack (default: `'data'`) |
| `drop_if_exists` | `bool` | `False` | Overwrite if exists |

#### Returns

- `None`

#### Raises

| Exception | Condition |
|-----------|-----------|
| `DependencyError` | If PyTorch is not installed |

#### Example

```python
import torch
from numpack.io import from_torch

tensor = torch.randn(1000, 128)
from_torch(tensor, 'output.npk', array_name='embeddings')
```

---

### `to_torch(input_path, array_name=None, device=None)`

Load a NumPack array as a PyTorch tensor.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input NumPack directory path |
| `array_name` | `str` or `None` | `None` | Array name to load (inferred if single array) |
| `device` | `str` or `torch.device` or `None` | `None` | Target device (default: CPU) |

#### Returns

- `torch.Tensor`: The loaded tensor

#### Raises

| Exception | Condition |
|-----------|-----------|
| `DependencyError` | If PyTorch is not installed |
| `ValueError` | If multiple arrays exist and `array_name` not specified |

#### Example

```python
from numpack.io import to_torch

tensor = to_torch('input.npk', array_name='embeddings')
tensor = to_torch('input.npk', device='cuda:0')
```

---

## File Functions (Streaming)

### `from_torch_file(input_path, output_path, key=None, drop_if_exists=False, chunk_size=DEFAULT_CHUNK_SIZE)`

Convert a `.pt` file to NumPack (streaming for large files).

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input `.pt` file path |
| `output_path` | `str` or `Path` | *required* | Output NumPack directory path |
| `key` | `str` or `None` | `None` | Key to load from state dict (if applicable) |
| `drop_if_exists` | `bool` | `False` | Overwrite if exists |
| `chunk_size` | `int` | `DEFAULT_CHUNK_SIZE` | Chunk size in bytes for streaming |

#### Returns

- `None`

#### Example

```python
from numpack.io import from_torch_file

# Convert entire file
from_torch_file('model.pt', 'output.npk')

# Convert specific key from state dict
from_torch_file('model.pt', 'output.npk', key='encoder.weight')
```

---

### `to_torch_file(input_path, output_path, array_names=None, chunk_size=DEFAULT_CHUNK_SIZE)`

Export NumPack arrays to a `.pt` file.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` or `Path` | *required* | Input NumPack directory path |
| `output_path` | `str` or `Path` | *required* | Output `.pt` file path |
| `array_names` | `List[str]` or `None` | `None` | Arrays to export (all if `None`) |
| `chunk_size` | `int` | `DEFAULT_CHUNK_SIZE` | Chunk size for streaming |

#### Returns

- `None`

#### Example

```python
from numpack.io import to_torch_file

# Export all arrays
to_torch_file('input.npk', 'output.pt')

# Export specific arrays
to_torch_file('input.npk', 'output.pt', array_names=['weights', 'biases'])
```

---

## Legacy Aliases

```python
from_pytorch = from_torch_file  # Alias for compatibility
to_pytorch = to_torch_file      # Alias for compatibility
```

---

## Usage Examples

### Save Model Weights

```python
import torch
from numpack.io import from_torch

model = MyModel()
for name, param in model.named_parameters():
    from_torch(param.data, 'model.npk', array_name=name)
```

### Load to GPU

```python
from numpack.io import to_torch

weights = to_torch('model.npk', array_name='encoder.weight', device='cuda:0')
```

### Batch Conversion

```python
from numpack.io import from_torch_file, to_torch_file

# Convert PyTorch checkpoint to NumPack
from_torch_file('checkpoint.pt', 'model.npk')

# Later, export back to PyTorch
to_torch_file('model.npk', 'restored.pt')
```
