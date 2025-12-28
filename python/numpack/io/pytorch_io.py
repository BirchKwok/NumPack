"""PyTorch tensor conversion utilities for NumPack.

This module provides two types of conversions:

1. **Memory-to-file / File-to-memory conversions**:
   - `from_torch(tensor, npk_path)` - Save PyTorch tensor to .npk file
   - `to_torch(npk_path, array_name)` - Load from .npk file and return PyTorch tensor

2. **File-to-file conversions (streaming)**:
   - `from_torch_file(pt_path, npk_path)` - Convert .pt/.pth file to .npk
   - `to_torch_file(npk_path, pt_path)` - Convert .npk file to .pt/.pth
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .utils import (
    DEFAULT_CHUNK_SIZE,
    LARGE_FILE_THRESHOLD,
    _check_torch,
    _open_numpack_for_read,
    _open_numpack_for_write,
    estimate_chunk_rows,
)


# =============================================================================
# Memory-to-File / File-to-Memory Conversions
# =============================================================================

def from_tensor(
    tensor: Any,
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    drop_if_exists: bool = False,
    return_npk_obj: bool = False,
) -> Any:
    """Save a PyTorch tensor (from memory) to a NumPack file.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Input PyTorch tensor to save.
    output_path : str or Path
        Output NumPack directory path (.npk).
    array_name : str, optional
        Name of the array in the NumPack file. Default is 'data'.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    return_npk_obj : bool, optional
        If True, return an opened NumPack instance for output_path.
    
    Returns
    -------
    NumPack or None
        The NumPack instance if `return_npk_obj` is True, otherwise None.
    
    Raises
    ------
    DependencyError
        If PyTorch is not installed.
    TypeError
        If the input is not a PyTorch tensor.
    
    Notes
    -----
    Zero-copy conversion is used when the tensor is on CPU.
    GPU tensors are automatically moved to CPU.
    
    Examples
    --------
    >>> import torch
    >>> from numpack.io import from_tensor
    >>> tensor = torch.randn(1000, 128)
    >>> from_tensor(tensor, 'output.npk', array_name='embeddings')
    """
    torch = _check_torch()
    
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    
    if array_name is None:
        array_name = 'data'
    
    # Convert to numpy (zero-copy for CPU tensors)
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    arr = tensor.detach().numpy()
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    try:
        npk.save({array_name: arr})
    finally:
        npk.close()

    if return_npk_obj:
        return _open_numpack_for_read(output_path)
    return None


def to_tensor(
    input_path: Union[str, Path],
    array_name: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[Any] = None,
) -> Any:
    """Load an array from a NumPack file and return as a PyTorch tensor.
    
    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path (.npk).
    array_name : str, optional
        Name of the array to load. If None and only one array exists,
        that array is loaded.
    device : str, optional
        Target device ('cpu', 'cuda', 'cuda:0', etc.). Default is 'cpu'.
    dtype : torch.dtype, optional
        Target dtype. If None, infers from the array dtype.
    
    Returns
    -------
    torch.Tensor
        PyTorch tensor loaded from the NumPack file.
    
    Raises
    ------
    DependencyError
        If PyTorch is not installed.
    ValueError
        If array_name is None and multiple arrays exist.
    
    Examples
    --------
    >>> from numpack.io import to_tensor
    >>> tensor = to_tensor('input.npk', array_name='embeddings')
    >>> tensor = to_tensor('input.npk', device='cuda')
    """
    torch = _check_torch()
    
    npk = _open_numpack_for_read(input_path)
    
    try:
        if array_name is None:
            members = npk.get_member_list()
            if len(members) == 1:
                array_name = members[0]
            else:
                raise ValueError(
                    f"NumPack contains multiple arrays {members}; "
                    "please provide the array_name argument."
                )
        
        arr = npk.load(array_name)
        
        # Ensure contiguous for zero-copy
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        
        # Create tensor (zero-copy for CPU)
        tensor = torch.from_numpy(arr)
        
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        
        if device is not None and device != 'cpu':
            tensor = tensor.to(device=device)
        
        return tensor
    finally:
        npk.close()


# =============================================================================
# File-to-File Conversions (Streaming)
# =============================================================================

def from_pt(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    key: Optional[str] = None,
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    return_npk_obj: bool = False,
) -> Any:
    """Convert a PyTorch ``.pt``/``.pth`` file to NumPack format.
    
    Parameters
    ----------
    input_path : str or Path
        Path to the input PyTorch file (.pt or .pth).
    output_path : str or Path
        Output NumPack directory path (.npk).
    key : str, optional
        If the file contains a dict, convert only this key.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    chunk_size : int, optional
        Chunk size in bytes for streaming large tensors.
    return_npk_obj : bool, optional
        If True, return an opened NumPack instance for output_path.
    
    Returns
    -------
    NumPack or None
        The NumPack instance if `return_npk_obj` is True, otherwise None.
    
    Raises
    ------
    DependencyError
        If PyTorch is not installed.
    KeyError
        If `key` is provided but not found in the file.
    TypeError
        If the file contains unsupported data types.
    
    Examples
    --------
    >>> from numpack.io import from_pt
    >>> from_pt('model.pt', 'output.npk')
    >>> from_pt('data.pt', 'output.npk', key='features')
    """
    torch = _check_torch()
    input_path = Path(input_path)
    
    # Load PyTorch file (maps to CPU to avoid GPU memory issues)
    data = torch.load(str(input_path), map_location='cpu', weights_only=False)
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    
    try:
        if isinstance(data, dict):
            if key is not None:
                # Single key
                if key not in data:
                    raise KeyError(f"Key '{key}' not found. Available: {list(data.keys())}")
                tensor = data[key]
                if torch.is_tensor(tensor):
                    _save_tensor_streaming(npk, key, tensor, chunk_size)
                else:
                    raise TypeError(f"Value for key '{key}' is not a tensor")
            else:
                # All keys
                for name, tensor in data.items():
                    if torch.is_tensor(tensor):
                        _save_tensor_streaming(npk, name, tensor, chunk_size)
        elif torch.is_tensor(data):
            array_name = input_path.stem
            _save_tensor_streaming(npk, array_name, data, chunk_size)
        else:
            raise TypeError(f"Unsupported PyTorch data type: {type(data)}")
    finally:
        npk.close()

    if return_npk_obj:
        return _open_numpack_for_read(output_path)
    return None


def to_pt(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
    as_dict: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Convert a NumPack file to PyTorch ``.pt`` format.
    
    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path (.npk).
    output_path : str or Path
        Output PyTorch file path (.pt or .pth).
    array_names : list of str, optional
        Names of arrays to export. If None, exports all arrays.
    as_dict : bool, optional
        If True, save as dict. If False and single array, save tensor directly.
    chunk_size : int, optional
        Chunk size in bytes for streaming large arrays.
    
    Raises
    ------
    DependencyError
        If PyTorch is not installed.
    
    Examples
    --------
    >>> from numpack.io import to_pt
    >>> to_pt('input.npk', 'output.pt')
    """
    torch = _check_torch()
    
    npk = _open_numpack_for_read(input_path)
    
    try:
        if array_names is None:
            array_names = npk.get_member_list()
        
        tensors = {}
        for name in array_names:
            # Load array (uses memory mapping for large files)
            arr = npk.load(name)
            # Convert to tensor
            tensors[name] = torch.from_numpy(arr)
        
        if not as_dict and len(tensors) == 1:
            torch.save(list(tensors.values())[0], str(output_path))
        else:
            torch.save(tensors, str(output_path))
    finally:
        npk.close()


# =============================================================================
# Internal Helpers
# =============================================================================

def _save_tensor_streaming(
    npk: Any,
    name: str,
    tensor: Any,
    chunk_size: int,
) -> None:
    """Save a tensor to NumPack with streaming for large tensors."""
    torch = _check_torch()
    
    # Convert to numpy (this may copy for GPU tensors)
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    arr = tensor.detach().numpy()
    
    # Check if streaming is needed
    if arr.nbytes <= LARGE_FILE_THRESHOLD:
        # Small array: save directly
        npk.save({name: arr})
    else:
        # Large array: stream in chunks
        shape = arr.shape
        dtype = arr.dtype
        batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
        total_rows = shape[0]
        
        for start_idx in range(0, total_rows, batch_rows):
            end_idx = min(start_idx + batch_rows, total_rows)
            chunk = np.ascontiguousarray(arr[start_idx:end_idx])
            
            if start_idx == 0:
                npk.save({name: chunk})
            else:
                npk.append({name: chunk})


# =============================================================================
# Legacy Aliases (deprecated, will be removed in 0.6.0)
# =============================================================================

from .utils import deprecated_alias

# Memory conversion aliases
from_torch = deprecated_alias('from_tensor', from_tensor)
from_torch.__name__ = 'from_torch'

to_torch = deprecated_alias('to_tensor', to_tensor)
to_torch.__name__ = 'to_torch'

# File conversion aliases
from_torch_file = deprecated_alias('from_pt', from_pt)
from_torch_file.__name__ = 'from_torch_file'

to_torch_file = deprecated_alias('to_pt', to_pt)
to_torch_file.__name__ = 'to_torch_file'

from_pytorch = deprecated_alias('from_pt', from_pt)
from_pytorch.__name__ = 'from_pytorch'

to_pytorch = deprecated_alias('to_pt', to_pt)
to_pytorch.__name__ = 'to_pytorch'


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Primary names (recommended) - Memory conversions
    'from_tensor',
    'to_tensor',
    # Primary names (recommended) - File conversions
    'from_pt',
    'to_pt',
    # Legacy aliases (deprecated, will be removed in 0.6.0)
    'from_torch',
    'to_torch',
    'from_torch_file',
    'to_torch_file',
    'from_pytorch',
    'to_pytorch',
]
