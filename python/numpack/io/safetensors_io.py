"""SafeTensors conversion utilities for NumPack.

SafeTensors is a safe, fast, and simple file format for storing tensors,
designed by Hugging Face. It provides memory-mapped access and is widely
used for storing ML model weights.

This module provides two types of conversions:

1. **Memory-to-file / File-to-memory conversions**:
   - `from_safetensors(tensors, npk_path)` - Save SafeTensors dict to .npk file
   - `to_safetensors(npk_path, array_names)` - Load from .npk file and return dict

2. **File-to-file conversions (streaming)**:
   - `from_safetensors_file(st_path, npk_path)` - Convert .safetensors to .npk
   - `to_safetensors_file(npk_path, st_path)` - Convert .npk to .safetensors
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .utils import (
    DEFAULT_CHUNK_SIZE,
    LARGE_FILE_THRESHOLD,
    DependencyError,
    _open_numpack_for_read,
    _open_numpack_for_write,
    estimate_chunk_rows,
)


def _check_safetensors():
    """Check if safetensors is installed and return the module."""
    try:
        import safetensors
        import safetensors.numpy
        return safetensors
    except ImportError:
        raise DependencyError(
            "safetensors is required for SafeTensors conversion.\n"
            "Install it with: pip install safetensors"
        )


# =============================================================================
# Memory-to-File / File-to-Memory Conversions
# =============================================================================

def from_safetensors(
    tensors: Dict[str, Any],
    output_path: Union[str, Path],
    drop_if_exists: bool = False,
) -> None:
    """Save SafeTensors tensors (from memory) to a NumPack file.
    
    Parameters
    ----------
    tensors : dict
        Dictionary of tensor name to tensor data. Can be:
        - Dict from `safetensors.numpy.load_file()` (already numpy arrays)
        - Dict from `safetensors.torch.load_file()` (torch tensors)
    output_path : str or Path
        Output NumPack directory path (.npk).
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    
    Raises
    ------
    DependencyError
        If safetensors is not installed.
    
    Notes
    -----
    SafeTensors numpy format already provides memory-mapped numpy arrays,
    so conversion is typically zero-copy.
    
    Examples
    --------
    >>> from safetensors.numpy import load_file
    >>> from numpack.io import from_safetensors
    >>> tensors = load_file('model.safetensors')
    >>> from_safetensors(tensors, 'output.npk')
    """
    _check_safetensors()
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    
    try:
        arrays = {}
        for name, tensor in tensors.items():
            if isinstance(tensor, np.ndarray):
                arrays[name] = tensor
            else:
                # Might be torch tensor
                try:
                    import torch
                    if torch.is_tensor(tensor):
                        arrays[name] = tensor.detach().cpu().numpy()
                    else:
                        raise TypeError(f"Unsupported tensor type: {type(tensor)}")
                except ImportError:
                    raise TypeError(f"Unsupported tensor type: {type(tensor)}")
        
        npk.save(arrays)
    finally:
        npk.close()


def to_safetensors(
    input_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Load arrays from a NumPack file and return as a dict (SafeTensors-compatible).
    
    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path (.npk).
    array_names : list of str, optional
        Names of arrays to load. If None, loads all arrays.
    
    Returns
    -------
    dict
        Dictionary mapping array names to NumPy arrays.
        This dict can be directly saved with `safetensors.numpy.save_file()`.
    
    Raises
    ------
    DependencyError
        If safetensors is not installed.
    
    Notes
    -----
    SafeTensors requires contiguous arrays. Non-contiguous arrays will
    be converted to contiguous format.
    
    Examples
    --------
    >>> from safetensors.numpy import save_file
    >>> from numpack.io import to_safetensors
    >>> arrays = to_safetensors('input.npk')
    >>> save_file(arrays, 'model.safetensors')
    """
    _check_safetensors()
    
    npk = _open_numpack_for_read(input_path)
    
    try:
        if array_names is None:
            array_names = npk.get_member_list()
        
        result = {}
        for name in array_names:
            arr = npk.load(name)
            # SafeTensors requires contiguous arrays
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)
            result[name] = arr
        
        return result
    finally:
        npk.close()


# =============================================================================
# File Conversions (Streaming)
# =============================================================================

def from_safetensors_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    keys: Optional[List[str]] = None,
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Convert a SafeTensors file to NumPack format with streaming.
    
    Parameters
    ----------
    input_path : str or Path
        Path to the input SafeTensors file (.safetensors).
    output_path : str or Path
        Output NumPack directory path (.npk).
    keys : list of str, optional
        Specific tensor names to convert. If None, converts all tensors.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    chunk_size : int, optional
        Chunk size in bytes for streaming large tensors.
    
    Raises
    ------
    DependencyError
        If safetensors is not installed.
    KeyError
        If any key in `keys` is not found in the file.
    
    Examples
    --------
    >>> from numpack.io import from_safetensors_file
    >>> from_safetensors_file('model.safetensors', 'output.npk')
    >>> from_safetensors_file('model.safetensors', 'output.npk', keys=['embeddings'])
    """
    _check_safetensors()
    from safetensors.numpy import load_file
    
    input_path = Path(input_path)
    
    # Load SafeTensors file (memory-mapped)
    tensors = load_file(str(input_path))
    
    # Filter keys if specified
    if keys is not None:
        missing = set(keys) - set(tensors.keys())
        if missing:
            raise KeyError(f"Keys not found: {missing}. Available: {list(tensors.keys())}")
        tensors = {k: tensors[k] for k in keys}
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    
    try:
        for name, arr in tensors.items():
            _save_array_streaming(npk, name, arr, chunk_size)
    finally:
        npk.close()


def to_safetensors_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Convert a NumPack file to SafeTensors format.
    
    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path (.npk).
    output_path : str or Path
        Output SafeTensors file path (.safetensors).
    array_names : list of str, optional
        Names of arrays to export. If None, exports all arrays.
    metadata : dict, optional
        Optional metadata to include in the SafeTensors file.
        Keys and values must be strings.
    
    Raises
    ------
    DependencyError
        If safetensors is not installed.
    
    Notes
    -----
    SafeTensors files are written atomically and are memory-mapped on read,
    making them very efficient for large model weights.
    
    Examples
    --------
    >>> from numpack.io import to_safetensors_file
    >>> to_safetensors_file('input.npk', 'output.safetensors')
    >>> to_safetensors_file('input.npk', 'output.safetensors', 
    ...                     metadata={'format': 'numpack', 'version': '1.0'})
    """
    _check_safetensors()
    from safetensors.numpy import save_file
    
    npk = _open_numpack_for_read(input_path)
    
    try:
        if array_names is None:
            array_names = npk.get_member_list()
        
        tensors = {}
        for name in array_names:
            arr = npk.load(name)
            # Ensure contiguous for SafeTensors
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)
            tensors[name] = arr
        
        # Save to SafeTensors
        save_file(tensors, str(output_path), metadata=metadata)
    finally:
        npk.close()


# =============================================================================
# Advanced Functions
# =============================================================================

def get_safetensors_metadata(
    path: Union[str, Path],
) -> Dict[str, Any]:
    """Get metadata from a SafeTensors file without loading tensors.
    
    Parameters
    ----------
    path : str or Path
        Path to the SafeTensors file.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'tensors': Dict mapping tensor names to {'dtype', 'shape'}
        - 'metadata': User-provided metadata (if any)
    
    Examples
    --------
    >>> from numpack.io import get_safetensors_metadata
    >>> info = get_safetensors_metadata('model.safetensors')
    >>> print(info['tensors'].keys())  # tensor names
    >>> print(info['metadata'])  # custom metadata
    """
    _check_safetensors()
    from safetensors import safe_open
    
    result = {'tensors': {}, 'metadata': {}}
    
    with safe_open(str(path), framework='numpy') as f:
        # Get tensor info
        for key in f.keys():
            tensor = f.get_tensor(key)
            result['tensors'][key] = {
                'dtype': str(tensor.dtype),
                'shape': tensor.shape,
            }
        
        # Get metadata
        metadata = f.metadata()
        if metadata:
            result['metadata'] = dict(metadata)
    
    return result


def iter_safetensors(
    path: Union[str, Path],
    keys: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Iterate over tensors in a SafeTensors file one at a time.
    
    This is memory-efficient for large files as tensors are loaded
    individually via memory mapping.
    
    Parameters
    ----------
    path : str or Path
        Path to the SafeTensors file.
    keys : list of str, optional
        Specific tensor names to iterate. If None, iterates all.
    
    Yields
    ------
    tuple of (str, numpy.ndarray)
        Tensor name and data pairs.
    
    Examples
    --------
    >>> from numpack.io import iter_safetensors
    >>> for name, tensor in iter_safetensors('model.safetensors'):
    ...     print(f"{name}: {tensor.shape}")
    """
    _check_safetensors()
    from safetensors import safe_open
    
    with safe_open(str(path), framework='numpy') as f:
        tensor_keys = keys if keys is not None else list(f.keys())
        
        for key in tensor_keys:
            yield key, f.get_tensor(key)


# =============================================================================
# Internal Helpers
# =============================================================================

def _save_array_streaming(
    npk: Any,
    name: str,
    arr: np.ndarray,
    chunk_size: int,
) -> None:
    """Save an array to NumPack with streaming for large arrays."""
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
# Exports
# =============================================================================

__all__ = [
    # Memory conversions
    'from_safetensors',
    'to_safetensors',
    # File conversions
    'from_safetensors_file',
    'to_safetensors_file',
    # Utilities
    'get_safetensors_metadata',
    'iter_safetensors',
]
