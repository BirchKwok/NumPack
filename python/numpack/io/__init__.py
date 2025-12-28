"""NumPack I/O helpers for converting common data formats.

This module provides import/export utilities between NumPack and common
data-science storage formats (NumPy ``.npy``/``.npz``, Zarr, HDF5, Parquet,
Feather, Pandas, CSV/TXT, PyTorch, SafeTensors).

Function Naming Convention
--------------------------
- **File conversions** use file extension names: `from_npy`, `from_parquet`, `from_pt`
- **Memory conversions** use object type names: `from_dataframe`, `from_tensor`, `from_table`

Notes
-----
- Optional dependencies are imported lazily and validated only when needed.
- Large files (by default > 1 GB) are handled using streaming/batched I/O.
- Implementations use parallel I/O and memory mapping where applicable.

Examples
--------
Convert a NumPy file to NumPack:

>>> from numpack.io import from_npy
>>> from_npy('data.npy', 'output.npk')

Convert NumPack to HDF5:

>>> from numpack.io import to_hdf5
>>> to_hdf5('input.npk', 'output.h5')

Save a pandas DataFrame to NumPack:

>>> from numpack.io import from_dataframe
>>> from_dataframe(df, 'output.npk')
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import (
    Any, 
    Callable, 
    Dict, 
    Iterator, 
    List, 
    Optional, 
    Tuple, 
    Union,
    TYPE_CHECKING
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import h5py
    import zarr
    import pyarrow as pa
    import torch

# Large file threshold: 1GB
LARGE_FILE_THRESHOLD = 1 * 1024 * 1024 * 1024  # 1GB in bytes

# Default chunk size: 100MB (row count is computed based on dtype)
DEFAULT_CHUNK_SIZE = 100 * 1024 * 1024  # 100MB in bytes

# Default batch rows (used for streaming)
DEFAULT_BATCH_ROWS = 100000


# =============================================================================
# Dependency checking utilities
# =============================================================================

class DependencyError(ImportError):
    """Raised when an optional dependency is not installed."""
    pass


def _check_dependency(module_name: str, package_name: Optional[str] = None) -> Any:
    """Validate and import an optional dependency.
    
    Parameters
    ----------
    module_name : str
        Module name to import.
    package_name : str, optional
        Package name for pip installation (if different from module name).
    
    Returns
    -------
    module
        Imported module.
    
    Raises
    ------
    DependencyError
        If the dependency is not installed.
    """
    import importlib
    
    if package_name is None:
        package_name = module_name
    
    try:
        return importlib.import_module(module_name)
    except ImportError:
        raise DependencyError(
            f"The optional dependency '{package_name}' is required to use this feature.\n"
            f"Please run: pip install {package_name}"
        )


def _check_h5py():
    """Validate and import h5py."""
    return _check_dependency('h5py')


def _check_zarr():
    """Validate and import zarr."""
    return _check_dependency('zarr')


def _check_pyarrow():
    """Validate and import pyarrow."""
    return _check_dependency('pyarrow')


def _check_pandas():
    """Validate and import pandas."""
    return _check_dependency('pandas')


def _check_torch():
    """Validate and import torch (PyTorch)."""
    return _check_dependency('torch', 'torch')


def _check_s3fs():
    """Validate and import s3fs."""
    return _check_dependency('s3fs')


def _check_boto3():
    """Validate and import boto3."""
    return _check_dependency('boto3')


# =============================================================================
# File size and streaming utilities
# =============================================================================

def get_file_size(path: Union[str, Path]) -> int:
    """Get file size in bytes.
    
    Parameters
    ----------
    path : str or Path
        File path.
    
    Returns
    -------
    int
        File size in bytes. If `path` is a directory, returns the total size of
        all files under it.
    """
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    elif path.is_dir():
        total = 0
        for f in path.rglob('*'):
            if f.is_file():
                total += f.stat().st_size
        return total
    return 0


def is_large_file(path: Union[str, Path], threshold: int = LARGE_FILE_THRESHOLD) -> bool:
    """Check whether a file is considered large (requiring streaming I/O).
    
    Parameters
    ----------
    path : str or Path
        File path.
    threshold : int, optional
        Large-file threshold in bytes. Defaults to 1GB.
    
    Returns
    -------
    bool
        True if the file size exceeds `threshold`.
    """
    return get_file_size(path) > threshold


def estimate_chunk_rows(
    shape: Tuple[int, ...], 
    dtype: np.dtype, 
    target_chunk_bytes: int = DEFAULT_CHUNK_SIZE
) -> int:
    """Estimate how many rows a chunk should contain.
    
    Parameters
    ----------
    shape : tuple
        Array shape.
    dtype : numpy.dtype
        Array dtype.
    target_chunk_bytes : int, optional
        Target chunk size in bytes. Defaults to 100MB.
    
    Returns
    -------
    int
        Suggested number of rows per batch.
    """
    if len(shape) == 0:
        return 1
    
    # Compute bytes per row
    row_elements = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    bytes_per_row = row_elements * dtype.itemsize
    
    if bytes_per_row == 0:
        return DEFAULT_BATCH_ROWS
    
    # Compute target row count
    target_rows = max(1, target_chunk_bytes // bytes_per_row)
    
    # Clamp to a reasonable range
    return min(target_rows, shape[0], DEFAULT_BATCH_ROWS * 10)


# =============================================================================
# NumPack helper functions
# =============================================================================

def _get_numpack_class():
    """Get the NumPack class."""
    from numpack import NumPack
    return NumPack


def _open_numpack_for_write(
    output_path: Union[str, Path], 
    drop_if_exists: bool = False
) -> Any:
    """Open a NumPack file for writing.
    
    Parameters
    ----------
    output_path : str or Path
        Output path.
    drop_if_exists : bool, optional
        If True, delete the existing output directory first.
    
    Returns
    -------
    NumPack
        NumPack instance.
    """
    NumPack = _get_numpack_class()
    npk = NumPack(str(output_path), drop_if_exists=drop_if_exists)
    npk.open()
    return npk


def _open_numpack_for_read(input_path: Union[str, Path]) -> Any:
    """Open a NumPack file for reading.
    
    Parameters
    ----------
    input_path : str or Path
        Input path.
    
    Returns
    -------
    NumPack
        NumPack instance.
    """
    NumPack = _get_numpack_class()
    npk = NumPack(str(input_path))
    npk.open()
    return npk

# NumPy format conversion (npy/npz)
# =============================================================================

def from_numpy(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    return_npk_obj: bool = False,
) -> Any:
    """Import a NumPy ``.npy``/``.npz`` file into NumPack.
    
    For large files (by default > 1 GB), this function uses memory mapping and
    chunked streaming writes.
    
    Parameters
    ----------
    input_path : str or Path
        Path to the input ``.npy`` or ``.npz`` file.
    output_path : str or Path
        Output NumPack directory path.
    array_name : str, optional
        Name of the output array. If None, defaults to the file stem.
        For ``.npz`` input, this parameter is ignored and the keys inside the
        archive are used as array names.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    chunk_size : int, optional
        Chunk size in bytes used for streaming conversion.
    return_npk_obj : bool, optional
        If True, return the NumPack instance after import.
    
    Returns
    -------
    NumPack or None
        The NumPack instance if `return_npk_obj` is True, otherwise None.
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    suffix = input_path.suffix.lower()
    
    if suffix == '.npy':
        _from_npy(input_path, output_path, array_name, drop_if_exists, chunk_size)
    elif suffix == '.npz':
        _from_npz(input_path, output_path, drop_if_exists, chunk_size)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: .npy and .npz")

    if return_npk_obj:
        return _open_numpack_for_read(output_path)
    return None


def _from_npy(
    input_path: Path,
    output_path: Union[str, Path],
    array_name: Optional[str],
    drop_if_exists: bool,
    chunk_size: int
) -> None:
    """Import from a single ``.npy`` file."""
    if array_name is None:
        array_name = input_path.stem
    
    file_size = get_file_size(input_path)
    
    if file_size > LARGE_FILE_THRESHOLD:
        # Large file: memory-map and stream writes
        _from_npy_streaming(input_path, output_path, array_name, drop_if_exists, chunk_size)
    else:
        # Small file: load directly
        arr = np.load(str(input_path))
        npk = _open_numpack_for_write(output_path, drop_if_exists)
        try:
            npk.save({array_name: arr})
        finally:
            npk.close()


def _from_npy_streaming(
    input_path: Path,
    output_path: Union[str, Path],
    array_name: str,
    drop_if_exists: bool,
    chunk_size: int
) -> None:
    """Stream-import a large ``.npy`` file."""
    # Load with memory mapping
    arr_mmap = np.load(str(input_path), mmap_mode='r')
    shape = arr_mmap.shape
    dtype = arr_mmap.dtype
    
    # Compute chunk row count
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    try:
        # Write in chunks
        for start_idx in range(0, total_rows, batch_rows):
            end_idx = min(start_idx + batch_rows, total_rows)
            chunk = np.array(arr_mmap[start_idx:end_idx])  # copy into memory
            
            if start_idx == 0:
                npk.save({array_name: chunk})
            else:
                npk.append({array_name: chunk})
    finally:
        npk.close()
        del arr_mmap


def _from_npz(
    input_path: Path,
    output_path: Union[str, Path],
    drop_if_exists: bool,
    chunk_size: int
) -> None:
    """Import from a ``.npz`` file."""
    # Check file size
    file_size = get_file_size(input_path)
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    try:
        if file_size > LARGE_FILE_THRESHOLD:
            # Large file: load arrays one by one
            with np.load(str(input_path), mmap_mode='r') as npz:
                for name in npz.files:
                    arr = npz[name]
                    # For large arrays, stream writes
                    if arr.nbytes > LARGE_FILE_THRESHOLD:
                        _save_array_streaming(npk, name, arr, chunk_size)
                    else:
                        npk.save({name: np.array(arr)})
        else:
            # Small file: load directly
            with np.load(str(input_path)) as npz:
                arrays = {name: npz[name] for name in npz.files}
                npk.save(arrays)
    finally:
        npk.close()


def _save_array_streaming(
    npk: Any, 
    array_name: str, 
    arr: np.ndarray, 
    chunk_size: int
) -> None:
    """Stream-save a large array to NumPack."""
    shape = arr.shape
    dtype = arr.dtype
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]
    
    for start_idx in range(0, total_rows, batch_rows):
        end_idx = min(start_idx + batch_rows, total_rows)
        chunk = np.array(arr[start_idx:end_idx])
        
        if start_idx == 0:
            npk.save({array_name: chunk})
        else:
            npk.append({array_name: chunk})


# =============================================================================
# Zarr format conversion
# =============================================================================

def from_zarr(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
    group: str = '/',
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    return_npk_obj: bool = False,
) -> Any:
    """Import arrays from a Zarr store into NumPack.
    
    Zarr stores are chunked natively. Large arrays are imported in batches and
    streamed into NumPack.
    
    Parameters
    ----------
    input_path : str or Path
        Path to the input Zarr store.
    output_path : str or Path
        Output NumPack directory path.
    array_names : list of str, optional
        Names of arrays to import. If None, imports all arrays under `group`.
    group : str, optional
        Zarr group path.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    chunk_size : int, optional
        Chunk size in bytes used for streaming conversion.
    return_npk_obj : bool, optional
        If True, return the NumPack instance after import.
    
    Returns
    -------
    NumPack or None
        The NumPack instance if `return_npk_obj` is True, otherwise None.
    """
    zarr = _check_zarr()
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    
    try:
        with zarr.open(str(input_path), mode='r') as store:
            if group != '/':
                store = store[group]
            
            if array_names is None:
                # Collect all arrays
                array_names = [name for name in store.array_keys()]
            
            for name in array_names:
                arr = store[name]
                shape = arr.shape
                dtype = arr.dtype
                estimated_size = int(np.prod(shape)) * dtype.itemsize
                
                if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
                    # Large array: streamed reads
                    _from_zarr_array_streaming(npk, arr, name, chunk_size)
                else:
                    # Small array: load directly
                    npk.save({name: arr[...]})
    finally:
        npk.close()

    if return_npk_obj:
        return _open_numpack_for_read(output_path)
    return None


def _from_zarr_array_streaming(
    npk: Any,
    zarr_arr: Any,  # zarr.Array
    array_name: str,
    chunk_size: int
) -> None:
    """Stream-import a Zarr array."""
    shape = zarr_arr.shape
    dtype = zarr_arr.dtype
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]
    
    for start_idx in range(0, total_rows, batch_rows):
        end_idx = min(start_idx + batch_rows, total_rows)
        chunk = zarr_arr[start_idx:end_idx]
        
        if start_idx == 0:
            npk.save({array_name: chunk})
        else:
            npk.append({array_name: chunk})


# =============================================================================
# Parquet format conversion
# =============================================================================

def from_parquet(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    return_npk_obj: bool = False,
) -> Any:
    """Import a Parquet file into NumPack.
    
    Large Parquet files (by default > 1 GB) are imported by iterating record
    batches and streaming the result into NumPack.
    
    Parameters
    ----------
    input_path : str or Path
        Path to the input Parquet file.
    output_path : str or Path
        Output NumPack directory path.
    array_name : str, optional
        Name of the output array. If None, defaults to the file stem.
    columns : list of str, optional
        Columns to read. If None, reads all columns.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    chunk_size : int, optional
        Chunk size in bytes used for streaming conversion.
    return_npk_obj : bool, optional
        If True, return the NumPack instance after import.
    
    Returns
    -------
    NumPack or None
        The NumPack instance if `return_npk_obj` is True, otherwise None.
    """
    pa = _check_pyarrow()
    import pyarrow.parquet as pq
    
    input_path = Path(input_path)
    
    if array_name is None:
        array_name = input_path.stem
    
    # Read Parquet metadata
    parquet_file = pq.ParquetFile(str(input_path))
    metadata = parquet_file.metadata
    file_size = get_file_size(input_path)
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    
    try:
        if file_size > LARGE_FILE_THRESHOLD:
            # Large file: stream record batches
            _from_parquet_streaming(npk, parquet_file, array_name, columns)
        else:
            # Small file: load directly
            table = pq.read_table(str(input_path), columns=columns)
            arr = np.ascontiguousarray(table.to_pandas().values)
            npk.save({array_name: arr})
    finally:
        npk.close()

    if return_npk_obj:
        return _open_numpack_for_read(output_path)
    return None


def _from_parquet_streaming(
    npk: Any,
    parquet_file: Any,  # pyarrow.parquet.ParquetFile
    array_name: str,
    columns: Optional[List[str]]
) -> None:
    """Stream-import a Parquet file."""
    first_batch = True
    
    for batch in parquet_file.iter_batches(columns=columns):
        arr = np.ascontiguousarray(batch.to_pandas().values)
        
        if first_batch:
            npk.save({array_name: arr})
            first_batch = False
        else:
            npk.append({array_name: arr})


# =============================================================================
# Feather format conversion
# =============================================================================

def from_feather(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
    drop_if_exists: bool = False,
    return_npk_obj: bool = False,
) -> Any:
    """Import a Feather file into NumPack.
    
    Feather is a fast, lightweight columnar format.
    
    Parameters
    ----------
    input_path : str or Path
        Path to the input Feather file.
    output_path : str or Path
        Output NumPack directory path.
    array_name : str, optional
        Name of the output array. If None, defaults to the file stem.
    columns : list of str, optional
        Columns to read.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    return_npk_obj : bool, optional
        If True, return the NumPack instance after import.
    
    Returns
    -------
    NumPack or None
        The NumPack instance if `return_npk_obj` is True, otherwise None.
    """
    pa = _check_pyarrow()
    import pyarrow.feather as feather
    
    input_path = Path(input_path)
    
    if array_name is None:
        array_name = input_path.stem
    
    # Feather requires full materialization (no streaming reads here)
    table = feather.read_table(str(input_path), columns=columns)
    arr = table.to_pandas().values
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    try:
        npk.save({array_name: arr})
    finally:
        npk.close()

    if return_npk_obj:
        return _open_numpack_for_read(output_path)
    return None


# =============================================================================
# Pandas DataFrame conversion
# =============================================================================

def from_pandas(
    df: "pd.DataFrame",
    output_path: Union[str, Path],
    array_name: str = 'data',
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    return_npk_obj: bool = False,
) -> Any:
    """Import a pandas DataFrame into NumPack.
    
    Large DataFrames (by default > 1 GB) are streamed into NumPack in chunks.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    output_path : str or Path
        Output NumPack directory path.
    array_name : str, optional
        Name of the output array.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    chunk_size : int, optional
        Chunk size in bytes used for streaming write.
    return_npk_obj : bool, optional
        If True, return the NumPack instance after import.
    
    Returns
    -------
    NumPack or None
        The NumPack instance if `return_npk_obj` is True, otherwise None.
    """
    pd = _check_pandas()
    
    arr = np.ascontiguousarray(df.values)
    estimated_size = arr.nbytes
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    
    try:
        if estimated_size > LARGE_FILE_THRESHOLD:
            # Large DataFrame: chunked writes
            _save_array_streaming(npk, array_name, arr, chunk_size)
        else:
            npk.save({array_name: arr})
    finally:
        npk.close()

    if return_npk_obj:
        return _open_numpack_for_read(output_path)
    return None


# =============================================================================
# PyTorch tensor conversion
# =============================================================================

def from_pytorch(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    key: Optional[str] = None,
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    return_npk_obj: bool = False,
) -> Any:
    """Import tensors from a PyTorch ``.pt``/``.pth`` file into NumPack.
    
    Parameters
    ----------
    input_path : str or Path
        Path to the input PyTorch file.
    output_path : str or Path
        Output NumPack directory path.
    key : str, optional
        If the file contains a dict, load only this key.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    chunk_size : int, optional
        Chunk size in bytes used for streaming conversion.
    return_npk_obj : bool, optional
        If True, return the NumPack instance after import.
    
    Returns
    -------
    NumPack or None
        The NumPack instance if `return_npk_obj` is True, otherwise None.
    """
    torch = _check_torch()
    
    input_path = Path(input_path)
    
    # Load PyTorch file
    data = torch.load(str(input_path), map_location='cpu', weights_only=False)
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    
    try:
        if isinstance(data, dict):
            # Dict: save all tensors or a specified key
            if key is not None:
                if key not in data:
                    raise KeyError(f"Key '{key}' was not found in the file. Available keys: {list(data.keys())}")
                tensor = data[key]
                if torch.is_tensor(tensor):
                    arr = tensor.detach().cpu().numpy()
                    _save_array_with_streaming_check(npk, key, arr, chunk_size)
                else:
                    raise TypeError(f"Value for key '{key}' is not a tensor")
            else:
                for name, tensor in data.items():
                    if torch.is_tensor(tensor):
                        arr = tensor.detach().cpu().numpy()
                        _save_array_with_streaming_check(npk, name, arr, chunk_size)
        elif torch.is_tensor(data):
            # Single tensor
            array_name = input_path.stem
            arr = data.detach().cpu().numpy()
            _save_array_with_streaming_check(npk, array_name, arr, chunk_size)
        else:
            raise TypeError(f"Unsupported PyTorch data type: {type(data)}")
    finally:
        npk.close()

    if return_npk_obj:
        return _open_numpack_for_read(output_path)
    return None


def _save_array_with_streaming_check(
    npk: Any,
    array_name: str,
    arr: np.ndarray,
    chunk_size: int
) -> None:
    """Check array size and decide whether to use streaming writes."""
    if arr.nbytes > LARGE_FILE_THRESHOLD and arr.ndim > 0:
        _save_array_streaming(npk, array_name, arr, chunk_size)
    else:
        npk.save({array_name: arr})


# =============================================================================
# S3 remote storage support
# =============================================================================

def from_s3(
    s3_path: str,
    output_path: Union[str, Path],
    format: str = 'auto',
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    return_npk_obj: bool = False,
    **s3_kwargs
) -> Any:
    """Download a file from S3 and import it into NumPack.
    
    Supported formats: npy, npz, csv, parquet, feather, hdf5.
    
    Parameters
    ----------
    s3_path : str
        S3 URL in the form ``"s3://bucket/path/to/file"``.
    output_path : str or Path
        Output NumPack directory path.
    format : str, optional
        Input format. If ``"auto"``, it is inferred from the file suffix.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    chunk_size : int, optional
        Chunk size in bytes used for streaming conversion.
    return_npk_obj : bool, optional
        If True, return the NumPack instance after import.
    **s3_kwargs
        Keyword arguments forwarded to ``s3fs.S3FileSystem`` (for example,
        ``anon=True`` for public buckets).
    
    Returns
    -------
    NumPack or None
        The NumPack instance if `return_npk_obj` is True, otherwise None.
    """
    s3fs = _check_s3fs()
    import tempfile
    
    # Create S3 filesystem
    fs = s3fs.S3FileSystem(**s3_kwargs)
    
    # Infer format
    if format == 'auto':
        suffix = Path(s3_path).suffix.lower()
        format_map = {
            '.npy': 'numpy',
            '.npz': 'numpy',
            '.csv': 'csv',
            '.txt': 'txt',
            '.parquet': 'parquet',
            '.feather': 'feather',
            '.h5': 'hdf5',
            '.hdf5': 'hdf5',
        }
        format = format_map.get(suffix, 'numpy')
    
    # Download into a temporary file and convert
    with tempfile.NamedTemporaryFile(suffix=Path(s3_path).suffix, delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Download
        fs.get(s3_path, tmp_path)
        
        # Dispatch to the corresponding import function
        format_handlers = {
            'numpy': from_numpy,
            'csv': from_csv,
            'txt': from_txt,
            'parquet': from_parquet,
            'feather': from_feather,
            'hdf5': from_hdf5,
        }
        
        handler = format_handlers.get(format)
        if handler is None:
            raise ValueError(f"Unsupported format: {format}")
        
        return handler(
            tmp_path,
            output_path,
            drop_if_exists=drop_if_exists,
            chunk_size=chunk_size,
            return_npk_obj=return_npk_obj,
        )
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def to_s3(
    input_path: Union[str, Path],
    s3_path: str,
    format: str = 'auto',
    array_name: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    **s3_kwargs
) -> None:
    """Export from NumPack and upload to S3.
    
    Supported formats: npy, npz, csv, parquet, feather, hdf5.
    
    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path.
    s3_path : str
        S3 URL in the form ``"s3://bucket/path/to/file"``.
    format : str, optional
        Output format. If ``"auto"``, it is inferred from the file suffix.
    array_name : str, optional
        Name of the array to export.
    chunk_size : int, optional
        Chunk size in bytes used for streaming export.
    **s3_kwargs
        Keyword arguments forwarded to ``s3fs.S3FileSystem``.
    
    Examples
    --------
    >>> from numpack.io import to_s3
    >>> to_s3('input.npk', 's3://my-bucket/output.parquet')
    """
    s3fs = _check_s3fs()
    import tempfile
    
    # Create S3 filesystem
    fs = s3fs.S3FileSystem(**s3_kwargs)
    
    # Infer format
    if format == 'auto':
        suffix = Path(s3_path).suffix.lower()
        format_map = {
            '.npy': 'numpy',
            '.npz': 'numpy',
            '.csv': 'csv',
            '.txt': 'txt',
            '.parquet': 'parquet',
            '.feather': 'feather',
            '.h5': 'hdf5',
            '.hdf5': 'hdf5',
        }
        format = format_map.get(suffix, 'numpy')
    
    # Export into a temporary file
    with tempfile.NamedTemporaryFile(suffix=Path(s3_path).suffix, delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Dispatch to the corresponding export function
        format_handlers = {
            'numpy': lambda inp, out, **kw: to_numpy(inp, out, array_names=[array_name] if array_name else None, chunk_size=chunk_size),
            'csv': lambda inp, out, **kw: to_csv(inp, out, array_name=array_name, chunk_size=chunk_size),
            'txt': lambda inp, out, **kw: to_txt(inp, out, array_name=array_name, chunk_size=chunk_size),
            'parquet': lambda inp, out, **kw: to_parquet(inp, out, array_name=array_name, chunk_size=chunk_size),
            'feather': lambda inp, out, **kw: to_feather(inp, out, array_name=array_name, chunk_size=chunk_size),
            'hdf5': lambda inp, out, **kw: to_hdf5(inp, out, array_names=[array_name] if array_name else None, chunk_size=chunk_size),
        }
        
        handler = format_handlers.get(format)
        if handler is None:
            raise ValueError(f"Unsupported format: {format}")
        
        handler(input_path, tmp_path)
        
        # Upload to S3
        fs.put(tmp_path, s3_path)
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# =============================================================================
# Convenience functions
# =============================================================================

def _infer_format(path: Path) -> str:
    """Infer format from a file path."""
    suffix = path.suffix.lower()
    
    format_map = {
        '.npy': 'numpy',
        '.npz': 'numpy',
        '.csv': 'csv',
        '.txt': 'txt',
        '.tsv': 'csv',
        '.h5': 'hdf5',
        '.hdf5': 'hdf5',
        '.hdf': 'hdf5',
        '.zarr': 'zarr',
        '.parquet': 'parquet',
        '.pq': 'parquet',
        '.feather': 'feather',
        '.fea': 'feather',
        '.pt': 'pytorch',
        '.pth': 'pytorch',
        '.npk': 'numpack',
    }
    
    # Check whether it is a directory (could be NumPack or Zarr)
    if path.is_dir():
        if (path / 'metadata.npkm').exists():
            return 'numpack'
        if (path / '.zarray').exists() or (path / '.zgroup').exists():
            return 'zarr'
    
    return format_map.get(suffix, 'unknown')


# =============================================================================
# Export public API
# =============================================================================

__all__ = [
    # Exceptions
    'DependencyError',
    
    # Utility functions
    'get_file_size',
    'is_large_file',
    'estimate_chunk_rows',
    
    # NumPy file conversion (.npy/.npz <-> .npk)
    'from_npy',       # Primary name (recommended)
    'to_npy',         # Primary name (recommended)
    'from_numpy',     # Legacy alias
    'to_numpy',       # Legacy alias
    
    # CSV/TXT conversion
    'from_csv',
    'to_csv',
    'from_txt',
    'to_txt',
    
    # HDF5 conversion
    'from_hdf5',
    'to_hdf5',
    
    # Zarr conversion
    'from_zarr',
    'to_zarr',
    
    # Parquet file conversion (.parquet <-> .npk)
    'from_parquet',       # Primary name (recommended)
    'to_parquet',         # Primary name (recommended)
    'from_parquet_file',  # Verbose alias
    'to_parquet_file',    # Verbose alias
    # Parquet memory conversion (PyArrow Table <-> .npk)
    'from_parquet_table',
    'to_parquet_table',
    
    # Feather file conversion (.feather <-> .npk)
    'from_feather',       # Primary name (recommended)
    'to_feather',         # Primary name (recommended)
    'from_feather_file',  # Verbose alias
    'to_feather_file',    # Verbose alias
    # Arrow/Feather memory conversion (PyArrow Table <-> .npk)
    'from_table',         # Primary name (recommended)
    'to_table',           # Primary name (recommended)
    'from_arrow',         # Legacy alias
    'to_arrow',           # Legacy alias
    
    # Pandas conversion (DataFrame <-> .npk)
    'from_dataframe',     # Primary name (recommended)
    'to_dataframe',       # Primary name (recommended)
    'from_pandas',        # Legacy alias
    'to_pandas',          # Legacy alias
    
    # PyTorch file conversion (.pt/.pth <-> .npk)
    'from_pt',            # Primary name (recommended)
    'to_pt',              # Primary name (recommended)
    'from_torch_file',    # Verbose alias
    'to_torch_file',      # Verbose alias
    'from_pytorch',       # Legacy alias
    'to_pytorch',         # Legacy alias
    # PyTorch memory conversion (Tensor <-> .npk)
    'from_tensor',        # Primary name (recommended)
    'to_tensor',          # Primary name (recommended)
    'from_torch',         # Legacy alias
    'to_torch',           # Legacy alias
    
    # SafeTensors file conversion (.safetensors <-> .npk)
    'from_safetensors_file',
    'to_safetensors_file',
    # SafeTensors memory conversion (dict <-> .npk)
    'from_tensor_dict',       # Primary name (recommended)
    'to_tensor_dict',         # Primary name (recommended)
    'from_safetensors',       # Legacy alias (for backward compatibility)
    'to_safetensors',         # Legacy alias (for backward compatibility)
    'get_safetensors_metadata',
    'iter_safetensors',
    
    # S3 support
    'from_s3',
    'to_s3',
    
    # Package operations
    'pack',
    'unpack',
    'get_package_info',
    
    # Constants
    'LARGE_FILE_THRESHOLD',
    'DEFAULT_CHUNK_SIZE',
    'DEFAULT_BATCH_ROWS',
    
    # Zero-copy utilities
    'DLPackBuffer',
    'to_dlpack',
    'from_dlpack',
    'numpy_to_arrow_zero_copy',
    'arrow_to_numpy_zero_copy',
    'table_to_numpy_zero_copy',
    'numpy_to_torch_zero_copy',
    'torch_to_numpy_zero_copy',
    'ZeroCopyArray',
    'wrap_for_zero_copy',
]

from .utils import (
    DEFAULT_BATCH_ROWS,
    DEFAULT_CHUNK_SIZE,
    LARGE_FILE_THRESHOLD,
    DependencyError,
    estimate_chunk_rows,
    get_file_size,
    is_large_file,
)

from .csv_io import from_csv, from_txt, to_csv, to_txt
from .feather_io import (
    # Primary names
    from_feather, to_feather,
    from_table, to_table,
    # Verbose/Legacy aliases
    from_feather_file, to_feather_file,
    from_arrow, to_arrow,
)
from .hdf5_io import from_hdf5, to_hdf5
from .numpy_io import (
    # Primary names
    from_npy, to_npy,
    # Legacy aliases
    from_numpy, to_numpy,
)
from .pandas_io import (
    # Primary names
    from_dataframe, to_dataframe,
    # Legacy aliases
    from_pandas, to_pandas,
)
from .parquet_io import (
    # Primary names
    from_parquet, to_parquet,
    from_parquet_table, to_parquet_table,
    # Verbose aliases
    from_parquet_file, to_parquet_file,
)
from .pytorch_io import (
    # Primary names
    from_pt, to_pt,
    from_tensor, to_tensor,
    # Verbose/Legacy aliases
    from_torch_file, to_torch_file,
    from_torch, to_torch,
    from_pytorch, to_pytorch,
)
from .safetensors_io import (
    # File conversions
    from_safetensors_file, to_safetensors_file,
    # Memory conversions (primary names)
    from_tensor_dict, to_tensor_dict,
    # Memory conversions (legacy aliases)
    from_safetensors, to_safetensors,
    # Utilities
    get_safetensors_metadata, iter_safetensors,
)
from .s3_io import from_s3, to_s3
from .zarr_io import from_zarr, to_zarr
from .package_io import pack, unpack, get_package_info
from .zero_copy import (
    DLPackBuffer,
    to_dlpack,
    from_dlpack,
    numpy_to_arrow_zero_copy,
    arrow_to_numpy_zero_copy,
    table_to_numpy_zero_copy,
    numpy_to_torch_zero_copy,
    torch_to_numpy_zero_copy,
    ZeroCopyArray,
    wrap_for_zero_copy,
)
