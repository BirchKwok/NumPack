"""NumPack I/O helpers for converting common data formats.

 This module provides import/export utilities between NumPack and common
 data-science storage formats (NumPy ``.npy``/``.npz``, Zarr, HDF5, Parquet,
 Feather, Pandas, CSV/TXT, PyTorch).

 Notes
 -----
 - Optional dependencies are imported lazily and validated only when needed.
 - Large files (by default > 1 GB) are handled using streaming/batched I/O.
 - Implementations use parallel I/O and memory mapping where applicable.

 Examples
 --------
 Import from a NumPy file:

 >>> from numpack.io import from_numpy
 >>> from_numpy('data.npy', 'output.npk')

 Export to HDF5:

 >>> from numpack.io import to_hdf5
 >>> to_hdf5('input.npk', 'output.h5')

 Stream conversion for a large CSV:

 >>> from numpack.io import from_csv
 >>> from_csv('large_data.csv', 'output.npk')
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


# =============================================================================
# NumPy format conversion (npy/npz)
# =============================================================================

def from_numpy(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
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
        Array name for ``.npy`` input. If None, defaults to the file stem.
        For ``.npz`` input, this parameter is ignored and the keys inside the
        archive are used as array names.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    chunk_size : int, optional
        Chunk size in bytes used for streaming conversion.
    
    Examples
    --------
    >>> from numpack.io import from_numpy
    >>> from_numpy('data.npy', 'output.npk')
    >>> from_numpy('data.npz', 'output.npk')  # keep all array names
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


def to_numpy(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
    compressed: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
    """Export NumPack arrays to NumPy ``.npy``/``.npz``.
    
    For large arrays (by default > 1 GB), this function streams reads from NumPack
    and writes the output in chunks.
    
    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path.
    output_path : str or Path
        Output ``.npy`` or ``.npz`` file path.
    array_names : list of str, optional
        Names of arrays to export. If None, exports all arrays.
        If output is ``.npy``, exactly one array must be selected.
    compressed : bool, optional
        Whether to compress the ``.npz`` output.
    chunk_size : int, optional
        Chunk size in bytes used for streaming export.
    
    Examples
    --------
    >>> from numpack.io import to_numpy
    >>> to_numpy('input.npk', 'output.npz')
    >>> to_numpy('input.npk', 'single_array.npy', array_names=['my_array'])
    """
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    
    npk = _open_numpack_for_read(input_path)
    try:
        if array_names is None:
            array_names = npk.get_member_list()
        
        if suffix == '.npy':
            if len(array_names) != 1:
                raise ValueError(
                    f"The .npy format can only store one array, but {len(array_names)} arrays were provided. "
                    "Use .npz or provide exactly one array name."
                )
            _to_npy(npk, output_path, array_names[0], chunk_size)
        elif suffix == '.npz':
            _to_npz(npk, output_path, array_names, compressed, chunk_size)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported: .npy and .npz")
    finally:
        npk.close()


def _to_npy(
    npk: Any, 
    output_path: Path, 
    array_name: str, 
    chunk_size: int
) -> None:
    """Export a single array to a ``.npy`` file."""
    shape = npk.get_shape(array_name)
    
    # Estimate size
    arr_sample = npk.getitem(array_name, [0])
    dtype = arr_sample.dtype
    estimated_size = int(np.prod(shape)) * dtype.itemsize
    
    if estimated_size > LARGE_FILE_THRESHOLD:
        # Large array: stream reads and writes
        _to_npy_streaming(npk, output_path, array_name, shape, dtype, chunk_size)
    else:
        # Small array: load directly
        arr = npk.load(array_name)
        np.save(str(output_path), arr)


def _to_npy_streaming(
    npk: Any,
    output_path: Path,
    array_name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    chunk_size: int
) -> None:
    """Stream-export a large array to a ``.npy`` file."""
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]
    
    # Create output file (pre-allocate space)
    # Use NumPy's format module to create a proper .npy header
    from numpy.lib import format as npy_format
    
    with open(output_path, 'wb') as f:
        # Write .npy header
        npy_format.write_array_header_1_0(f, {'descr': dtype.str, 'fortran_order': False, 'shape': shape})
        header_size = f.tell()
    
    # Write data via memory mapping
    total_bytes = int(np.prod(shape)) * dtype.itemsize
    with open(output_path, 'r+b') as f:
        f.seek(0, 2)  # Seek to end
        f.truncate(header_size + total_bytes)  # Extend file
    
    # Memory-mapped write
    arr_out = np.memmap(output_path, dtype=dtype, mode='r+', offset=header_size, shape=shape)
    
    try:
        for start_idx in range(0, total_rows, batch_rows):
            end_idx = min(start_idx + batch_rows, total_rows)
            chunk = npk.getitem(array_name, slice(start_idx, end_idx))
            arr_out[start_idx:end_idx] = chunk
        arr_out.flush()
    finally:
        del arr_out


def _to_npz(
    npk: Any,
    output_path: Path,
    array_names: List[str],
    compressed: bool,
    chunk_size: int
) -> None:
    """Export multiple arrays to a ``.npz`` file."""
    # The NPZ format does not support true streaming writes; data must be collected.
    # For large datasets, consider using other formats (for example, HDF5 or Zarr).
    
    arrays = {}
    for name in array_names:
        shape = npk.get_shape(name)
        arr_sample = npk.getitem(name, [0])
        dtype = arr_sample.dtype
        estimated_size = int(np.prod(shape)) * dtype.itemsize
        
        if estimated_size > LARGE_FILE_THRESHOLD:
            warnings.warn(
                f"Array '{name}' is large (>{estimated_size / 1e9:.1f}GB). "
                "The NPZ format loads all data into memory. "
                "For large datasets, consider using to_hdf5 or to_zarr.",
                UserWarning
            )
        
        arrays[name] = npk.load(name)
    
    if compressed:
        np.savez_compressed(str(output_path), **arrays)
    else:
        np.savez(str(output_path), **arrays)


# =============================================================================
# CSV format conversion
# =============================================================================

def from_csv(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    drop_if_exists: bool = False,
    dtype: Optional[np.dtype] = None,
    delimiter: str = ',',
    skiprows: int = 0,
    max_rows: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    **kwargs
) -> None:
    """Import a CSV file into NumPack.
    
    For large files (by default > 1 GB), this function uses streaming I/O.
    
    Parameters
    ----------
    input_path : str or Path
        Path to the input CSV file.
    output_path : str or Path
        Output NumPack directory path.
    array_name : str, optional
        Name of the output array. If None, defaults to the file stem.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    dtype : numpy.dtype, optional
        Target dtype. If None, dtype is inferred.
    delimiter : str, optional
        Column delimiter.
    skiprows : int, optional
        Number of rows to skip.
    max_rows : int, optional
        Maximum number of rows to read. If None, reads all rows.
    chunk_size : int, optional
        Chunk size in bytes used for streaming conversion.
    **kwargs
        Additional keyword arguments forwarded to `numpy.loadtxt` or `pandas.read_csv`.
    
    Examples
    --------
    >>> from numpack.io import from_csv
    >>> from_csv('data.csv', 'output.npk')
    >>> from_csv('data.csv', 'output.npk', dtype=np.float32, delimiter=';')
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    if array_name is None:
        array_name = input_path.stem
    
    file_size = get_file_size(input_path)
    
    if file_size > LARGE_FILE_THRESHOLD:
        # Large file: use pandas chunked reads (when available)
        _from_csv_streaming(
            input_path, output_path, array_name, drop_if_exists,
            dtype, delimiter, skiprows, chunk_size, **kwargs
        )
    else:
        # Small file: load directly
        try:
            # Prefer pandas when available (faster and more flexible)
            pd = _check_pandas()
            if 'header' not in kwargs:
                kwargs['header'] = None
            df = pd.read_csv(
                input_path, 
                delimiter=delimiter, 
                skiprows=skiprows,
                nrows=max_rows,
                dtype=dtype,
                **kwargs
            )
            # Ensure the array is C-contiguous (required by NumPack)
            arr = np.ascontiguousarray(df.values)
        except DependencyError:
            # Fall back to numpy
            arr = np.loadtxt(
                str(input_path),
                delimiter=delimiter,
                skiprows=skiprows,
                max_rows=max_rows,
                dtype=dtype if dtype is not None else np.float64,
                **{k: v for k, v in kwargs.items() if k in ['comments', 'usecols', 'unpack', 'ndmin', 'encoding']}
            )
        
        npk = _open_numpack_for_write(output_path, drop_if_exists)
        try:
            npk.save({array_name: arr})
        finally:
            npk.close()


def _from_csv_streaming(
    input_path: Path,
    output_path: Union[str, Path],
    array_name: str,
    drop_if_exists: bool,
    dtype: Optional[np.dtype],
    delimiter: str,
    skiprows: int,
    chunk_size: int,
    **kwargs
) -> None:
    """Stream-import a large CSV file."""
    pd = _check_pandas()
    
    # Estimate chunk row count (assume ~100 bytes per row)
    chunk_rows = max(1000, chunk_size // 100)
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    first_chunk = True
    
    try:
        if 'header' not in kwargs:
            kwargs['header'] = None
        reader = pd.read_csv(
            input_path,
            delimiter=delimiter,
            skiprows=skiprows,
            dtype=dtype,
            chunksize=chunk_rows,
            **kwargs
        )
        
        for chunk_df in reader:
            # Ensure the array is C-contiguous (required by NumPack)
            arr = np.ascontiguousarray(chunk_df.values)
            if dtype is not None:
                arr = arr.astype(dtype)
            
            if first_chunk:
                npk.save({array_name: arr})
                first_chunk = False
            else:
                npk.append({array_name: arr})
    finally:
        npk.close()


def to_csv(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    delimiter: str = ',',
    header: bool = False,
    fmt: str = '%.18e',
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    **kwargs
) -> None:
    """Export a NumPack array to CSV.
    
    For large arrays (by default > 1 GB), this function streams reads from NumPack
    and writes the output in chunks.
    
    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path.
    output_path : str or Path
        Output CSV file path.
    array_name : str, optional
        Name of the array to export. If None and the NumPack file contains exactly
        one array, that array is used.
    delimiter : str, optional
        Column delimiter.
    header : bool, optional
        Whether to write a header line.
    fmt : str, optional
        Numeric format string.
    chunk_size : int, optional
        Chunk size in bytes used for streaming export.
    **kwargs
        Additional keyword arguments forwarded to `numpy.savetxt`.
    
    Examples
    --------
    >>> from numpack.io import to_csv
    >>> to_csv('input.npk', 'output.csv')
    >>> to_csv('input.npk', 'output.csv', delimiter=';', fmt='%.6f')
    """
    npk = _open_numpack_for_read(input_path)
    try:
        if array_name is None:
            members = npk.get_member_list()
            if len(members) == 1:
                array_name = members[0]
            else:
                raise ValueError(
                    f"NumPack contains multiple arrays {members}; please provide the array_name argument."
                )
        
        shape = npk.get_shape(array_name)
        arr_sample = npk.getitem(array_name, [0])
        dtype = arr_sample.dtype
        estimated_size = int(np.prod(shape)) * dtype.itemsize
        
        if estimated_size > LARGE_FILE_THRESHOLD:
            _to_csv_streaming(npk, output_path, array_name, shape, dtype, 
                            delimiter, header, fmt, chunk_size, **kwargs)
        else:
            arr = npk.load(array_name)
            np.savetxt(str(output_path), arr, delimiter=delimiter, fmt=fmt, 
                      header='' if not header else delimiter.join([f'col{i}' for i in range(arr.shape[1] if arr.ndim > 1 else 1)]),
                      **kwargs)
    finally:
        npk.close()


def _to_csv_streaming(
    npk: Any,
    output_path: Union[str, Path],
    array_name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    delimiter: str,
    header: bool,
    fmt: str,
    chunk_size: int,
    **kwargs
) -> None:
    """Stream-export a large array to a CSV file."""
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]
    
    with open(output_path, 'w') as f:
        # Write header
        if header and len(shape) > 1:
            header_line = delimiter.join([f'col{i}' for i in range(shape[1])])
            f.write(f"# {header_line}\n")
        
        for start_idx in range(0, total_rows, batch_rows):
            end_idx = min(start_idx + batch_rows, total_rows)
            chunk = npk.getitem(array_name, slice(start_idx, end_idx))
            
            # Write chunk
            if isinstance(chunk, np.ndarray) and chunk.ndim == 2:
                np.savetxt(f, chunk, delimiter=delimiter, fmt=fmt)
            else:
                for row in np.atleast_1d(chunk):
                    if np.isscalar(row) or getattr(row, "ndim", 0) == 0:
                        line = fmt % row
                    else:
                        line = delimiter.join([fmt % val for val in np.atleast_1d(row)])
                    f.write(line + '\n')


# =============================================================================
# TXT format conversion (similar to CSV, but defaults to whitespace delimiter)
# =============================================================================

def from_txt(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    drop_if_exists: bool = False,
    dtype: Optional[np.dtype] = None,
    delimiter: Optional[str] = None,
    skiprows: int = 0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    **kwargs
) -> None:
    """Import a whitespace-delimited text file into NumPack.
    
    This is equivalent to `from_csv` but uses whitespace as the default delimiter.
    
    Parameters
    ----------
    input_path : str or Path
        Path to the input text file.
    output_path : str or Path
        Output NumPack directory path.
    array_name : str, optional
        Name of the output array. If None, defaults to the file stem.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    dtype : numpy.dtype, optional
        Target dtype.
    delimiter : str, optional
        Delimiter forwarded to `from_csv`. If None, whitespace is used.
    skiprows : int, optional
        Number of rows to skip.
    chunk_size : int, optional
        Chunk size in bytes used for streaming conversion.
    **kwargs
        Additional keyword arguments forwarded to the underlying reader.
    
    Examples
    --------
    >>> from numpack.io import from_txt
    >>> from_txt('data.txt', 'output.npk')
    """
    # Use from_csv but default delimiter is whitespace
    from_csv(
        input_path, output_path, array_name, drop_if_exists,
        dtype, delimiter if delimiter else ' ', skiprows, None, chunk_size,
        **kwargs
    )


def to_txt(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    delimiter: str = ' ',
    fmt: str = '%.18e',
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    **kwargs
) -> None:
    """Export a NumPack array to a whitespace-delimited text file.
    
    This is equivalent to `to_csv` but uses a space character as the default delimiter.
    
    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path.
    output_path : str or Path
        Output text file path.
    array_name : str, optional
        Name of the array to export.
    delimiter : str, optional
        Column delimiter.
    fmt : str, optional
        Numeric format string.
    chunk_size : int, optional
        Chunk size in bytes used for streaming export.
    **kwargs
        Additional keyword arguments forwarded to `numpy.savetxt`.
    
    Examples
    --------
    >>> from numpack.io import to_txt
    >>> to_txt('input.npk', 'output.txt')
    """
    to_csv(input_path, output_path, array_name, delimiter, False, fmt, chunk_size, **kwargs)


# =============================================================================
# HDF5 format conversion
# =============================================================================

def from_hdf5(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    dataset_names: Optional[List[str]] = None,
    group: str = '/',
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
    """Import datasets from an HDF5 file into NumPack.
    
    For large datasets (by default > 1 GB), this function uses streamed reads.
    
    Parameters
    ----------
    input_path : str or Path
        Path to the input HDF5 file.
    output_path : str or Path
        Output NumPack directory path.
    dataset_names : list of str, optional
        Names of datasets to import. If None, imports all datasets under `group`.
    group : str, optional
        HDF5 group path.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    chunk_size : int, optional
        Chunk size in bytes used for streaming conversion.
    
    Examples
    --------
    >>> from numpack.io import from_hdf5
    >>> from_hdf5('data.h5', 'output.npk')
    >>> from_hdf5('data.h5', 'output.npk', dataset_names=['dataset1', 'dataset2'])
    >>> from_hdf5('data.h5', 'output.npk', group='/experiments/run1')
    """
    h5py = _check_h5py()
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    
    try:
        with h5py.File(str(input_path), 'r') as h5f:
            grp = h5f[group]
            
            if dataset_names is None:
                # Collect all datasets under the group
                dataset_names = [name for name in grp.keys() 
                               if isinstance(grp[name], h5py.Dataset)]
            
            for name in dataset_names:
                dataset = grp[name]
                if not isinstance(dataset, h5py.Dataset):
                    warnings.warn(f"Skipping non-dataset object: {name}")
                    continue
                
                shape = dataset.shape
                dtype = dataset.dtype
                estimated_size = int(np.prod(shape)) * dtype.itemsize
                
                if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
                    # Large dataset: streamed reads
                    _from_hdf5_dataset_streaming(npk, dataset, name, chunk_size)
                else:
                    # Small dataset: load directly
                    arr = dataset[...]
                    npk.save({name: arr})
    finally:
        npk.close()


def _from_hdf5_dataset_streaming(
    npk: Any,
    dataset: Any,  # h5py.Dataset
    array_name: str,
    chunk_size: int
) -> None:
    """Stream-import an HDF5 dataset."""
    shape = dataset.shape
    dtype = dataset.dtype
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]
    
    for start_idx in range(0, total_rows, batch_rows):
        end_idx = min(start_idx + batch_rows, total_rows)
        chunk = dataset[start_idx:end_idx]
        
        if start_idx == 0:
            npk.save({array_name: chunk})
        else:
            npk.append({array_name: chunk})


def to_hdf5(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
    group: str = '/',
    compression: Optional[str] = 'gzip',
    compression_opts: Optional[int] = 4,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
    """Export NumPack arrays to an HDF5 file.
    
    For large arrays (by default > 1 GB), this function uses streamed reads and
    chunked writes.
    
    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path.
    output_path : str or Path
        Output HDF5 file path.
    array_names : list of str, optional
        Names of arrays to export. If None, exports all arrays.
    group : str, optional
        HDF5 group path.
    compression : str, optional
        Compression codec (for example, ``"gzip"``). Use None to disable.
    compression_opts : int, optional
        Compression level/options.
    chunk_size : int, optional
        Chunk size in bytes used for streaming export.
    
    Examples
    --------
    >>> from numpack.io import to_hdf5
    >>> to_hdf5('input.npk', 'output.h5')
    >>> to_hdf5('input.npk', 'output.h5', compression='lzf')
    """
    h5py = _check_h5py()
    
    npk = _open_numpack_for_read(input_path)
    
    try:
        if array_names is None:
            array_names = npk.get_member_list()
        
        with h5py.File(str(output_path), 'w') as h5f:
            # Create the group (if needed)
            if group != '/':
                grp = h5f.require_group(group)
            else:
                grp = h5f
            
            for name in array_names:
                shape = npk.get_shape(name)
                arr_sample = npk.getitem(name, [0])
                dtype = arr_sample.dtype
                estimated_size = int(np.prod(shape)) * dtype.itemsize
                
                # Compute HDF5 chunk shape
                if len(shape) > 0:
                    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
                    chunks = (min(batch_rows, shape[0]),) + shape[1:]
                else:
                    chunks = None
                
                # Create dataset
                ds = grp.create_dataset(
                    name, 
                    shape=shape, 
                    dtype=dtype,
                    chunks=chunks if chunks else True,
                    compression=compression,
                    compression_opts=compression_opts if compression else None
                )
                
                if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
                    # Large array: streamed writes
                    _to_hdf5_dataset_streaming(npk, ds, name, shape, dtype, chunk_size)
                else:
                    # Small array: write directly
                    ds[...] = npk.load(name)
    finally:
        npk.close()


def _to_hdf5_dataset_streaming(
    npk: Any,
    dataset: Any,  # h5py.Dataset
    array_name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    chunk_size: int
) -> None:
    """Stream-export a large array to an HDF5 dataset."""
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]
    
    for start_idx in range(0, total_rows, batch_rows):
        end_idx = min(start_idx + batch_rows, total_rows)
        chunk = npk.getitem(array_name, slice(start_idx, end_idx))
        dataset[start_idx:end_idx] = chunk


# =============================================================================
# Zarr format conversion
# =============================================================================

def from_zarr(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
    group: str = '/',
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
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
    
    Examples
    --------
    >>> from numpack.io import from_zarr
    >>> from_zarr('data.zarr', 'output.npk')
    >>> from_zarr('data.zarr', 'output.npk', array_names=['arr1', 'arr2'])
    """
    zarr = _check_zarr()
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    
    try:
        store = zarr.open(str(input_path), mode='r')
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


def to_zarr(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
    group: str = '/',
    compressor: Optional[str] = 'default',
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
    """Export NumPack arrays to a Zarr store.
    
    Zarr stores are chunked natively and are well-suited for large datasets.
    
    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path.
    output_path : str or Path
        Output Zarr store path.
    array_names : list of str, optional
        Names of arrays to export. If None, exports all arrays.
    group : str, optional
        Zarr group path.
    compressor : str or None, optional
        Compressor configuration. The default value attempts to use Blosc.
        Use None to disable compression.
    chunk_size : int, optional
        Chunk size in bytes used for streaming export.
    
    Examples
    --------
    >>> from numpack.io import to_zarr
    >>> to_zarr('input.npk', 'output.zarr')
    """
    zarr = _check_zarr()
    
    npk = _open_numpack_for_read(input_path)
    
    try:
        if array_names is None:
            array_names = npk.get_member_list()
        
        # Create Zarr store
        store = zarr.open(str(output_path), mode='w')
        if group != '/':
            store = store.require_group(group)
        
        # Configure compressor
        if compressor == 'default':
            try:
                from zarr.codecs import BloscCodec, BloscCname, BloscShuffle
                compressor_obj = BloscCodec(cname=BloscCname.zstd, clevel=3, shuffle=BloscShuffle.bitshuffle)
            except ImportError:
                compressor_obj = None
        elif compressor is None:
            compressor_obj = None
        else:
            compressor_obj = compressor
        
        for name in array_names:
            shape = npk.get_shape(name)
            arr_sample = npk.getitem(name, [0])
            dtype = arr_sample.dtype
            estimated_size = int(np.prod(shape)) * dtype.itemsize
            
            # Compute chunk shape
            if len(shape) > 0:
                batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
                chunks = (min(batch_rows, shape[0]),) + shape[1:]
            else:
                chunks = shape
            
            # Create Zarr array
            if hasattr(store, 'create_array'):
                zarr_arr = store.create_array(
                    name,
                    shape=shape,
                    dtype=dtype,
                    chunks=chunks if chunks else None,
                    compressors=compressor_obj,
                    overwrite=True
                )
            else:
                zarr_arr = store.create_dataset(
                    name,
                    shape=shape,
                    dtype=dtype,
                    chunks=chunks if chunks else None,
                    compressor=compressor_obj
                )
            
            if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
                # Large array: streamed writes
                _to_zarr_array_streaming(npk, zarr_arr, name, shape, dtype, chunk_size)
            else:
                # Small array: write directly
                zarr_arr[...] = npk.load(name)
    finally:
        npk.close()


def _to_zarr_array_streaming(
    npk: Any,
    zarr_arr: Any,  # zarr.Array
    array_name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    chunk_size: int
) -> None:
    """Stream-export a large array to Zarr."""
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]
    
    for start_idx in range(0, total_rows, batch_rows):
        end_idx = min(start_idx + batch_rows, total_rows)
        chunk = npk.getitem(array_name, slice(start_idx, end_idx))
        zarr_arr[start_idx:end_idx] = chunk


# =============================================================================
# Parquet format conversion
# =============================================================================

def from_parquet(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
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
    
    Examples
    --------
    >>> from numpack.io import from_parquet
    >>> from_parquet('data.parquet', 'output.npk')
    >>> from_parquet('data.parquet', 'output.npk', columns=['col1', 'col2'])
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


def to_parquet(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    compression: str = 'snappy',
    row_group_size: int = 100000,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
    """Export a NumPack array to Parquet.
    
    For large arrays (by default > 1 GB), this function streams reads from NumPack
    and writes the output in batches.
    
    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path.
    output_path : str or Path
        Output Parquet file path.
    array_name : str, optional
        Name of the array to export.
    compression : str, optional
        Parquet compression codec.
    row_group_size : int, optional
        Parquet row group size used for non-streaming writes.
    chunk_size : int, optional
        Chunk size in bytes used for streaming export.
    
    Examples
    --------
    >>> from numpack.io import to_parquet
    >>> to_parquet('input.npk', 'output.parquet')
    """
    pa = _check_pyarrow()
    import pyarrow.parquet as pq
    
    npk = _open_numpack_for_read(input_path)
    
    try:
        if array_name is None:
            members = npk.get_member_list()
            if len(members) == 1:
                array_name = members[0]
            else:
                raise ValueError(
                    f"NumPack contains multiple arrays {members}; please provide the array_name argument."
                )
        
        shape = npk.get_shape(array_name)
        arr_sample = npk.getitem(array_name, [0])
        dtype = arr_sample.dtype
        estimated_size = int(np.prod(shape)) * dtype.itemsize
        
        if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
            # Large array: streamed writes
            _to_parquet_streaming(
                npk, output_path, array_name, shape, dtype, 
                compression, row_group_size, chunk_size
            )
        else:
            # Small array: write directly
            arr = npk.load(array_name)
            # Convert to a PyArrow Table
            if arr.ndim == 1:
                table = pa.table({'data': arr})
            else:
                # Convert a multi-dimensional array into columns
                columns = {f'col{i}': arr[:, i] for i in range(arr.shape[1])} if arr.ndim == 2 else {'data': arr.flatten()}
                table = pa.table(columns)
            
            pq.write_table(table, str(output_path), compression=compression, 
                          row_group_size=row_group_size)
    finally:
        npk.close()


def _to_parquet_streaming(
    npk: Any,
    output_path: Union[str, Path],
    array_name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    compression: str,
    row_group_size: int,
    chunk_size: int
) -> None:
    """Stream-export a large array to Parquet."""
    pa = _check_pyarrow()
    import pyarrow.parquet as pq
    
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]
    
    writer = None
    
    try:
        for start_idx in range(0, total_rows, batch_rows):
            end_idx = min(start_idx + batch_rows, total_rows)
            chunk = npk.getitem(array_name, slice(start_idx, end_idx))
            
            # Convert to a Table
            if chunk.ndim == 1:
                table = pa.table({'data': chunk})
            else:
                columns = {f'col{i}': chunk[:, i] for i in range(chunk.shape[1])} if chunk.ndim == 2 else {'data': chunk.flatten()}
                table = pa.table(columns)
            
            if writer is None:
                writer = pq.ParquetWriter(
                    str(output_path), 
                    table.schema,
                    compression=compression
                )
            
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


# =============================================================================
# Feather format conversion
# =============================================================================

def from_feather(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
    drop_if_exists: bool = False
) -> None:
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
    
    Examples
    --------
    >>> from numpack.io import from_feather
    >>> from_feather('data.feather', 'output.npk')
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


def to_feather(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    compression: str = 'zstd',
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
    """Export a NumPack array to Feather.
    
    Notes
    -----
    Feather does not support true streaming writes here; the array is loaded into
    memory before writing.
    
    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path.
    output_path : str or Path
        Output Feather file path.
    array_name : str, optional
        Name of the array to export.
    compression : str, optional
        Feather compression codec.
    chunk_size : int, optional
        Chunk size in bytes (used only for size estimation / warning logic).
    
    Examples
    --------
    >>> from numpack.io import to_feather
    >>> to_feather('input.npk', 'output.feather')
    """
    pa = _check_pyarrow()
    import pyarrow.feather as feather
    
    npk = _open_numpack_for_read(input_path)
    
    try:
        if array_name is None:
            members = npk.get_member_list()
            if len(members) == 1:
                array_name = members[0]
            else:
                raise ValueError(
                    f"NumPack contains multiple arrays {members}; please provide the array_name argument."
                )
        
        shape = npk.get_shape(array_name)
        estimated_size = int(np.prod(shape)) * npk.getitem(array_name, [0]).dtype.itemsize
        
        if estimated_size > LARGE_FILE_THRESHOLD:
            warnings.warn(
                f"Array '{array_name}' is large (>{estimated_size / 1e9:.1f}GB). "
                "The Feather format loads all data into memory. "
                "For large datasets, consider using to_parquet or to_zarr.",
                UserWarning
            )
        
        arr = npk.load(array_name)
        
        # Convert to a Table
        if arr.ndim == 1:
            table = pa.table({'data': arr})
        else:
            columns = {f'col{i}': arr[:, i] for i in range(arr.shape[1])} if arr.ndim == 2 else {'data': arr.flatten()}
            table = pa.table(columns)
        
        feather.write_feather(table, str(output_path), compression=compression)
    finally:
        npk.close()


# =============================================================================
# Pandas DataFrame conversion
# =============================================================================

def from_pandas(
    df: "pd.DataFrame",
    output_path: Union[str, Path],
    array_name: str = 'data',
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
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
    
    Examples
    --------
    >>> import pandas as pd
    >>> from numpack.io import from_pandas
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> from_pandas(df, 'output.npk')
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


def to_pandas(
    input_path: Union[str, Path],
    array_name: Optional[str] = None,
    columns: Optional[List[str]] = None
) -> "pd.DataFrame":
    """Export a NumPack array as a pandas DataFrame.
    
    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path.
    array_name : str, optional
        Name of the array to export.
    columns : list of str, optional
        Column names. If None and the array is 2D, column names are generated.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame view of the exported array.
    
    Examples
    --------
    >>> from numpack.io import to_pandas
    >>> df = to_pandas('input.npk')
    """
    pd = _check_pandas()
    
    npk = _open_numpack_for_read(input_path)
    
    try:
        if array_name is None:
            members = npk.get_member_list()
            if len(members) == 1:
                array_name = members[0]
            else:
                raise ValueError(
                    f"NumPack contains multiple arrays {members}; please provide the array_name argument."
                )
        
        arr = npk.load(array_name)
        
        if columns is None and arr.ndim == 2:
            columns = [f'col{i}' for i in range(arr.shape[1])]
        
        return pd.DataFrame(arr, columns=columns)
    finally:
        npk.close()


# =============================================================================
# PyTorch tensor conversion
# =============================================================================

def from_pytorch(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    key: Optional[str] = None,
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
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
    
    Examples
    --------
    >>> from numpack.io import from_pytorch
    >>> from_pytorch('model.pt', 'output.npk')
    >>> from_pytorch('data.pt', 'output.npk', key='features')
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


def to_pytorch(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
    as_dict: bool = True
) -> None:
    """Export NumPack arrays to a PyTorch ``.pt`` file.
    
    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path.
    output_path : str or Path
        Output PyTorch file path.
    array_names : list of str, optional
        Names of arrays to export. If None, exports all arrays.
    as_dict : bool, optional
        If True, save a dict mapping array names to tensors. If False and only one
        array is exported, save the tensor directly.
    
    Examples
    --------
    >>> from numpack.io import to_pytorch
    >>> to_pytorch('input.npk', 'output.pt')
    """
    torch = _check_torch()
    
    npk = _open_numpack_for_read(input_path)
    
    try:
        if array_names is None:
            array_names = npk.get_member_list()
        
        tensors = {}
        for name in array_names:
            arr = npk.load(name)
            tensors[name] = torch.from_numpy(arr)
        
        if not as_dict and len(tensors) == 1:
            # Save a single tensor
            torch.save(list(tensors.values())[0], str(output_path))
        else:
            # Save a dict
            torch.save(tensors, str(output_path))
    finally:
        npk.close()


# =============================================================================
# S3 remote storage support
# =============================================================================

def from_s3(
    s3_path: str,
    output_path: Union[str, Path],
    format: str = 'auto',
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    **s3_kwargs
) -> None:
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
    **s3_kwargs
        Keyword arguments forwarded to ``s3fs.S3FileSystem`` (for example,
        ``anon=True`` for public buckets).
    
    Examples
    --------
    >>> from numpack.io import from_s3
    >>> from_s3('s3://my-bucket/data.npy', 'output.npk')
    >>> from_s3('s3://public-bucket/data.csv', 'output.npk', anon=True)
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
        
        handler(tmp_path, output_path, drop_if_exists=drop_if_exists, chunk_size=chunk_size)
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
    
    # NumPy conversion
    'from_numpy',
    'to_numpy',
    
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
    
    # Parquet/Feather conversion
    'from_parquet',
    'to_parquet',
    'from_feather',
    'to_feather',
    
    # Pandas conversion
    'from_pandas',
    'to_pandas',
    
    # PyTorch conversion (memory - zero-copy)
    'from_torch',
    'to_torch',
    # PyTorch conversion (file - streaming)
    'from_torch_file',
    'to_torch_file',
    'from_pytorch',  # legacy alias
    'to_pytorch',    # legacy alias
    
    # SafeTensors conversion (memory - zero-copy)
    'from_safetensors',
    'to_safetensors',
    # SafeTensors conversion (file - streaming)
    'from_safetensors_file',
    'to_safetensors_file',
    'get_safetensors_metadata',
    'iter_safetensors',
    
    # Arrow/Feather conversion (memory - zero-copy)
    'from_arrow',
    'to_arrow',
    # Feather conversion (file - streaming)
    'from_feather_file',
    'to_feather_file',
    'from_feather',  # legacy alias
    'to_feather',    # legacy alias
    
    # Parquet conversion (memory - zero-copy)
    'from_parquet_table',
    'to_parquet_table',
    # Parquet conversion (file - streaming)
    'from_parquet_file',
    'to_parquet_file',
    'from_parquet',  # legacy alias
    'to_parquet',    # legacy alias
    
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
    from_arrow, to_arrow,
    from_feather_file, to_feather_file,
    from_feather, to_feather,
)
from .hdf5_io import from_hdf5, to_hdf5
from .numpy_io import from_numpy, to_numpy
from .pandas_io import from_pandas, to_pandas
from .parquet_io import (
    from_parquet_table, to_parquet_table,
    from_parquet_file, to_parquet_file,
    from_parquet, to_parquet,
)
from .pytorch_io import (
    from_torch, to_torch,
    from_torch_file, to_torch_file,
    from_pytorch, to_pytorch,
)
from .safetensors_io import (
    from_safetensors, to_safetensors,
    from_safetensors_file, to_safetensors_file,
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
