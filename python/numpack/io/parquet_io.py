"""Apache Parquet conversion utilities for NumPack.

Parquet is a columnar storage format optimized for analytics workloads,
with excellent compression and support for streaming I/O.

This module provides two types of conversions:

1. **Memory-to-file / File-to-memory conversions**:
   - `from_parquet_table(table, npk_path)` - Save PyArrow Table to .npk file
   - `to_parquet_table(npk_path, array_name)` - Load from .npk file and return PyArrow Table

2. **File-to-file conversions (streaming)**:
   - `from_parquet_file(parquet_path, npk_path)` - Convert .parquet to .npk
   - `to_parquet_file(npk_path, parquet_path)` - Convert .npk to .parquet
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .utils import (
    DEFAULT_CHUNK_SIZE,
    LARGE_FILE_THRESHOLD,
    _check_pyarrow,
    estimate_chunk_rows,
    get_file_size,
    _open_numpack_for_read,
    _open_numpack_for_write,
)


# =============================================================================
# Memory-to-File / File-to-Memory Conversions
# =============================================================================

def from_parquet_table(
    table: Any,
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
    drop_if_exists: bool = False,
) -> None:
    """Save a PyArrow Table (from memory) to a NumPack file.
    
    Parameters
    ----------
    table : pyarrow.Table
        Input PyArrow Table to save.
    output_path : str or Path
        Output NumPack directory path (.npk).
    array_name : str, optional
        Name of the array in the NumPack file. Default is 'data'.
    columns : list of str, optional
        Specific columns to save. If None, saves all columns as a 2D array.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    
    Raises
    ------
    DependencyError
        If PyArrow is not installed.
    
    Notes
    -----
    Zero-copy conversion is used when columns have no null values
    and are numeric types.
    
    Examples
    --------
    >>> import pyarrow.parquet as pq
    >>> from numpack.io import from_parquet_table
    >>> table = pq.read_table('data.parquet')
    >>> from_parquet_table(table, 'output.npk', array_name='my_data')
    """
    _check_pyarrow()
    
    if array_name is None:
        array_name = 'data'
    
    if columns is not None:
        table = table.select(columns)
    
    # Convert to numpy with zero-copy when possible
    arr = _table_to_numpy_zero_copy(table)
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    try:
        npk.save({array_name: arr})
    finally:
        npk.close()


def to_parquet_table(
    input_path: Union[str, Path],
    array_name: Optional[str] = None,
    column_names: Optional[List[str]] = None,
) -> Any:
    """Load an array from a NumPack file and return as a PyArrow Table.
    
    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path (.npk).
    array_name : str, optional
        Name of the array to load. If None and only one array exists,
        that array is loaded.
    column_names : list of str, optional
        Names for the columns in the resulting Table. If None, uses
        'col0', 'col1', etc. for 2D arrays, or 'data' for 1D arrays.
    
    Returns
    -------
    pyarrow.Table
        PyArrow Table loaded from the NumPack file.
    
    Raises
    ------
    DependencyError
        If PyArrow is not installed.
    ValueError
        If array_name is None and multiple arrays exist.
    
    Examples
    --------
    >>> from numpack.io import to_parquet_table
    >>> table = to_parquet_table('input.npk', array_name='embeddings')
    """
    _check_pyarrow()
    import pyarrow as pa
    
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
        
        # Convert to PyArrow Table
        if arr.ndim == 1:
            name = column_names[0] if column_names else 'data'
            return pa.table({name: arr})
        elif arr.ndim == 2:
            names = column_names or [f'col{i}' for i in range(arr.shape[1])]
            return pa.table({names[i]: arr[:, i] for i in range(arr.shape[1])})
        else:
            name = column_names[0] if column_names else 'data'
            return pa.table({name: arr.ravel()})
    finally:
        npk.close()


# =============================================================================
# File Conversions (Streaming)
# =============================================================================

def from_parquet_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Convert a Parquet file to NumPack format with streaming.
    
    Large Parquet files (> 1 GB) are imported by iterating record batches
    for memory efficiency.
    
    Parameters
    ----------
    input_path : str or Path
        Path to the input Parquet file (.parquet).
    output_path : str or Path
        Output NumPack directory path (.npk).
    array_name : str, optional
        Name for the output array. If None, uses the file stem.
    columns : list of str, optional
        Specific columns to convert. If None, converts all columns.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    chunk_size : int, optional
        Chunk size in bytes for streaming.
    
    Raises
    ------
    DependencyError
        If PyArrow is not installed.
    
    Examples
    --------
    >>> from numpack.io import from_parquet_file
    >>> from_parquet_file('data.parquet', 'output.npk')
    >>> from_parquet_file('data.parquet', 'output.npk', columns=['col1', 'col2'])
    """
    _check_pyarrow()
    import pyarrow.parquet as pq
    
    input_path = Path(input_path)
    
    if array_name is None:
        array_name = input_path.stem
    
    parquet_file = pq.ParquetFile(str(input_path))
    file_size = get_file_size(input_path)
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    
    try:
        if file_size > LARGE_FILE_THRESHOLD:
            _from_parquet_streaming(npk, parquet_file, array_name, columns)
        else:
            table = pq.read_table(str(input_path), columns=columns)
            arr = _table_to_numpy_zero_copy(table)
            npk.save({array_name: arr})
    finally:
        npk.close()


def to_parquet_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    compression: str = 'snappy',
    row_group_size: int = 100000,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Convert a NumPack file to Parquet format with streaming.
    
    Large arrays are written in chunks for memory efficiency.
    
    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path (.npk).
    output_path : str or Path
        Output Parquet file path (.parquet).
    array_name : str, optional
        Name of the array to export. If None and only one array exists,
        that array is used.
    compression : str, optional
        Compression codec ('snappy', 'gzip', 'brotli', 'zstd', 'lz4', 'none').
    row_group_size : int, optional
        Parquet row group size.
    chunk_size : int, optional
        Chunk size in bytes for streaming.
    
    Raises
    ------
    DependencyError
        If PyArrow is not installed.
    ValueError
        If `array_name` is None and multiple arrays exist.
    
    Examples
    --------
    >>> from numpack.io import to_parquet_file
    >>> to_parquet_file('input.npk', 'output.parquet')
    """
    _check_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq
    
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
        
        shape = npk.get_shape(array_name)
        arr_sample = npk.getitem(array_name, [0])
        dtype = arr_sample.dtype
        estimated_size = int(np.prod(shape)) * dtype.itemsize
        
        if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
            _to_parquet_streaming(
                npk, output_path, array_name, shape, dtype,
                compression, row_group_size, chunk_size,
            )
        else:
            arr = npk.load(array_name)
            table = _numpy_to_arrow_table(arr)
            pq.write_table(
                table, str(output_path),
                compression=compression,
                row_group_size=row_group_size,
            )
    finally:
        npk.close()


# =============================================================================
# Internal Helpers
# =============================================================================

def _numpy_to_arrow_table(arr: np.ndarray) -> Any:
    """Convert a NumPy array to a PyArrow Table.
    
    Parameters
    ----------
    arr : numpy.ndarray
        Input array.
    
    Returns
    -------
    pyarrow.Table
        PyArrow Table.
    """
    import pyarrow as pa
    
    if arr.ndim == 1:
        return pa.table({'data': arr})
    elif arr.ndim == 2:
        names = [f'col{i}' for i in range(arr.shape[1])]
        return pa.table({names[i]: arr[:, i] for i in range(arr.shape[1])})
    else:
        return pa.table({'data': arr.ravel()})


def _table_to_numpy_zero_copy(table) -> np.ndarray:
    """Convert a PyArrow Table to NumPy with zero-copy when possible.
    
    Parameters
    ----------
    table : pyarrow.Table
        Input Arrow table.
    
    Returns
    -------
    numpy.ndarray
        2D array with columns stacked horizontally.
    """
    import pyarrow as pa
    
    num_columns = table.num_columns
    num_rows = table.num_rows
    
    if num_columns == 0 or num_rows == 0:
        return np.empty((num_rows, num_columns))
    
    # Try zero-copy for each column
    columns = []
    
    for col_name in table.column_names:
        col = table.column(col_name).combine_chunks()
        
        # Check if zero-copy is possible
        can_zero_copy = (
            col.null_count == 0 and
            col.type in (
                pa.int8(), pa.int16(), pa.int32(), pa.int64(),
                pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64(),
                pa.float16(), pa.float32(), pa.float64(),
                pa.bool_(),
            )
        )
        
        if can_zero_copy:
            try:
                arr = col.to_numpy(zero_copy_only=True)
            except pa.ArrowInvalid:
                arr = col.to_numpy(zero_copy_only=False)
        else:
            arr = col.to_numpy(zero_copy_only=False)
        
        columns.append(arr)
    
    # Stack columns into 2D array
    if len(columns) == 1:
        result = columns[0].reshape(-1, 1) if columns[0].ndim == 1 else columns[0]
    else:
        result = np.column_stack(columns)
    
    return np.ascontiguousarray(result)


def _record_batch_to_numpy_zero_copy(batch) -> np.ndarray:
    """Convert a PyArrow RecordBatch to NumPy with zero-copy when possible.
    
    Parameters
    ----------
    batch : pyarrow.RecordBatch
        Input Arrow record batch.
    
    Returns
    -------
    numpy.ndarray
        2D array with columns stacked horizontally.
    """
    import pyarrow as pa
    
    num_columns = batch.num_columns
    num_rows = batch.num_rows
    
    if num_columns == 0 or num_rows == 0:
        return np.empty((num_rows, num_columns))
    
    # Try zero-copy for each column
    columns = []
    
    for i in range(num_columns):
        col = batch.column(i)
        
        # Check if zero-copy is possible
        can_zero_copy = (
            col.null_count == 0 and
            col.type in (
                pa.int8(), pa.int16(), pa.int32(), pa.int64(),
                pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64(),
                pa.float16(), pa.float32(), pa.float64(),
                pa.bool_(),
            )
        )
        
        if can_zero_copy:
            try:
                arr = col.to_numpy(zero_copy_only=True)
            except pa.ArrowInvalid:
                arr = col.to_numpy(zero_copy_only=False)
        else:
            arr = col.to_numpy(zero_copy_only=False)
        
        columns.append(arr)
    
    # Stack columns into 2D array
    if len(columns) == 1:
        result = columns[0].reshape(-1, 1) if columns[0].ndim == 1 else columns[0]
    else:
        result = np.column_stack(columns)
    
    return np.ascontiguousarray(result)


# =============================================================================
# Legacy Aliases (for backward compatibility)
# =============================================================================

from_parquet = from_parquet_file
to_parquet = to_parquet_file


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Memory conversions
    'from_parquet_table',
    'to_parquet_table',
    # File conversions
    'from_parquet_file',
    'to_parquet_file',
    # Legacy aliases
    'from_parquet',
    'to_parquet',
]


# NOTE: The following old code has been replaced by the new functions above.
# Keeping this comment for reference during the transition period.
# Old functions from_parquet and to_parquet are now aliases to from_parquet_file and to_parquet_file.

