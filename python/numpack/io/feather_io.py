"""Apache Feather/Arrow conversion utilities for NumPack.

Feather is a fast, lightweight columnar format based on Apache Arrow,
designed for efficient data exchange between Python and R.

This module provides two types of conversions:

1. **File conversions**:
   - `from_feather(feather_path, npk_path)` - Convert .feather to .npk
   - `to_feather(npk_path, feather_path)` - Convert .npk to .feather

2. **Memory-to-file / File-to-memory conversions**:
   - `from_table(table, npk_path)` - Save PyArrow Table to .npk file
   - `to_table(npk_path, array_name)` - Load from .npk file and return PyArrow Table
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .utils import (
    DEFAULT_CHUNK_SIZE,
    LARGE_FILE_THRESHOLD,
    _check_pyarrow,
    _open_numpack_for_read,
    _open_numpack_for_write,
)


# =============================================================================
# Memory-to-File / File-to-Memory Conversions
# =============================================================================

def from_table(
    table: Any,
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
    drop_if_exists: bool = False,
    return_npk_obj: bool = False,
) -> Any:
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
    return_npk_obj : bool, optional
        If True, return an opened NumPack instance for output_path.
    
    Returns
    -------
    NumPack or None
        The NumPack instance if `return_npk_obj` is True, otherwise None.
    
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
    >>> import pyarrow as pa
    >>> from numpack.io import from_table
    >>> table = pa.table({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
    >>> from_table(table, 'output.npk', array_name='my_data')
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

    if return_npk_obj:
        return _open_numpack_for_read(output_path)
    return None


def to_table(
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
    >>> from numpack.io import to_table
    >>> table = to_table('input.npk', array_name='embeddings')
    >>> table.column_names
    ['col0', 'col1', ...]
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

def from_feather_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
    drop_if_exists: bool = False,
    return_npk_obj: bool = False,
) -> Any:
    """Convert a Feather file to NumPack format.
    
    Parameters
    ----------
    input_path : str or Path
        Path to the input Feather file (.feather).
    output_path : str or Path
        Output NumPack directory path (.npk).
    array_name : str, optional
        Name for the output array. If None, uses the file stem.
    columns : list of str, optional
        Specific columns to convert. If None, converts all columns.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    
    Raises
    ------
    DependencyError
        If PyArrow is not installed.
    
    Examples
    --------
    >>> from numpack.io import from_feather_file
    >>> from_feather_file('data.feather', 'output.npk')
    """
    _check_pyarrow()
    import pyarrow.feather as feather
    
    input_path = Path(input_path)
    
    if array_name is None:
        array_name = input_path.stem
    
    # Read Feather file
    table = feather.read_table(str(input_path), columns=columns)
    
    # Convert to numpy with zero-copy
    arr = _table_to_numpy_zero_copy(table)
    
    npk = _open_numpack_for_write(output_path, drop_if_exists)
    try:
        npk.save({array_name: arr})
    finally:
        npk.close()

    if return_npk_obj:
        return _open_numpack_for_read(output_path)
    return None


def to_feather_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    compression: str = 'zstd',
) -> None:
    """Convert a NumPack file to Feather format.
    
    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path (.npk).
    output_path : str or Path
        Output Feather file path (.feather).
    array_name : str, optional
        Name of the array to export. If None and only one array exists,
        that array is used.
    compression : str, optional
        Compression codec ('zstd', 'lz4', 'uncompressed').
    
    Raises
    ------
    DependencyError
        If PyArrow is not installed.
    ValueError
        If `array_name` is None and multiple arrays exist.
    
    Notes
    -----
    Feather does not support true streaming writes. For very large arrays,
    consider using Parquet format instead.
    
    Examples
    --------
    >>> from numpack.io import to_feather_file
    >>> to_feather_file('input.npk', 'output.feather')
    """
    _check_pyarrow()
    import pyarrow as pa
    import pyarrow.feather as feather
    
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
        estimated_size = int(np.prod(shape)) * npk.getitem(array_name, [0]).dtype.itemsize
        
        if estimated_size > LARGE_FILE_THRESHOLD:
            warnings.warn(
                f"Array '{array_name}' is large (>{estimated_size / 1e9:.1f}GB). "
                "Feather loads all data into memory. "
                "For large datasets, consider using to_parquet_file.",
                UserWarning,
            )
        
        arr = npk.load(array_name)
        
        # Convert to PyArrow Table
        table = _numpy_to_arrow_table(arr)
        
        feather.write_feather(table, str(output_path), compression=compression)
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
    """Convert a PyArrow Table to a 2D NumPy array with zero-copy when possible."""
    import pyarrow as pa
    
    num_columns = table.num_columns
    num_rows = table.num_rows
    
    if num_columns == 0 or num_rows == 0:
        return np.empty((num_rows, num_columns))
    
    columns = []
    for col_name in table.column_names:
        col = table.column(col_name).combine_chunks()
        
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
    
    if len(columns) == 1:
        result = columns[0].reshape(-1, 1) if columns[0].ndim == 1 else columns[0]
    else:
        result = np.column_stack(columns)
    
    return result


# =============================================================================
# Primary Names and Aliases
# =============================================================================

from .utils import deprecated_alias

# Primary names for file conversion (recommended)
from_feather = from_feather_file
to_feather = to_feather_file

# Legacy aliases for memory conversions (deprecated, will be removed in 0.6.0)
from_arrow = deprecated_alias('from_table', from_table)
from_arrow.__name__ = 'from_arrow'

to_arrow = deprecated_alias('to_table', to_table)
to_arrow.__name__ = 'to_arrow'


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Primary names (recommended) - File conversions
    'from_feather',
    'to_feather',
    # Primary names (recommended) - Memory conversions
    'from_table',
    'to_table',
    # Verbose aliases (for clarity)
    'from_feather_file',
    'to_feather_file',
    # Legacy aliases (deprecated, will be removed in 0.6.0)
    'from_arrow',
    'to_arrow',
]
