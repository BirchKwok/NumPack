from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from .utils import (
    DEFAULT_CHUNK_SIZE,
    LARGE_FILE_THRESHOLD,
    DependencyError,
    _check_pandas,
    estimate_chunk_rows,
    get_file_size,
    _open_numpack_for_read,
    _open_numpack_for_write,
)


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
    **kwargs,
) -> None:
    """Import a CSV file into NumPack.

    For large files (by default > 1 GB), this function uses pandas chunked I/O
    when available.

    Parameters
    ----------
    input_path : str or Path
        Path to the input CSV file.
    output_path : str or Path
        Output NumPack directory path.
    array_name : str, optional
        Name of the array to store. If None, defaults to the file stem.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    dtype : numpy.dtype, optional
        Target dtype. If None, dtype is inferred by the reader.
    delimiter : str, optional
        Column delimiter.
    skiprows : int, optional
        Number of rows to skip at the start of the file.
    max_rows : int, optional
        Maximum number of rows to read. If None, reads all rows.
    chunk_size : int, optional
        Target chunk size in bytes for streaming conversion.
    **kwargs
        Additional keyword arguments forwarded to `pandas.read_csv` (preferred) or
        `numpy.loadtxt` (fallback).

    Raises
    ------
    FileNotFoundError
        If `input_path` does not exist.
    DependencyError
        If pandas is required for streaming conversion but is not installed.

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
        # Large file: use pandas chunked reads when available
        _from_csv_streaming(
            input_path,
            output_path,
            array_name,
            drop_if_exists,
            dtype,
            delimiter,
            skiprows,
            chunk_size,
            **kwargs,
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
                **kwargs,
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
                **{k: v for k, v in kwargs.items() if k in ['comments', 'usecols', 'unpack', 'ndmin', 'encoding']},
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
    **kwargs,
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
            **kwargs,
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
    **kwargs,
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
        Name of the array to export. If None, the array is inferred only when the
        NumPack file contains exactly one array.
    delimiter : str, optional
        Column delimiter.
    header : bool, optional
        If True, write a header line (only meaningful for 2D arrays).
    fmt : str, optional
        Numeric format string passed to `numpy.savetxt`.
    chunk_size : int, optional
        Chunk size in bytes used for streaming export.
    **kwargs
        Additional keyword arguments forwarded to `numpy.savetxt`.

    Raises
    ------
    ValueError
        If `array_name` is not provided and the NumPack file contains multiple arrays.

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
            _to_csv_streaming(
                npk,
                output_path,
                array_name,
                shape,
                dtype,
                delimiter,
                header,
                fmt,
                chunk_size,
                **kwargs,
            )
        else:
            arr = npk.load(array_name)
            np.savetxt(
                str(output_path),
                arr,
                delimiter=delimiter,
                fmt=fmt,
                header=''
                if not header
                else delimiter.join(
                    [
                        f'col{i}'
                        for i in range(arr.shape[1] if arr.ndim > 1 else 1)
                    ]
                ),
                **kwargs,
            )
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
    **kwargs,
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
            # numpy.savetxt is faster for 2D arrays (C implementation), avoiding per-row Python loops
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
    **kwargs,
) -> None:
    """Import a whitespace-delimited text file into NumPack.

    This function is equivalent to `from_csv` but uses whitespace as the default
    delimiter.

    Parameters
    ----------
    input_path : str or Path
        Path to the input text file.
    output_path : str or Path
        Output NumPack directory path.
    array_name : str, optional
        Name of the array to store. If None, defaults to the file stem.
    drop_if_exists : bool, optional
        If True, delete the output path first if it already exists.
    dtype : numpy.dtype, optional
        Target dtype.
    delimiter : str, optional
        Delimiter passed through to `from_csv`. If None, whitespace is used.
    skiprows : int, optional
        Number of rows to skip.
    chunk_size : int, optional
        Chunk size in bytes used for streaming conversion.
    **kwargs
        Additional keyword arguments forwarded to the underlying reader.

    Raises
    ------
    FileNotFoundError
        If `input_path` does not exist.
    DependencyError
        If pandas is required for streaming conversion but is not installed.

    Examples
    --------
    >>> from numpack.io import from_txt
    >>> from_txt('data.txt', 'output.npk')
    """
    # Use from_csv but default delimiter is whitespace
    from_csv(
        input_path,
        output_path,
        array_name,
        drop_if_exists,
        dtype,
        delimiter if delimiter else ' ',
        skiprows,
        None,
        chunk_size,
        **kwargs,
    )


def to_txt(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    delimiter: str = ' ',
    fmt: str = '%.18e',
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    **kwargs,
) -> None:
    """Export a NumPack array to a whitespace-delimited text file.

    This function is equivalent to `to_csv` but uses a space character as the
    default delimiter.

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
        Numeric format string passed to `numpy.savetxt`.
    chunk_size : int, optional
        Chunk size in bytes used for streaming export.
    **kwargs
        Additional keyword arguments forwarded to `numpy.savetxt`.

    Raises
    ------
    ValueError
        If `array_name` is not provided and the NumPack file contains multiple arrays.

    Examples
    --------
    >>> from numpack.io import to_txt
    >>> to_txt('input.npk', 'output.txt')
    """
    to_csv(input_path, output_path, array_name, delimiter, False, fmt, chunk_size, **kwargs)
