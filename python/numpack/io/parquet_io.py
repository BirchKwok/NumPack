from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

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
# Parquet format conversion
# =============================================================================

def from_parquet(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
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

    Raises
    ------
    DependencyError
        If the optional dependency ``pyarrow`` is not installed.

    Examples
    --------
    >>> from numpack.io import from_parquet
    >>> from_parquet('data.parquet', 'output.npk')
    >>> from_parquet('data.parquet', 'output.npk', columns=['col1', 'col2'])
    """
    _check_pyarrow()
    import pyarrow.parquet as pq

    input_path = Path(input_path)

    if array_name is None:
        array_name = input_path.stem

    # Read Parquet metadata
    parquet_file = pq.ParquetFile(str(input_path))
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
    columns: Optional[List[str]],
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
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Export a NumPack array to Parquet.

    Parameters
    ----------
    input_path : str or Path
        Input NumPack directory path.
    output_path : str or Path
        Output Parquet file path.
    array_name : str, optional
        Name of the array to export. If None, the array is inferred only when
        the NumPack file contains exactly one array.
    compression : str, optional
        Parquet compression codec.
    row_group_size : int, optional
        Parquet row group size used for non-streaming writes.
    chunk_size : int, optional
        Chunk size in bytes used for streaming export.

    Raises
    ------
    DependencyError
        If the optional dependency ``pyarrow`` is not installed.
    ValueError
        If `array_name` is not provided and the NumPack file contains multiple arrays.

    Examples
    --------
    >>> from numpack.io import to_parquet
    >>> to_parquet('input.npk', 'output.parquet')
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
                    f"NumPack contains multiple arrays {members}; please provide the array_name argument."
                )

        shape = npk.get_shape(array_name)
        arr_sample = npk.getitem(array_name, [0])
        dtype = arr_sample.dtype
        estimated_size = int(np.prod(shape)) * dtype.itemsize

        if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
            # Large array: streamed writes
            _to_parquet_streaming(
                npk,
                output_path,
                array_name,
                shape,
                dtype,
                compression,
                row_group_size,
                chunk_size,
            )
        else:
            # Small array: write directly
            arr = npk.load(array_name)
            # Convert to a PyArrow Table
            if arr.ndim == 1:
                table = pa.table({'data': arr})
            else:
                # Convert a multi-dimensional array into columns
                columns = (
                    {f'col{i}': arr[:, i] for i in range(arr.shape[1])}
                    if arr.ndim == 2
                    else {'data': arr.flatten()}
                )
                table = pa.table(columns)

            pq.write_table(
                table,
                str(output_path),
                compression=compression,
                row_group_size=row_group_size,
            )
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
    chunk_size: int,
) -> None:
    """Stream-export a large array to Parquet."""
    _check_pyarrow()
    import pyarrow as pa
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
                columns = (
                    {f'col{i}': chunk[:, i] for i in range(chunk.shape[1])}
                    if chunk.ndim == 2
                    else {'data': chunk.flatten()}
                )
                table = pa.table(columns)

            if writer is None:
                writer = pq.ParquetWriter(
                    str(output_path),
                    table.schema,
                    compression=compression,
                )

            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
