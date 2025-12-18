from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Union

from .utils import DEFAULT_CHUNK_SIZE, _check_s3fs, _safe_unlink


# =============================================================================
# S3 remote storage support
# =============================================================================

def from_s3(
    s3_path: str,
    output_path: Union[str, Path],
    format: str = 'auto',
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    **s3_kwargs,
) -> None:
    """Download an object from S3 and import it into NumPack.

    Supported formats include NumPy (``.npy``/``.npz``), CSV/TXT, Parquet,
    Feather and HDF5.

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

    Raises
    ------
    DependencyError
        If the optional dependency ``s3fs`` is not installed.
    ValueError
        If `format` is not supported.

    Examples
    --------
    >>> from numpack.io import from_s3
    >>> from_s3('s3://my-bucket/data.npy', 'output.npk')
    >>> from_s3('s3://public-bucket/data.csv', 'output.npk', anon=True)
    """
    s3fs = _check_s3fs()

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
        from .csv_io import from_csv, from_txt
        from .feather_io import from_feather
        from .hdf5_io import from_hdf5
        from .numpy_io import from_numpy
        from .parquet_io import from_parquet

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
        _safe_unlink(tmp_path)


def to_s3(
    input_path: Union[str, Path],
    s3_path: str,
    format: str = 'auto',
    array_name: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    **s3_kwargs,
) -> None:
    """Export a NumPack array and upload it to S3.

    Supported formats include NumPy (``.npy``/``.npz``), CSV/TXT, Parquet,
    Feather and HDF5.

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

    Raises
    ------
    DependencyError
        If the optional dependency ``s3fs`` is not installed.
    ValueError
        If `format` is not supported.

    Examples
    --------
    >>> from numpack.io import to_s3
    >>> to_s3('input.npk', 's3://my-bucket/output.parquet')
    """
    s3fs = _check_s3fs()

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
        from .csv_io import to_csv, to_txt
        from .feather_io import to_feather
        from .hdf5_io import to_hdf5
        from .numpy_io import to_numpy
        from .parquet_io import to_parquet

        # Dispatch to the corresponding export function
        format_handlers = {
            'numpy': lambda inp, out, **kw: to_numpy(
                inp,
                out,
                array_names=[array_name] if array_name else None,
                chunk_size=chunk_size,
            ),
            'csv': lambda inp, out, **kw: to_csv(inp, out, array_name=array_name, chunk_size=chunk_size),
            'txt': lambda inp, out, **kw: to_txt(inp, out, array_name=array_name, chunk_size=chunk_size),
            'parquet': lambda inp, out, **kw: to_parquet(inp, out, array_name=array_name, chunk_size=chunk_size),
            'feather': lambda inp, out, **kw: to_feather(inp, out, array_name=array_name, chunk_size=chunk_size),
            'hdf5': lambda inp, out, **kw: to_hdf5(
                inp,
                out,
                array_names=[array_name] if array_name else None,
                chunk_size=chunk_size,
            ),
        }

        handler = format_handlers.get(format)
        if handler is None:
            raise ValueError(f"Unsupported format: {format}")

        handler(input_path, tmp_path)

        # Upload to S3
        fs.put(tmp_path, s3_path)
    finally:
        # Clean up temporary file
        _safe_unlink(tmp_path)
