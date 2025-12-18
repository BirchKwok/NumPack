from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Union

from .utils import DEFAULT_CHUNK_SIZE, _check_s3fs, _safe_unlink


# =============================================================================
# S3 远程存储支持
# =============================================================================

def from_s3(
    s3_path: str,
    output_path: Union[str, Path],
    format: str = 'auto',
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    **s3_kwargs,
) -> None:
    """从 S3 下载文件并导入为 NumPack 格式

    支持的格式：npy, npz, csv, parquet, feather, hdf5

    Parameters
    ----------
    s3_path : str
        S3 路径，格式为 's3://bucket/path/to/file'
    output_path : str or Path
        输出的 NumPack 文件路径
    format : str, optional
        文件格式，默认 'auto'（从扩展名推断）
    drop_if_exists : bool, optional
        如果输出文件存在是否删除，默认 False
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    **s3_kwargs
        传递给 s3fs 的其他参数（如 anon=True 用于公开桶）

    Examples
    --------
    >>> from numpack.io import from_s3
    >>> from_s3('s3://my-bucket/data.npy', 'output.npk')
    >>> from_s3('s3://public-bucket/data.csv', 'output.npk', anon=True)
    """
    s3fs = _check_s3fs()

    # 创建 S3 文件系统
    fs = s3fs.S3FileSystem(**s3_kwargs)

    # 推断格式
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

    # 下载到临时文件并转换
    with tempfile.NamedTemporaryFile(suffix=Path(s3_path).suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # 流式下载
        fs.get(s3_path, tmp_path)

        # 根据格式调用相应的导入函数
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
            raise ValueError(f"不支持的格式: {format}")

        handler(tmp_path, output_path, drop_if_exists=drop_if_exists, chunk_size=chunk_size)
    finally:
        # 清理临时文件
        _safe_unlink(tmp_path)


def to_s3(
    input_path: Union[str, Path],
    s3_path: str,
    format: str = 'auto',
    array_name: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    **s3_kwargs,
) -> None:
    """从 NumPack 导出并上传到 S3

    支持的格式：npy, npz, csv, parquet, feather, hdf5

    Parameters
    ----------
    input_path : str or Path
        输入的 NumPack 文件路径
    s3_path : str
        S3 路径，格式为 's3://bucket/path/to/file'
    format : str, optional
        输出文件格式，默认 'auto'（从扩展名推断）
    array_name : str, optional
        要导出的数组名
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    **s3_kwargs
        传递给 s3fs 的其他参数

    Examples
    --------
    >>> from numpack.io import to_s3
    >>> to_s3('input.npk', 's3://my-bucket/output.parquet')
    """
    s3fs = _check_s3fs()

    # 创建 S3 文件系统
    fs = s3fs.S3FileSystem(**s3_kwargs)

    # 推断格式
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

    # 导出到临时文件
    with tempfile.NamedTemporaryFile(suffix=Path(s3_path).suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        from .csv_io import to_csv, to_txt
        from .feather_io import to_feather
        from .hdf5_io import to_hdf5
        from .numpy_io import to_numpy
        from .parquet_io import to_parquet

        # 根据格式调用相应的导出函数
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
            raise ValueError(f"不支持的格式: {format}")

        handler(input_path, tmp_path)

        # 上传到 S3
        fs.put(tmp_path, s3_path)
    finally:
        # 清理临时文件
        _safe_unlink(tmp_path)
