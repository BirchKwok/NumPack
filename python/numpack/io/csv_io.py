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
# CSV 格式转换
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
    """从 CSV 文件导入为 NumPack 格式

    对于大文件（>1GB），自动使用流式处理。

    Parameters
    ----------
    input_path : str or Path
        输入的 CSV 文件路径
    output_path : str or Path
        输出的 NumPack 文件路径
    array_name : str, optional
        数组名称，默认使用文件名（不含扩展名）
    drop_if_exists : bool, optional
        如果输出文件存在是否删除，默认 False
    dtype : numpy.dtype, optional
        数据类型，默认自动推断
    delimiter : str, optional
        分隔符，默认 ','
    skiprows : int, optional
        跳过的行数，默认 0
    max_rows : int, optional
        最大读取行数，默认 None（读取全部）
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    **kwargs
        传递给 numpy.loadtxt 或 pandas.read_csv 的其他参数

    Examples
    --------
    >>> from numpack.io import from_csv
    >>> from_csv('data.csv', 'output.npk')
    >>> from_csv('data.csv', 'output.npk', dtype=np.float32, delimiter=';')
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")

    if array_name is None:
        array_name = input_path.stem

    file_size = get_file_size(input_path)

    if file_size > LARGE_FILE_THRESHOLD:
        # 大文件：使用 pandas 分块读取（如果可用）
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
        # 小文件：直接加载
        try:
            # 尝试使用 pandas（更快更灵活）
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
            # 确保数组是 C-contiguous 的（NumPack 要求）
            arr = np.ascontiguousarray(df.values)
        except DependencyError:
            # 回退到 numpy
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
    """大文件 CSV 流式导入"""
    pd = _check_pandas()

    # 计算分块行数（估算每行约 100 字节）
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
            # 确保数组是 C-contiguous 的（NumPack 要求）
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
    """从 NumPack 导出为 CSV 格式

    对于大文件（>1GB），使用流式读取。

    Parameters
    ----------
    input_path : str or Path
        输入的 NumPack 文件路径
    output_path : str or Path
        输出的 CSV 文件路径
    array_name : str, optional
        要导出的数组名，如果为 None 且只有一个数组则使用该数组
    delimiter : str, optional
        分隔符，默认 ','
    header : bool, optional
        是否写入列标题，默认 False
    fmt : str, optional
        数值格式，默认 '%.18e'
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    **kwargs
        传递给 numpy.savetxt 的其他参数

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
                    f"NumPack 文件包含多个数组 {members}，请指定 array_name 参数"
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
    """流式导出大数组为 CSV 文件"""
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]

    with open(output_path, 'w') as f:
        # 写入标题
        if header and len(shape) > 1:
            header_line = delimiter.join([f'col{i}' for i in range(shape[1])])
            f.write(f"# {header_line}\n")

        for start_idx in range(0, total_rows, batch_rows):
            end_idx = min(start_idx + batch_rows, total_rows)
            chunk = npk.getitem(array_name, slice(start_idx, end_idx))

            # 写入分块
            # 2D 数组用 numpy.savetxt 更快（C 实现），避免逐行 Python 循环
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
# TXT 格式转换（与 CSV 类似，但默认空格分隔）
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
    """从 TXT 文件导入为 NumPack 格式

    与 from_csv 类似，但默认使用空白字符作为分隔符。

    Parameters
    ----------
    input_path : str or Path
        输入的 TXT 文件路径
    output_path : str or Path
        输出的 NumPack 文件路径
    array_name : str, optional
        数组名称，默认使用文件名（不含扩展名）
    drop_if_exists : bool, optional
        如果输出文件存在是否删除，默认 False
    dtype : numpy.dtype, optional
        数据类型，默认自动推断
    delimiter : str, optional
        分隔符，默认 None（任意空白字符）
    skiprows : int, optional
        跳过的行数，默认 0
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    **kwargs
        传递给 numpy.loadtxt 的其他参数

    Examples
    --------
    >>> from numpack.io import from_txt
    >>> from_txt('data.txt', 'output.npk')
    """
    # 使用 from_csv 但默认分隔符为空白
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
    """从 NumPack 导出为 TXT 格式

    与 to_csv 类似，但默认使用空格作为分隔符。

    Parameters
    ----------
    input_path : str or Path
        输入的 NumPack 文件路径
    output_path : str or Path
        输出的 TXT 文件路径
    array_name : str, optional
        要导出的数组名
    delimiter : str, optional
        分隔符，默认 ' '（空格）
    fmt : str, optional
        数值格式，默认 '%.18e'
    chunk_size : int, optional
        分块大小（字节），默认 100MB
    **kwargs
        传递给 numpy.savetxt 的其他参数

    Examples
    --------
    >>> from numpack.io import to_txt
    >>> to_txt('input.npk', 'output.txt')
    """
    to_csv(input_path, output_path, array_name, delimiter, False, fmt, chunk_size, **kwargs)
