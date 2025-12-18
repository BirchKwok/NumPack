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
# Parquet 格式转换
# =============================================================================

def from_parquet(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """从 Parquet 文件导入为 NumPack 格式

    对于大文件（>1GB），自动使用流式处理。

    Parameters
    ----------
    input_path : str or Path
        输入的 Parquet 文件路径
    output_path : str or Path
        输出的 NumPack 文件路径
    array_name : str, optional
        数组名称，默认使用文件名（不含扩展名）
    columns : list of str, optional
        要读取的列名列表，默认读取全部
    drop_if_exists : bool, optional
        如果输出文件存在是否删除，默认 False
    chunk_size : int, optional
        分块大小（字节），默认 100MB

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

    # 获取文件元信息
    parquet_file = pq.ParquetFile(str(input_path))
    file_size = get_file_size(input_path)

    npk = _open_numpack_for_write(output_path, drop_if_exists)

    try:
        if file_size > LARGE_FILE_THRESHOLD:
            # 大文件：按行组流式读取
            _from_parquet_streaming(npk, parquet_file, array_name, columns)
        else:
            # 小文件：直接加载
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
    """流式导入 Parquet 文件"""
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
    """从 NumPack 导出为 Parquet 格式

    对于大数组（>1GB），使用流式读取和分批写入。

    Parameters
    ----------
    input_path : str or Path
        输入的 NumPack 文件路径
    output_path : str or Path
        输出的 Parquet 文件路径
    array_name : str, optional
        要导出的数组名
    compression : str, optional
        压缩算法，默认 'snappy'
    row_group_size : int, optional
        行组大小，默认 100000
    chunk_size : int, optional
        分块大小（字节），默认 100MB

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
                    f"NumPack 文件包含多个数组 {members}，请指定 array_name 参数"
                )

        shape = npk.get_shape(array_name)
        arr_sample = npk.getitem(array_name, [0])
        dtype = arr_sample.dtype
        estimated_size = int(np.prod(shape)) * dtype.itemsize

        if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
            # 大数组：流式写入
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
            # 小数组：直接写入
            arr = npk.load(array_name)
            # 转换为 PyArrow Table
            if arr.ndim == 1:
                table = pa.table({'data': arr})
            else:
                # 多维数组转换为列
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
    """流式导出大数组到 Parquet"""
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

            # 转换为 Table
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
