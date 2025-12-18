from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .utils import (
    DEFAULT_CHUNK_SIZE,
    LARGE_FILE_THRESHOLD,
    _check_pyarrow,
    _open_numpack_for_read,
    _open_numpack_for_write,
)


# =============================================================================
# Feather 格式转换
# =============================================================================

def from_feather(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
    drop_if_exists: bool = False,
) -> None:
    """从 Feather 文件导入为 NumPack 格式

    Feather 是一种快速、轻量级的列式存储格式。

    Parameters
    ----------
    input_path : str or Path
        输入的 Feather 文件路径
    output_path : str or Path
        输出的 NumPack 文件路径
    array_name : str, optional
        数组名称，默认使用文件名（不含扩展名）
    columns : list of str, optional
        要读取的列名列表
    drop_if_exists : bool, optional
        如果输出文件存在是否删除，默认 False

    Examples
    --------
    >>> from numpack.io import from_feather
    >>> from_feather('data.feather', 'output.npk')
    """
    _check_pyarrow()
    import pyarrow.feather as feather

    input_path = Path(input_path)

    if array_name is None:
        array_name = input_path.stem

    # Feather 格式需要一次性加载（不支持流式读取）
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
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """从 NumPack 导出为 Feather 格式

    Feather 是一种快速、轻量级的列式存储格式。
    注意：Feather 不支持真正的流式写入，大文件会完全加载到内存。

    Parameters
    ----------
    input_path : str or Path
        输入的 NumPack 文件路径
    output_path : str or Path
        输出的 Feather 文件路径
    array_name : str, optional
        要导出的数组名
    compression : str, optional
        压缩算法，默认 'zstd'
    chunk_size : int, optional
        分块大小（字节），默认 100MB

    Examples
    --------
    >>> from numpack.io import to_feather
    >>> to_feather('input.npk', 'output.feather')
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
                    f"NumPack 文件包含多个数组 {members}，请指定 array_name 参数"
                )

        shape = npk.get_shape(array_name)
        estimated_size = int(np.prod(shape)) * npk.getitem(array_name, [0]).dtype.itemsize

        if estimated_size > LARGE_FILE_THRESHOLD:
            warnings.warn(
                f"数组 '{array_name}' 较大（>{estimated_size / 1e9:.1f}GB），"
                "Feather 格式需要将所有数据加载到内存。"
                "对于大数据集，建议使用 to_parquet 或 to_zarr。",
                UserWarning,
            )

        arr = npk.load(array_name)

        # 转换为 Table
        if arr.ndim == 1:
            table = pa.table({'data': arr})
        else:
            columns = (
                {f'col{i}': arr[:, i] for i in range(arr.shape[1])}
                if arr.ndim == 2
                else {'data': arr.flatten()}
            )
            table = pa.table(columns)

        feather.write_feather(table, str(output_path), compression=compression)
    finally:
        npk.close()
