from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from .utils import (
    DEFAULT_CHUNK_SIZE,
    LARGE_FILE_THRESHOLD,
    _check_h5py,
    estimate_chunk_rows,
    _open_numpack_for_read,
    _open_numpack_for_write,
)


# =============================================================================
# HDF5 格式转换
# =============================================================================

def from_hdf5(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    dataset_names: Optional[List[str]] = None,
    group: str = '/',
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """从 HDF5 文件导入为 NumPack 格式

    对于大数据集（>1GB），自动使用流式处理。

    Parameters
    ----------
    input_path : str or Path
        输入的 HDF5 文件路径
    output_path : str or Path
        输出的 NumPack 文件路径
    dataset_names : list of str, optional
        要导入的数据集名列表。如果为 None，导入组内所有数据集。
    group : str, optional
        HDF5 组路径，默认 '/'（根组）
    drop_if_exists : bool, optional
        如果输出文件存在是否删除，默认 False
    chunk_size : int, optional
        分块大小（字节），默认 100MB

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
                # 获取组内所有数据集
                dataset_names = [
                    name
                    for name in grp.keys()
                    if isinstance(grp[name], h5py.Dataset)
                ]

            for name in dataset_names:
                dataset = grp[name]
                if not isinstance(dataset, h5py.Dataset):
                    warnings.warn(f"跳过非数据集对象: {name}")
                    continue

                shape = dataset.shape
                dtype = dataset.dtype
                estimated_size = int(np.prod(shape)) * dtype.itemsize

                if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
                    # 大数据集：流式读取
                    _from_hdf5_dataset_streaming(npk, dataset, name, chunk_size)
                else:
                    # 小数据集：直接加载
                    arr = dataset[...]
                    npk.save({name: arr})
    finally:
        npk.close()


def _from_hdf5_dataset_streaming(
    npk: Any,
    dataset: Any,  # h5py.Dataset
    array_name: str,
    chunk_size: int,
) -> None:
    """流式导入 HDF5 数据集"""
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
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """从 NumPack 导出为 HDF5 格式

    对于大数组（>1GB），使用流式读取和分块写入。

    Parameters
    ----------
    input_path : str or Path
        输入的 NumPack 文件路径
    output_path : str or Path
        输出的 HDF5 文件路径
    array_names : list of str, optional
        要导出的数组名列表。如果为 None，导出所有数组。
    group : str, optional
        HDF5 组路径，默认 '/'（根组）
    compression : str, optional
        压缩算法，默认 'gzip'。设为 None 禁用压缩。
    compression_opts : int, optional
        压缩级别（0-9），默认 4
    chunk_size : int, optional
        分块大小（字节），默认 100MB

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
            # 创建组（如果需要）
            if group != '/':
                grp = h5f.require_group(group)
            else:
                grp = h5f

            for name in array_names:
                shape = npk.get_shape(name)
                arr_sample = npk.getitem(name, [0])
                dtype = arr_sample.dtype
                estimated_size = int(np.prod(shape)) * dtype.itemsize

                # 计算 HDF5 分块大小
                if len(shape) > 0:
                    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
                    chunks = (min(batch_rows, shape[0]),) + shape[1:]
                else:
                    chunks = None

                # 创建数据集
                ds = grp.create_dataset(
                    name,
                    shape=shape,
                    dtype=dtype,
                    chunks=chunks if chunks else True,
                    compression=compression,
                    compression_opts=compression_opts if compression else None,
                )

                if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
                    # 大数组：流式写入
                    _to_hdf5_dataset_streaming(npk, ds, name, shape, dtype, chunk_size)
                else:
                    # 小数组：直接写入
                    ds[...] = npk.load(name)
    finally:
        npk.close()


def _to_hdf5_dataset_streaming(
    npk: Any,
    dataset: Any,  # h5py.Dataset
    array_name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    chunk_size: int,
) -> None:
    """流式导出大数组到 HDF5 数据集"""
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]

    for start_idx in range(0, total_rows, batch_rows):
        end_idx = min(start_idx + batch_rows, total_rows)
        indices = list(range(start_idx, end_idx))
        chunk = npk.getitem(array_name, indices)
        dataset[start_idx:end_idx] = chunk
