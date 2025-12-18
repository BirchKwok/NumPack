from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from .utils import (
    DEFAULT_CHUNK_SIZE,
    LARGE_FILE_THRESHOLD,
    _check_zarr,
    estimate_chunk_rows,
    _open_numpack_for_read,
    _open_numpack_for_write,
)


# =============================================================================
# Zarr 格式转换
# =============================================================================

def from_zarr(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
    group: str = '/',
    drop_if_exists: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """从 Zarr 存储导入为 NumPack 格式

    Zarr 原生支持分块存储，对于大数据集自动使用流式处理。

    Parameters
    ----------
    input_path : str or Path
        输入的 Zarr 存储路径（目录或 .zarr 文件）
    output_path : str or Path
        输出的 NumPack 文件路径
    array_names : list of str, optional
        要导入的数组名列表。如果为 None，导入组内所有数组。
    group : str, optional
        Zarr 组路径，默认 '/'（根组）
    drop_if_exists : bool, optional
        如果输出文件存在是否删除，默认 False
    chunk_size : int, optional
        分块大小（字节），默认 100MB

    Examples
    --------
    >>> from numpack.io import from_zarr
    >>> from_zarr('data.zarr', 'output.npk')
    >>> from_zarr('data.zarr', 'output.npk', array_names=['arr1', 'arr2'])
    """
    zarr = _check_zarr()

    npk = _open_numpack_for_write(output_path, drop_if_exists)

    try:
        store = zarr.open(str(input_path), mode='r')
        if group != '/':
            store = store[group]

        if array_names is None:
            # 获取所有数组
            array_names = [name for name in store.array_keys()]

        for name in array_names:
            arr = store[name]
            shape = arr.shape
            dtype = arr.dtype
            estimated_size = int(np.prod(shape)) * dtype.itemsize

            if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
                # 大数组：流式读取
                _from_zarr_array_streaming(npk, arr, name, chunk_size)
            else:
                # 小数组：直接加载
                npk.save({name: arr[...]})
    finally:
        npk.close()


def _from_zarr_array_streaming(
    npk: Any,
    zarr_arr: Any,  # zarr.Array
    array_name: str,
    chunk_size: int,
) -> None:
    """流式导入 Zarr 数组"""
    shape = zarr_arr.shape
    dtype = zarr_arr.dtype
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]

    for start_idx in range(0, total_rows, batch_rows):
        end_idx = min(start_idx + batch_rows, total_rows)
        chunk = zarr_arr[start_idx:end_idx]

        if start_idx == 0:
            npk.save({array_name: chunk})
        else:
            npk.append({array_name: chunk})


def to_zarr(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    array_names: Optional[List[str]] = None,
    group: str = '/',
    compressor: Optional[str] = 'default',
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """从 NumPack 导出为 Zarr 格式

    Zarr 原生支持分块存储，适合大数据集。

    Parameters
    ----------
    input_path : str or Path
        输入的 NumPack 文件路径
    output_path : str or Path
        输出的 Zarr 存储路径
    array_names : list of str, optional
        要导出的数组名列表。如果为 None，导出所有数组。
    group : str, optional
        Zarr 组路径，默认 '/'（根组）
    compressor : str or None, optional
        压缩器，默认 'default'（使用 Blosc）。设为 None 禁用压缩。
    chunk_size : int, optional
        分块大小（字节），默认 100MB

    Examples
    --------
    >>> from numpack.io import to_zarr
    >>> to_zarr('input.npk', 'output.zarr')
    """
    zarr = _check_zarr()

    npk = _open_numpack_for_read(input_path)

    try:
        if array_names is None:
            array_names = npk.get_member_list()

        # 创建 Zarr 存储
        store = zarr.open(str(output_path), mode='w')
        if group != '/':
            store = store.require_group(group)

        # 配置压缩器
        if compressor == 'default':
            try:
                from zarr.codecs import BloscCodec, BloscCname, BloscShuffle

                compressor_obj = BloscCodec(
                    cname=BloscCname.zstd,
                    clevel=3,
                    shuffle=BloscShuffle.bitshuffle,
                )
            except ImportError:
                compressor_obj = None
        elif compressor is None:
            compressor_obj = None
        else:
            compressor_obj = compressor

        for name in array_names:
            shape = npk.get_shape(name)
            arr_sample = npk.getitem(name, [0])
            dtype = arr_sample.dtype
            estimated_size = int(np.prod(shape)) * dtype.itemsize

            # 计算分块大小
            if len(shape) > 0:
                batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
                chunks = (min(batch_rows, shape[0]),) + shape[1:]
            else:
                chunks = shape

            # 创建 Zarr 数组
            if hasattr(store, 'create_array'):
                zarr_arr = store.create_array(
                    name,
                    shape=shape,
                    dtype=dtype,
                    chunks=chunks if chunks else None,
                    compressors=compressor_obj,
                    overwrite=True,
                )
            else:
                zarr_arr = store.create_dataset(
                    name,
                    shape=shape,
                    dtype=dtype,
                    chunks=chunks if chunks else None,
                    compressor=compressor_obj,
                )

            if estimated_size > LARGE_FILE_THRESHOLD and len(shape) > 0:
                # 大数组：流式写入
                _to_zarr_array_streaming(npk, zarr_arr, name, shape, dtype, chunk_size)
            else:
                # 小数组：直接写入
                zarr_arr[...] = npk.load(name)
    finally:
        npk.close()


def _to_zarr_array_streaming(
    npk: Any,
    zarr_arr: Any,  # zarr.Array
    array_name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    chunk_size: int,
) -> None:
    """流式导出大数组到 Zarr"""
    batch_rows = estimate_chunk_rows(shape, dtype, chunk_size)
    total_rows = shape[0]

    for start_idx in range(0, total_rows, batch_rows):
        end_idx = min(start_idx + batch_rows, total_rows)
        chunk = npk.getitem(array_name, slice(start_idx, end_idx))
        zarr_arr[start_idx:end_idx] = chunk
