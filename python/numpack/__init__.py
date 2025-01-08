import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union, Optional
import numpy as np
from collections import OrderedDict
import threading
import resource
import os

from numpack._lib_numpack import NumPack as _NumPack

# package message
__version__ = "0.1.0"

# 获取系统最大文件描述符限制
soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)

class MmapCache:
    """LRU缓存，用于管理内存映射对象"""
    def __init__(self, max_size: int = max(soft_limit // 2, 100)):
        """初始化LRU缓存
        
        Parameters:
            max_size (int): 最大缓存数量，默认为系统文件描述符限制的一半或100，取较大值
        """
        self.max_size = max_size
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[np.memmap]:
        """获取缓存的内存映射对象
        
        Parameters:
            key (str): 缓存键值
            
        Returns:
            Optional[np.memmap]: 内存映射对象，如果不存在则返回None
        """
        with self._lock:
            if key in self._cache:
                # 移动到最新使用
                value = self._cache.pop(key)
                self._cache[key] = value
                return value
            return None

    def put(self, key: str, value: np.memmap) -> None:
        """添加内存映射对象到缓存
        
        Parameters:
            key (str): 缓存键值
            value (np.memmap): 内存映射对象
        """
        with self._lock:
            if key in self._cache:
                # 更新现有项
                self._cache.pop(key)
            elif len(self._cache) >= self.max_size:
                # 移除最老的项
                oldest_key, oldest_value = self._cache.popitem(last=False)
                try:
                    if hasattr(oldest_value, '_mmap'):
                        oldest_value._mmap.close()
                except Exception:
                    pass
            self._cache[key] = value

    def remove(self, key: str) -> None:
        """从缓存中移除内存映射对象
        
        Parameters:
            key (str): 要移除的缓存键值
        """
        with self._lock:
            if key in self._cache:
                value = self._cache.pop(key)
                try:
                    if hasattr(value, '_mmap'):
                        value._mmap.close()
                except Exception:
                    pass

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            for value in self._cache.values():
                try:
                    if hasattr(value, '_mmap'):
                        value._mmap.close()
                except Exception:
                    pass
            self._cache.clear()

    def __del__(self):
        """析构函数，确保所有内存映射对象被正确关闭"""
        self.clear()

# 创建全局缓存实例
_mmap_cache = MmapCache()

_dtype_map = {
    "Bool": np.bool_,
    "Uint8": np.uint8,
    "Uint16": np.uint16,
    "Uint32": np.uint32,
    "Uint64": np.uint64,
    "Int8": np.int8,
    "Int16": np.int16,
    "Int32": np.int32,
    "Int64": np.int64,
    "Float16": np.float16,
    "Float32": np.float32,
    "Float64": np.float64,
}

class _LazyArrayDict:
    def __init__(self, npk: _NumPack, mmap_mode: bool, chunk_size: int = 1024*1024):
        """Initialize LazyArrayDict object
    
        Parameters:
            npk (_NumPack): The NumPack object
            mmap_mode (bool): Whether to use memory mapping mode
            chunk_size (int): Size of chunks for reading large arrays (in bytes)
        """
        self.npk = npk
        self.mmap_mode = mmap_mode
        self.chunk_size = chunk_size

    def keys(self) -> List[str]:
        """Get the list of array names in NumPack file"""
        return self.npk.get_member_list()
    
    def __getitem__(self, key: str) -> np.ndarray:
        """Get the array by key"""
        if key not in self.npk.get_member_list():
            raise KeyError(f"Key {key} not found in NumPack")
        
        if self.mmap_mode:
            # 首先尝试从缓存获取
            cache_key = f"{self.npk.get_array_path(key)}_{key}"
            mmap_array = _mmap_cache.get(cache_key)
            
            if mmap_array is None:
                # 如果缓存中不存在，创建新的内存映射
                mmap_array = np.memmap(
                    self.npk.get_array_path(key),
                    mode='r',
                    shape=self.npk.get_shape(key),
                    dtype=_dtype_map[self.npk.get_metadata()["arrays"][key]["dtype"]]
                )
                # 添加到缓存
                _mmap_cache.put(cache_key, mmap_array)
            
            return mmap_array
        
        return self.npk.load([key])[key]
    
    def __keys__(self) -> List[str]:
        """Get the list of array names in NumPack file"""
        return self.npk.get_member_list()
    
    def __len__(self) -> int:
        """Get the number of arrays in NumPack file"""
        return len(self.npk.get_member_list())
    
    def __contains__(self, key: str) -> bool:
        """Check if the key is in NumPack file"""
        return key in self.npk.get_member_list()

class NumPack:
    def __init__(self, filename: Union[str, Path], drop_if_exists: bool = False):
        """Initialize NumPack object
    
        Parameters:
            filename (Union[str, Path]): The name of the NumPack file
            drop_if_exists (bool): Whether to drop the file if it already exists
        """
        if drop_if_exists and Path(filename).exists() and Path(filename).is_dir():
            shutil.rmtree(filename)

        Path(filename).mkdir(parents=True, exist_ok=True)
        
        self._npk = _NumPack(filename)

    def save(self, arrays: Dict[str, np.ndarray], array_names: Optional[Union[List[str], str]] = None) -> None:
        """Save arrays to NumPack file
    
        Parameters:
            arrays (Dict[str, np.ndarray]): The arrays to save
            array_names (Optional[Union[List[str], str]]): The names of the arrays to save, if None, use the keys of the dictionary
        """
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
            
        self._npk.save(arrays, array_names)

    def load(self, mmap_mode: bool = False) -> Dict[str, np.ndarray]:
        """Load arrays from NumPack file
    
        Parameters:
            mmap_mode (bool): Whether to use memory mapping mode
    
        Returns:
            A LazyArrayDict object, which can load arrays on demand
        """
        return _LazyArrayDict(self._npk, mmap_mode)

    def replace(self, arrays: Dict[str, np.ndarray], indexes: Union[List[int], int, np.ndarray, slice]) -> None:
        """Replace arrays in NumPack file
    
        Parameters:
            arrays (Dict[str, np.ndarray]): The arrays to replace
            indexes (Union[List[int], int, np.ndarray]): The indexes to replace
        """
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
        
        if isinstance(indexes, int):
            indexes = [indexes]
        elif isinstance(indexes, np.ndarray):
            indexes = indexes.tolist()
        elif not isinstance(indexes, (list, slice)):
            raise ValueError("The indexes must be int or list or numpy.ndarray or slice.")
            
        self._npk.replace(arrays, indexes)

    def append(self, arrays: Dict[str, np.ndarray]) -> None:
        """Append arrays to NumPack file
    
        Parameters:
            arrays (Dict[str, np.ndarray]): The arrays to append
        """
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
        
        array_tuples = [(name, array) for name, array in arrays.items()]
        self._npk.append(array_tuples)

    def drop(self, array_name: Union[str, List[str]], indexes: Optional[Union[List[int], int, np.ndarray]] = None) -> None:
        """Drop arrays from NumPack file
    
        Parameters:
            array_name (Union[str, List[str]]): The name or names of the arrays to drop
            indexes (Optional[Union[List[int], int, np.ndarray]]): The indexes to drop, if None, drop all rows
        """
        # 将单个字符串转换为列表
        if isinstance(array_name, str):
            array_name = [array_name]
            
        # 如果使用了内存映射，需要从缓存中移除
        if indexes is None:
            for name in array_name:
                cache_key = f"{self._npk.get_array_path(name)}_{name}"
                _mmap_cache.remove(cache_key)
                self._npk.drop(name, indexes)
        else:
            for name in array_name:
                self._npk.drop(name, indexes)

    def getitem(self, array_name: str, indexes: Union[List[int], int, np.ndarray]) -> np.ndarray:
        """Randomly access the data of specified rows from NumPack file
    
        Parameters:
            array_name (str): The name of the array to access
            indexes (Union[List[int], int, np.ndarray]): The indexes to access, can be integers, lists, slices or numpy arrays

        Returns:
            The specified row data
        """
        if isinstance(indexes, int):
            indexes = [indexes]
        elif isinstance(indexes, np.ndarray):
            indexes = indexes.tolist()
        
        return self._npk.getitem(array_name, indexes)
    
    def get_shape(self, array_names: Optional[Union[List[str], str]] = None) -> Dict[str, Tuple[int, int]]:
        """Get the shape of specified arrays in NumPack file
    
        Parameters:
            array_names (Optional[Union[List[str], str]]): The names of the arrays to get the shape, if None, get the shape of all arrays
    
        Returns:
            A dictionary containing the shapes of the specified arrays
        """
        return self._npk.get_shape(array_names)
    
    def get_member_list(self) -> List[str]:
        """Get the list of array names in NumPack file
    
        Returns:
            A list containing the names of the arrays
        """
        return self._npk.get_member_list()
    
    def get_modify_time(self, array_name: str) -> Optional[int]:
        """Get the modify time of specified array in NumPack file
    
        Parameters:
            array_name (str): The name of the array to get the modify time
    
        Returns:
            The modify time of the array, if the array does not exist, return None
        """
        return self._npk.get_modify_time(array_name)
    
    def reset(self) -> None:
        """Clear all arrays in NumPack file"""
        # 清理所有内存映射缓存
        _mmap_cache.clear()
        self._npk.reset()

    def get_metadata(self) -> Dict[str, Any]:
        """Get the metadata of NumPack file"""
        return self._npk.get_metadata()
    
    def __getitem__(self, key: str) -> np.ndarray:
        """Get the array by key"""
        return self.load([key])[key]
    
    def __iter__(self):
        """Iterate over the arrays in NumPack file"""
        return iter(self.get_member_list())
    
    def stream_load(self, array_name: str, buffer_size: Union[int, None] = None) -> Iterator[np.ndarray]:
        """Stream the array by name with buffering support
        
        Parameters:
            array_name (str): The name of the array to stream
            buffer_size (Union[int, None]): Number of rows to load in each batch, if None, load all rows one by one
            
        Returns:
            Iterator yielding numpy arrays of size up to buffer_size
        """
        if buffer_size <= 0:
            raise ValueError("buffer_size must be greater than 0")
        
        if buffer_size is None:
            for i in range(self.get_shape(array_name)[0]):
                yield self.getitem(i, array_name)
        else:
            total_rows = self.get_shape(array_name)[0]
            for start_idx in range(0, total_rows, buffer_size):
                end_idx = min(start_idx + buffer_size, total_rows)
                yield self.getitem(list(range(start_idx, end_idx)), array_name)

    def chunked_load(self, array_name: str, chunk_rows: int = 100000) -> List[np.memmap]:
        """Load large arrays in chunks
        
        Parameters:
            array_name (str): The name of the array to load
            chunk_rows (int): The number of rows to load in each chunk
            
        Returns:
            List[np.memmap]: A list of np.memmap objects
        """
        memmap_array = self.load(mmap_mode=True)[array_name]
        shape = memmap_array.shape
        
        memmap_chunks = []
        for i in range(0, shape[0], chunk_rows):
            memmap_chunks.append(memmap_array[i:i+chunk_rows])
        
        return memmap_chunks
        