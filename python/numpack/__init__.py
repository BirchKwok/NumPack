import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union, Optional
import numpy as np
from collections import OrderedDict
import threading
import platform

from numpack._lib_numpack import NumPack as _NumPack

# package message
__version__ = "0.1.0"

if platform.system() == 'Windows':
    DEFAULT_MAX_FILES = 512
    soft_limit = DEFAULT_MAX_FILES
    hard_limit = DEFAULT_MAX_FILES
else:
    import resource
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)

class FileHandlePool:
    """Pool for managing file handles with LRU cache strategy"""
    def __init__(self, max_open_files: int = max(soft_limit // 2, 100)):
        self.max_open_files = max_open_files
        self.open_files = OrderedDict()
        self._lock = threading.Lock()

    def get_file_handle(self, file_path: str, shape: tuple, dtype: np.dtype) -> np.memmap:
        """Get a file handle from the pool or create a new one
        
        Parameters:
            file_path (str): Path to the file
            shape (tuple): Shape of the array
            dtype (np.dtype): Data type of the array
            
        Returns:
            np.memmap: Memory mapped array
        """
        with self._lock:
            if file_path in self.open_files:
                # Move to end (most recently used)
                self.open_files.move_to_end(file_path)
                return self.open_files[file_path]

            # Close oldest file if we've reached the limit
            if len(self.open_files) >= self.max_open_files:
                _, oldest_handle = self.open_files.popitem(last=False)
                try:
                    if hasattr(oldest_handle, '_mmap'):
                        oldest_handle._mmap.close()
                    del oldest_handle
                except Exception:
                    pass

            # Create new memory mapping
            file_handle = np.memmap(file_path, mode='r', dtype=dtype, shape=shape)
            self.open_files[file_path] = file_handle
            return file_handle

    def remove_handle(self, file_path: str) -> None:
        """Remove a file handle from the pool
        
        Parameters:
            file_path (str): Path to the file
        """
        with self._lock:
            if file_path in self.open_files:
                handle = self.open_files.pop(file_path)
                try:
                    if hasattr(handle, '_mmap'):
                        handle._mmap.close()
                    del handle
                except Exception:
                    pass

    def close_all(self) -> None:
        """Close all open file handles"""
        with self._lock:
            for handle in self.open_files.values():
                try:
                    if hasattr(handle, '_mmap'):
                        handle._mmap.close()
                    del handle
                except Exception:
                    pass
            self.open_files.clear()

    def __del__(self):
        """Ensure all handles are closed on deletion"""
        self.close_all()

class MmapCache:
    """LRU cache for memory mapping objects"""
    def __init__(self, max_size: int = max(soft_limit // 2, 100)):
        """Initialize LRU cache
        
        Parameters:
            max_size (int): Maximum cache size, default to half of the system file descriptor limit or 100, whichever is larger
        """
        self.file_pool = FileHandlePool(max_size)
        self._lock = threading.Lock()

    def get(self, key: str, shape: tuple, dtype: np.dtype) -> Optional[np.memmap]:
        """Get the memory mapping object from the cache
        
        Parameters:
            key (str): Cache key
            shape (tuple): Shape of the array
            dtype (np.dtype): Data type of the array
            
        Returns:
            Optional[np.memmap]: Memory mapping object
        """
        return self.file_pool.get_file_handle(key, shape, dtype)

    def remove(self, key: str) -> None:
        """Remove memory mapping object from cache
        
        Parameters:
            key (str): Cache key to remove
        """
        self.file_pool.remove_handle(key)

    def clear(self) -> None:
        """Clear cache"""
        self.file_pool.close_all()

    def __del__(self):
        """Destructor, ensure all memory mapping objects are closed correctly"""
        self.clear()

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
    def __init__(self, npk: _NumPack, mmap_mode: bool, mmap_cache: MmapCache):
        """Initialize LazyArrayDict object
    
        Parameters:
            npk (_NumPack): The NumPack object
            mmap_mode (bool): Whether to use memory mapping mode
            mmap_cache (MmapCache): Cache for memory mapped files
        """
        self.npk = npk
        self.mmap_mode = mmap_mode
        self.mmap_cache = mmap_cache

    def keys(self) -> List[str]:
        """Get the list of array names in NumPack file"""
        return self.npk.get_member_list()
    
    def __getitem__(self, key: str) -> np.ndarray:
        """Get the array by key"""
        if key not in self.npk.get_member_list():
            raise KeyError(f"Key {key} not found in NumPack")
        
        if self.mmap_mode:
            # Get array metadata
            meta = self.npk.get_metadata()["arrays"][key]
            shape = tuple(meta["shape"])
            dtype = _dtype_map[meta["dtype"]]
            
            # Get from cache or create new mapping
            array_path = self.npk.get_array_path(key)
            return self.mmap_cache.get(array_path, shape, dtype)
        
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
        self._mmap_cache = MmapCache()

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
        return _LazyArrayDict(self._npk, mmap_mode, self._mmap_cache)

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
        if isinstance(array_name, str):
            array_name = [array_name]
            
        if indexes is None:
            for name in array_name:
                array_path = self._npk.get_array_path(name)
                self._mmap_cache.remove(array_path)
                self._npk.drop(name, indexes)
        else:
            # 确保在修改文件前关闭所有相关的内存映射
            for name in array_name:
                array_path = self._npk.get_array_path(name)
                self._mmap_cache.remove(array_path)
            # 执行删除操作
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
        
    def __del__(self):
        """Ensure proper cleanup of resources"""
        if hasattr(self, '_mmap_cache'):
            self._mmap_cache.clear()
        