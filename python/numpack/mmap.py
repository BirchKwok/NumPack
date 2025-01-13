from typing import List
import numpy as np
from collections import OrderedDict
import threading
import platform

from ._lib_numpack import NumPack as _NumPack


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


class MmapMode:
    def __init__(self, npk: _NumPack, file_pool: FileHandlePool):
        self.npk = npk
        self.file_pool = file_pool
        self.array_names = []

    def load(self, array_name: str) -> np.memmap:
        """Load an array into memory mapped mode
        
        Parameters:
            array_name (str): The name of the array to load
            
        Returns:
            np.memmap: A memory mapped array
        """
        if array_name not in self.npk.get_member_list():
            raise KeyError(f"Key {array_name} not found in NumPack")
        
        meta = self.npk.get_metadata()["arrays"][array_name]
        shape = tuple(meta["shape"])
        dtype = _dtype_map[meta["dtype"]]
        
        array_path = self.npk.get_array_path(array_name)
        self.array_names.append(array_name)
        return self.file_pool.get_file_handle(array_path, shape, dtype)
    
    def chunked_load(self, array_name: str, chunk_rows: int = 100000) -> List[np.memmap]:
        """Load large arrays in chunks
        
        Parameters:
            array_name (str): The name of the array to load
            chunk_rows (int): The number of rows to load in each chunk
            
        Returns:
            List[np.memmap]: A list of np.memmap objects
        """
        memmap_array = self.load(array_name)
        shape = memmap_array.shape
        
        memmap_chunks = []
        for i in range(0, shape[0], chunk_rows):
            memmap_chunks.append(memmap_array[i:i+chunk_rows])
        
        return memmap_chunks

    def close(self):
        for array_name in self.array_names:
            self.file_pool.remove_handle(self.npk.get_array_path(array_name))
        self.array_names = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        