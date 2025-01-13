from typing import List
import numpy as np
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


class MmapMode:
    def __init__(self, npk: _NumPack):
        self.npk = npk
        self.mmap_arrays = {}

    def load(self, array_name: str) -> np.memmap:
        """Load an array into memory mapped mode
        
        Parameters:
            array_name (str): The name of the array to load
            
        Returns:
            np.memmap: A memory mapped array
        """
        if array_name not in self.npk.get_member_list():
            raise KeyError(f"Key {array_name} not found in NumPack")
        
        if array_name in self.mmap_arrays:
            return self.mmap_arrays[array_name]
        
        meta = self.npk.get_metadata()["arrays"][array_name]
        shape = tuple(meta["shape"])
        dtype = _dtype_map[meta["dtype"]]
        
        array_path = self.npk.get_array_path(array_name)
        mmap_array = np.memmap(array_path, mode='r', dtype=dtype, shape=shape)
        self.mmap_arrays[array_name] = mmap_array
        return mmap_array
    
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for array_name in self.mmap_arrays:
            self.mmap_arrays[array_name]._mmap.close()
        self.mmap_arrays = {}
        