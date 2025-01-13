import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union, Optional
import numpy as np

from ._lib_numpack import NumPack as _NumPack
from .mmap import MmapMode


__version__ = "0.1.1"


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

    def load(self, array_name: str) -> np.ndarray:
        """Load arrays from NumPack file
    
        Parameters:
            array_name (str): The name of the array to load
    
        Returns:
            np.ndarray: The loaded array
        """
        return self._npk.load([array_name])[array_name]

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
            
        self._npk.drop(array_name, indexes)

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
    
    def mmap_mode(self) -> MmapMode:
        """Get a memory mapped reader for the NumPack file"""
        return MmapMode(self._npk)
    
    def reset(self) -> None:
        """Clear all arrays in NumPack file"""
        self._npk.reset()

    def get_metadata(self) -> Dict[str, Any]:
        """Get the metadata of NumPack file"""
        return self._npk.get_metadata()
    
    def __getitem__(self, key: str) -> np.ndarray:
        """Get the array by key"""
        return self.load(key)
    
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
    