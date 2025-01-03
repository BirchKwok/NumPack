from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import numpy as np

from numpack._lib_numpack import NumPack as _NumPack


class _LazyArrayDict:
    def __init__(self, npk: _NumPack, mmap_mode: bool):
        """Initialize LazyArrayDict object
    
        Parameters:
            npk (NumPack): The NumPack object
            mmap_mode (bool): Whether to use memory mapping mode
        """
        self.npk = npk
        self.mmap_mode = mmap_mode

    def keys(self) -> List[str]:
        """Get the list of array names in NumPack file"""
        return self.npk.get_member_list()
    
    def __getitem__(self, key: str) -> np.ndarray:
        """Get the array by key"""
        if key not in self.npk.get_member_list():
            raise KeyError(f"Key {key} not found in NumPack")
        
        return self.npk.load([key], self.mmap_mode)[key]
    
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
            Path(filename).unlink()

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

    def replace(self, arrays: Dict[str, np.ndarray], indexes: Union[List[int], int, np.ndarray]) -> None:
        """Replace arrays in NumPack file
    
        Parameters:
            arrays (Dict[str, np.ndarray]): The arrays to replace
            indexes (Union[List[int], int, np.ndarray]): The indexes to replace
        """
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
            
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

    def drop(self, array_names: Optional[Union[List[str], str]] = None, indexes: Optional[Union[List[int], int, np.ndarray]] = None) -> None:
        """Drop arrays from NumPack file
    
        Parameters:
            array_names (Optional[Union[List[str], str]]): The names of the arrays to drop, if None, drop all arrays
            indexes (Optional[Union[List[int], int, np.ndarray]]): The indexes to drop, if None, drop all rows
        """
        if array_names is not None and isinstance(array_names, str):
            array_names = [array_names]
            
        self._npk.drop(array_names, indexes)


    def getitem(self, indexes: Union[List[int], int, np.ndarray], array_names: Optional[Union[List[str], str]] = None) -> Dict[str, np.ndarray]:
        """Randomly access the data of specified rows from NumPack file
    
        Parameters:
            indexes (Union[List[int], int, np.ndarray]): The indexes to access, can be integers, lists, slices or numpy arrays
            array_names (Optional[Union[List[str], str]]): The names of the arrays to access, if None, access all arrays
    
        Returns:
            A dictionary containing the specified row data
        """
        raise NotImplementedError("getitem is not implemented")
    
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
        self._npk.reset()
    
__version__ = "0.1.0"
