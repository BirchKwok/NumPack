import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union, Optional
import numpy as np

from ._lib_numpack import NumPack as _NumPack, LazyArray


__version__ = "0.2.0"


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
        
        self._npk = _NumPack(str(filename))

    def save(self, arrays: Dict[str, np.ndarray]) -> None:
        """Save arrays to NumPack file
    
        Parameters:
            arrays (Dict[str, np.ndarray]): The arrays to save
        """
        if not isinstance(arrays, dict):
            raise ValueError("arrays must be a dictionary")
            
        self._npk.save(arrays, None)

    def load(self, array_name: str, lazy: bool = False) -> Union[np.ndarray, LazyArray]:
        """Load arrays from NumPack file
    
        Parameters:
            array_name (str): The name of the array to load
            lazy (bool): Whether to load the array in lazy mode (memory mapped)
    
        Returns:
            np.ndarray: The loaded array
        """
        return self._npk.load(array_name, lazy=lazy)

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
        
        self._npk.append(arrays)

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
    
    def get_shape(self, array_name: str) -> Tuple[int, int]:
        """Get the shape of specified arrays in NumPack file
    
        Parameters:
            array_names (str): The name of the array to get the shape
    
        Returns:
            tuple: the shape of the array
        """
        return self._npk.get_shape(array_name)
    
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
        if buffer_size is not None and buffer_size <= 0:
            raise ValueError("buffer_size must be greater than 0")
        
        if buffer_size is None:
            for i in range(self.get_shape(array_name)[0]):
                yield self.getitem(array_name, [i])
        else:
            total_rows = self.get_shape(array_name)[0]
            for start_idx in range(0, total_rows, buffer_size):
                end_idx = min(start_idx + buffer_size, total_rows)
                yield self.getitem(array_name, list(range(start_idx, end_idx)))
    

class LazyArray:
    """
    Lazy Array - Memory-mapped numpy-compatible array with zero-copy operations
    
    LazyArray provides efficient access to large arrays stored on disk through memory mapping.
    It supports most numpy-like operations without loading the entire array into memory.
    
    Features:
    - Zero-copy operations for maximum memory efficiency
    - Memory-mapped file access for large datasets
    - Numpy-compatible interface and dtype system
    - Advanced indexing with boolean masks and fancy indexing
    - Reshape operations without data copying
    - Context manager support for explicit resource management
    
    Note:
        This is a type stub for the actual Rust implementation. The real functionality
        is provided by the compiled Rust extension module.
    """
    
    def __init__(self):
        """
        LazyArray instances are created internally by NumPack.
        Do not instantiate directly.
        """
        raise RuntimeError("LazyArray cannot be instantiated directly. Use NumPack.load() with lazy=True")
    
    # ===========================
    # Core Properties (read-only)
    # ===========================
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of the array as a tuple of integers.
        
        Returns:
            Tuple[int, ...]: Dimensions of the array
            
        Example:
            >>> lazy_arr.shape
            (1000, 256)
        """
        ...
    
    @property
    def dtype(self) -> np.dtype:
        """
        Data type of the array elements.
        
        Returns:
            np.dtype: NumPy data type object
            
        Example:
            >>> lazy_arr.dtype
            dtype('float32')
        """
        ...
    
    @property
    def size(self) -> int:
        """
        Total number of elements in the array.
        
        Returns:
            int: Total number of elements (product of shape dimensions)
            
        Example:
            >>> lazy_arr.size
            256000
        """
        ...
    
    @property
    def ndim(self) -> int:
        """
        Number of dimensions of the array.
        
        Returns:
            int: Number of dimensions
            
        Example:
            >>> lazy_arr.ndim
            2
        """
        ...
    
    @property
    def itemsize(self) -> int:
        """
        Size in bytes of each array element.
        
        Returns:
            int: Size of one element in bytes
            
        Example:
            >>> lazy_arr.itemsize  # float32
            4
        """
        ...
    
    @property
    def nbytes(self) -> int:
        """
        Total number of bytes consumed by the array data.
        
        Returns:
            int: Total bytes (size * itemsize)
            
        Example:
            >>> lazy_arr.nbytes
            1024000
        """
        ...
    
    # ===========================
    # Context Manager Support
    # ===========================
    
    def __enter__(self) -> 'LazyArray':
        """
        Enter context manager.
        
        Enables using LazyArray with 'with' statement for improved resource management.
        
        Returns:
            LazyArray: Self reference
            
        Example:
            >>> with lazy_arr as arr:
            ...     result = arr[10:20]  # Resources properly managed
        """
        ...
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Exit context manager.
        
        Ensures proper cleanup of resources when exiting the 'with' block.
        Particularly important on Windows for ensuring file handles are released.
        
        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
            
        Returns:
            bool: False (exceptions are not suppressed)
        """
        ...
    
    # ===========================
    # Core Array Operations
    # ===========================
    
    def __getitem__(self, key: Union[int, slice, Tuple, List[int], np.ndarray]) -> Union['LazyArray', np.ndarray]:
        """
        Advanced indexing support with numpy-compatible interface.
        
        Supports:
        - Integer indexing: arr[0]
        - Slice indexing: arr[10:20]
        - Tuple indexing: arr[10:20, 5:10]
        - Fancy indexing: arr[[1, 3, 5]]
        - Boolean indexing: arr[mask]
        - Mixed indexing: arr[10:20, [1, 3, 5]]
        
        Args:
            key: Index specification
            
        Returns:
            Union[LazyArray, np.ndarray]: Indexed data
            
        Examples:
            >>> # Single row
            >>> row = lazy_arr[0]
            
            >>> # Slice of rows
            >>> subset = lazy_arr[10:100]
            
            >>> # Fancy indexing
            >>> selected = lazy_arr[[1, 5, 10, 15]]
            
            >>> # Boolean indexing
            >>> mask = np.array([True, False, True, ...])
            >>> filtered = lazy_arr[mask]
            
            >>> # Multi-dimensional indexing
            >>> block = lazy_arr[10:20, 5:15]
        """
        ...
    
    def __len__(self) -> int:
        """
        Length of the first dimension.
        
        Returns:
            int: Size of the first dimension
            
        Example:
            >>> len(lazy_arr)  # shape is (1000, 256)
            1000
        """
        ...
    
    def __repr__(self) -> str:
        """
        String representation of the LazyArray with preview of data.
        
        Returns:
            str: Formatted string representation
        """
        ...
    
    def reshape(self, new_shape: Union[int, Tuple[int, ...], List[int]]) -> 'LazyArray':
        """
        Return a new LazyArray with a different shape (zero-copy view operation).
        
        The reshape operation creates a new view of the same data with a different
        shape. No data is copied, making this operation very efficient for large arrays.
        
        Args:
            new_shape: New shape for the array. Can be:
                - int: Single dimension (e.g., 1000 for 1D array)
                - Tuple[int, ...]: Multi-dimensional (e.g., (100, 10))
                - List[int]: Multi-dimensional as list (e.g., [100, 10])
                
        Returns:
            LazyArray: New LazyArray with the specified shape
            
        Raises:
            ValueError: If the total number of elements doesn't match
            ValueError: If negative dimensions are provided
            ValueError: If invalid shape type is provided
            
        Examples:
            >>> # Original array shape: (1000, 256)
            >>> arr_1d = lazy_arr.reshape(256000)  # Shape: (256000,)
            >>> arr_2d = lazy_arr.reshape((500, 512))  # Shape: (500, 512)
            >>> arr_3d = lazy_arr.reshape([100, 100, 256])  # Shape: (100, 100, 256)
            
            >>> # Chain reshaping operations
            >>> result = lazy_arr.reshape(-1).reshape((2, 128000))
            
        Note:
            - The total number of elements must remain the same
            - This is a view operation - original array is unchanged
            - Multiple reshaped views can coexist safely
            - All views share the same underlying memory-mapped data
        """
        ...
    
    # ===========================
    # Iterator Support
    # ===========================
    
    def __iter__(self):
        """
        Iterate over the first dimension of the array.
        
        Yields:
            np.ndarray: Each row of the array
            
        Example:
            >>> for row in lazy_arr:
            ...     process(row)
        """
        for i in range(len(self)):
            yield self[i]