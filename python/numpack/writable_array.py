import numpy as np
import mmap
import os


class WritableArray:
    """Writable array wrapper - zero copy solution based on mmap
    
    Core optimization:
    1. Use writable mmap to directly map file
    2. Return NumPy array view, modify directly on the file
    3. Modify automatically sync to file (operating system managed)
    4. Zero memory overhead (just virtual memory mapping)
    """
    
    def __init__(self, file_path, shape, dtype, mode='r+'):
        """
        Args:
            file_path: Data file path
            shape: Array shape
            dtype: Data type
            mode: 'r+' writable, 'r' read only
        """
        self.file_path = file_path
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.mode = mode
        self._mmap = None
        self._array = None
        self._file = None
        
    def __enter__(self):
        """Open file and create mmap"""
        if self.mode == 'r+':
            self._file = open(self.file_path, 'r+b')
        else:
            self._file = open(self.file_path, 'rb')
        
        # Create mmap
        if self.mode == 'r+':
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_WRITE)
        else:
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Create NumPy array view (zero copy)
        self._array = np.ndarray(
            shape=self.shape,
            dtype=self.dtype,
            buffer=self._mmap
        )
        
        return self._array
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close mmap and file"""
        if self._mmap is not None:
            # Critical: Ensure modifications are written to disk
            if self.mode == 'r+':
                self._mmap.flush()
            self._mmap.close()
            self._mmap = None
        
        if self._file is not None:
            self._file.close()
            self._file = None
        
        self._array = None
        return False


class WritableBatchMode:
    """Writable batch mode - zero memory overhead
    
    Strategy:
    1. Use writable mmap to open all array files
    2. Modify directly on the file (zero copy)
    3. The operating system automatically manages dirty page writeback
    4. Exit time unified flush to ensure persistence
    
    Supported data types:
    - Boolean: bool
    - Unsigned integers: uint8, uint16, uint32, uint64
    - Signed integers: int8, int16, int32, int64
    - Floating point: float16, float32, float64
    - Complex numbers: complex64, complex128
    """
    
    def __init__(self, numpack_instance):
        self.npk = numpack_instance
        self.writable_arrays = {}  # array_name -> WritableArray
        self.array_cache = {}  # array_name -> numpy array view
        
    def __enter__(self):
        return self
    
    def load(self, array_name):
        """Load writable array view
        
        Returns:
            numpy array: Array view directly mapped to the file (writable)
        """
        if array_name in self.array_cache:
            return self.array_cache[array_name]
        
        # Get array metadata (using Python API)
        try:
            metadata = self.npk.get_metadata()
            if array_name not in metadata['arrays']:
                raise KeyError(f"Array '{array_name}' not found")
            
            array_meta = metadata['arrays'][array_name]
            shape = array_meta['shape']
            dtype_str = array_meta['dtype']  # Format like "Bool", "Uint8", "Float32", etc.
        except Exception as e:
            raise KeyError(f"Array '{array_name}' not found: {e}")
        
        # Build file path
        file_path = os.path.join(str(self.npk._filename), f"data_{array_name}.npkd")
        
        # Map Rust DataType format to NumPy dtype
        # Supports all NumPack data types from specification
        dtype_map = {
            'Bool': np.bool_,
            'Uint8': np.uint8,
            'Uint16': np.uint16,
            'Uint32': np.uint32,
            'Uint64': np.uint64,
            'Int8': np.int8,
            'Int16': np.int16,
            'Int32': np.int32,
            'Int64': np.int64,
            'Float16': np.float16,
            'Float32': np.float32,
            'Float64': np.float64,
            'Complex64': np.complex64,
            'Complex128': np.complex128,
        }
        
        dtype = dtype_map.get(dtype_str)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        
        # Open file and create mmap
        file = open(file_path, 'r+b')
        mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE)
        
        # Create NumPy array view
        arr = np.ndarray(
            shape=tuple(shape),
            dtype=dtype,
            buffer=mm
        )
        
        # Save reference
        self.writable_arrays[array_name] = (file, mm)
        self.array_cache[array_name] = arr
        
        return arr
    
    def save(self, arrays_dict):
        """Save operation becomes no operation
        
        Because modifications are already directly on the file, no additional save is needed
        """
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close all mmap and flush"""
        for array_name, (file, mm) in self.writable_arrays.items():
            try:
                mm.flush()  # Ensure write to disk
                mm.close()
                file.close()
            except Exception as e:
                print(f"Warning: Failed to close {array_name}: {e}")
        
        self.writable_arrays.clear()
        self.array_cache.clear()
        return False

