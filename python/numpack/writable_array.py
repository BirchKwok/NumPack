import numpy as np
import mmap
import os


class WritableArray:
    """Writable NumPy array view backed by a memory-mapped file.

    This is a zero-copy wrapper around `mmap` that exposes an `ndarray` view of
    the underlying file. When opened in write mode, in-place modifications to
    the returned array are written back to disk via OS-managed dirty pages.

    Notes
    -----
    - This class maps the *entire* file.
    - The returned array does not own its data (``arr.flags['OWNDATA'] == False``).
    """

    def __init__(self, file_path, shape, dtype, mode='r+'):
        """Create a memory-mapped array view.

        Parameters
        ----------
        file_path : str
            Path to the underlying data file.
        shape : tuple of int
            Array shape.
        dtype : numpy.dtype or str
            NumPy dtype (or dtype string) used to interpret the file buffer.
        mode : {"r+", "r"}, optional
            File open mode. Use ``"r+"`` for a writable view or ``"r"`` for a
            read-only view.
        """
        self.file_path = file_path
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.mode = mode
        self._mmap = None
        self._array = None
        self._file = None

    def __enter__(self):
        """Open the file and return an `ndarray` view."""
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
        """Flush (if writable) and close the mapping and file."""
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
    """Writable batch mode backed by memory mapping.

    This context manager opens arrays as writable `mmap` views and returns
    `numpy.ndarray` objects that directly reference on-disk buffers.

    Notes
    -----
    - This mode is intended for *in-place* value updates.
    - Array shape changes (e.g., append/reshape) are not supported because the
      underlying file size/layout would change.
    """

    def __init__(self, numpack_instance):
        self.npk = numpack_instance
        self.writable_arrays = {}  # array_name -> WritableArray
        self.array_cache = {}  # array_name -> numpy array view

    def __enter__(self):
        """Enter the writable batch context."""
        return self

    def load(self, array_name):
        """Load a writable array view.

        Parameters
        ----------
        array_name : str
            Name of the array stored in the NumPack file.

        Returns
        -------
        numpy.ndarray
            A writable `ndarray` view directly backed by the array data file.

        Raises
        ------
        KeyError
            If `array_name` does not exist.
        ValueError
            If the stored dtype cannot be mapped to a NumPy dtype.
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
        """No-op save.

        In writable batch mode, modifications are applied directly to the
        memory-mapped files. This method exists for API symmetry.
        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Flush and close all memory maps and file handles."""
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

