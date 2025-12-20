"""Zero-copy conversion utilities for NumPack.

This module provides zero-copy or near-zero-copy conversion between NumPack
and common data frameworks (PyArrow, PyTorch, JAX, CuPy) using:

1. PyArrow zero-copy buffer sharing
2. DLPack protocol for tensor frameworks
3. NumPy buffer protocol

Notes
-----
Zero-copy conversions require contiguous memory layouts and compatible dtypes.
When zero-copy is not possible, a warning is emitted and a copy is made.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    import pyarrow as pa
    import torch


# =============================================================================
# DLPack Protocol Support
# =============================================================================

class DLPackBuffer:
    """A wrapper providing DLPack protocol for NumPack arrays.
    
    This enables zero-copy tensor exchange with PyTorch, JAX, CuPy, TensorFlow,
    and other frameworks that support DLPack.
    
    Examples
    --------
    >>> from numpack.io.zero_copy import DLPackBuffer
    >>> buffer = DLPackBuffer(numpy_array)
    >>> torch_tensor = torch.from_dlpack(buffer)  # Zero-copy!
    >>> jax_array = jax.dlpack.from_dlpack(buffer)  # Zero-copy!
    """
    
    # DLPack device type constants
    kDLCPU = 1
    kDLCUDA = 2
    kDLCUDAHost = 3
    kDLOpenCL = 4
    kDLVulkan = 7
    kDLMetal = 8
    kDLVPI = 9
    kDLROCM = 10
    kDLROCMHost = 11
    kDLExtDev = 12
    kDLCUDAManaged = 13
    kDLOneAPI = 14
    kDLWebGPU = 15
    kDLHexagon = 16
    
    # DLPack dtype constants
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLBfloat = 4
    kDLComplex = 5
    kDLBool = 6
    
    def __init__(
        self, 
        array: np.ndarray, 
        stream: Optional[int] = None,
        max_version: Optional[Tuple[int, int]] = None
    ):
        """Create a DLPack buffer from a NumPy array.
        
        Parameters
        ----------
        array : numpy.ndarray
            Source array. Must be contiguous in memory.
        stream : int, optional
            CUDA stream for GPU tensors. Ignored for CPU arrays.
        max_version : tuple of (int, int), optional
            Maximum DLPack version supported by the consumer.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(array)}")
        
        # Ensure contiguous layout
        if not array.flags['C_CONTIGUOUS']:
            warnings.warn(
                "Array is not C-contiguous. Creating a contiguous copy.",
                UserWarning
            )
            array = np.ascontiguousarray(array)
        
        self._array = array
        self._stream = stream
        self._max_version = max_version or (1, 0)
    
    def __dlpack__(self, *, stream: Optional[int] = None) -> Any:
        """Export the array via DLPack.
        
        Parameters
        ----------
        stream : int, optional
            CUDA stream for synchronization.
        
        Returns
        -------
        PyCapsule
            DLPack capsule containing the tensor.
        """
        # Use NumPy's built-in DLPack support (NumPy >= 1.22)
        if hasattr(self._array, '__dlpack__'):
            return self._array.__dlpack__(stream=stream)
        
        # Fallback for older NumPy versions
        return self._create_dlpack_capsule()
    
    def __dlpack_device__(self) -> Tuple[int, int]:
        """Return the device type and device ID.
        
        Returns
        -------
        tuple of (int, int)
            (device_type, device_id). CPU is (1, 0).
        """
        if hasattr(self._array, '__dlpack_device__'):
            return self._array.__dlpack_device__()
        return (self.kDLCPU, 0)  # CPU device
    
    def _create_dlpack_capsule(self) -> Any:
        """Create DLPack capsule for older NumPy versions."""
        # This is a simplified implementation for older NumPy
        # Modern NumPy (>=1.22) has native __dlpack__ support
        try:
            # Try to use NumPy's internal DLPack support
            from numpy.core._multiarray_umath import _dlpack
            return _dlpack.to_dlpack(self._array)
        except (ImportError, AttributeError):
            raise NotImplementedError(
                "DLPack export requires NumPy >= 1.22. "
                "Please upgrade NumPy: pip install numpy>=1.22"
            )
    
    @property
    def array(self) -> np.ndarray:
        """Return the underlying NumPy array."""
        return self._array
    
    @classmethod
    def from_dlpack(cls, capsule: Any) -> np.ndarray:
        """Import an array from a DLPack capsule or DLPack-compatible object.
        
        Parameters
        ----------
        capsule : PyCapsule or object with __dlpack__
            DLPack capsule or DLPack-compatible object from another framework.
        
        Returns
        -------
        numpy.ndarray
            Array created from the capsule (zero-copy if possible).
        """
        # Use NumPy's from_dlpack if available (NumPy >= 1.22)
        if hasattr(np, 'from_dlpack'):
            return np.from_dlpack(capsule)
        
        raise NotImplementedError(
            "DLPack import requires NumPy >= 1.22. "
            "Please upgrade NumPy: pip install numpy>=1.22"
        )


def to_dlpack(array: np.ndarray) -> DLPackBuffer:
    """Wrap a NumPy array for DLPack export.
    
    Parameters
    ----------
    array : numpy.ndarray
        Source array.
    
    Returns
    -------
    DLPackBuffer
        DLPack-compatible wrapper.
    
    Examples
    --------
    >>> import torch
    >>> from numpack.io.zero_copy import to_dlpack
    >>> arr = np.array([1, 2, 3], dtype=np.float32)
    >>> tensor = torch.from_dlpack(to_dlpack(arr))
    """
    return DLPackBuffer(array)


def from_dlpack(obj: Any) -> np.ndarray:
    """Import an array from DLPack.
    
    Parameters
    ----------
    obj : PyCapsule or object with __dlpack__
        DLPack capsule or DLPack-compatible object.
    
    Returns
    -------
    numpy.ndarray
        Array created from the object.
    
    Examples
    --------
    >>> import torch
    >>> from numpack.io.zero_copy import from_dlpack
    >>> tensor = torch.tensor([1, 2, 3])
    >>> arr = from_dlpack(tensor)
    """
    # Use NumPy's from_dlpack directly - it handles both objects with __dlpack__
    # and raw PyCapsules (in supported NumPy versions)
    if hasattr(np, 'from_dlpack'):
        return np.from_dlpack(obj)
    
    raise NotImplementedError(
        "DLPack import requires NumPy >= 1.22. "
        "Please upgrade NumPy: pip install numpy>=1.22"
    )


# =============================================================================
# PyArrow Zero-Copy Buffer Sharing
# =============================================================================

def numpy_to_arrow_zero_copy(
    array: np.ndarray,
    name: str = "data"
) -> "pa.Array":
    """Convert a NumPy array to PyArrow with zero-copy when possible.
    
    Parameters
    ----------
    array : numpy.ndarray
        Source array. Must be 1D and contiguous.
    name : str, optional
        Column name for the result.
    
    Returns
    -------
    pyarrow.Array
        Arrow array sharing memory with the input.
    
    Notes
    -----
    Zero-copy is only possible for:
    - 1D contiguous arrays
    - Numeric dtypes (not object or string)
    """
    try:
        import pyarrow as pa
    except ImportError:
        raise ImportError(
            "PyArrow is required for Arrow buffer sharing.\n"
            "Install it with: pip install pyarrow"
        )
    
    # Check if zero-copy is possible
    can_zero_copy = (
        array.ndim == 1 and
        array.flags['C_CONTIGUOUS'] and
        array.dtype.kind in ('i', 'u', 'f', 'b')  # int, uint, float, bool
    )
    
    if can_zero_copy:
        # Use Arrow's zero-copy buffer wrapping
        buf = pa.py_buffer(array)
        arrow_type = pa.from_numpy_dtype(array.dtype)
        return pa.Array.from_buffers(arrow_type, len(array), [None, buf])
    else:
        # Fallback to standard conversion (may copy)
        return pa.array(array)


def arrow_to_numpy_zero_copy(
    arrow_array: "pa.Array",
    writable: bool = False
) -> np.ndarray:
    """Convert a PyArrow array to NumPy with zero-copy when possible.
    
    Parameters
    ----------
    arrow_array : pyarrow.Array
        Source Arrow array.
    writable : bool, optional
        If True, make the result writable (forces a copy).
    
    Returns
    -------
    numpy.ndarray
        NumPy array, zero-copy if possible.
    
    Notes
    -----
    Zero-copy fails if:
    - The array contains nulls
    - The dtype is not directly mappable
    - writable=True is requested
    """
    try:
        import pyarrow as pa
    except ImportError:
        raise ImportError(
            "PyArrow is required for Arrow buffer sharing.\n"
            "Install it with: pip install pyarrow"
        )
    
    if writable:
        # Writable arrays cannot be zero-copy (Arrow buffers are immutable)
        return arrow_array.to_numpy(zero_copy_only=False).copy()
    
    try:
        # Attempt zero-copy conversion
        return arrow_array.to_numpy(zero_copy_only=True)
    except pa.ArrowInvalid:
        # Zero-copy not possible, fall back to copy
        warnings.warn(
            "Zero-copy conversion from Arrow failed (nulls or incompatible type). "
            "Creating a copy.",
            UserWarning
        )
        return arrow_array.to_numpy(zero_copy_only=False)


def table_to_numpy_zero_copy(
    table: "pa.Table",
    column: Optional[str] = None
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Convert PyArrow Table columns to NumPy with zero-copy.
    
    Parameters
    ----------
    table : pyarrow.Table
        Source Arrow table.
    column : str, optional
        If provided, convert only this column. Otherwise, convert all columns.
    
    Returns
    -------
    numpy.ndarray or dict
        If `column` is provided, returns the array. Otherwise, returns a dict
        mapping column names to arrays.
    """
    if column is not None:
        return arrow_to_numpy_zero_copy(table.column(column).combine_chunks())
    
    result = {}
    for col_name in table.column_names:
        arr = table.column(col_name).combine_chunks()
        result[col_name] = arrow_to_numpy_zero_copy(arr)
    
    return result


# =============================================================================
# PyTorch Zero-Copy Integration
# =============================================================================

def numpy_to_torch_zero_copy(
    array: np.ndarray,
    device: Optional[str] = None
) -> "torch.Tensor":
    """Convert NumPy array to PyTorch tensor with zero-copy.
    
    Parameters
    ----------
    array : numpy.ndarray
        Source array.
    device : str, optional
        Target device. If 'cuda' or GPU device, data will be copied to GPU.
    
    Returns
    -------
    torch.Tensor
        Tensor sharing memory with the input (if CPU).
    
    Notes
    -----
    Zero-copy is only possible for CPU tensors. GPU transfers always copy.
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for tensor conversion.\n"
            "Install it with: pip install torch"
        )
    
    # Ensure contiguous
    if not array.flags['C_CONTIGUOUS']:
        warnings.warn(
            "Array is not contiguous. Creating a contiguous copy first.",
            UserWarning
        )
        array = np.ascontiguousarray(array)
    
    # Use torch.from_numpy for zero-copy CPU tensors
    tensor = torch.from_numpy(array)
    
    # Move to device if requested
    if device is not None and device != 'cpu':
        tensor = tensor.to(device)  # This copies to GPU
    
    return tensor


def torch_to_numpy_zero_copy(tensor: "torch.Tensor") -> np.ndarray:
    """Convert PyTorch tensor to NumPy array with zero-copy.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Source tensor. Must be on CPU.
    
    Returns
    -------
    numpy.ndarray
        Array sharing memory with the input (if contiguous CPU tensor).
    
    Notes
    -----
    GPU tensors are automatically moved to CPU first (requires a copy).
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for tensor conversion.\n"
            "Install it with: pip install torch"
        )
    
    # Move to CPU if on GPU
    if tensor.device.type != 'cpu':
        warnings.warn(
            f"Tensor is on {tensor.device}. Moving to CPU (requires copy).",
            UserWarning
        )
        tensor = tensor.cpu()
    
    # Detach from computation graph
    tensor = tensor.detach()
    
    # Convert to NumPy (zero-copy for contiguous tensors)
    return tensor.numpy()


# =============================================================================
# NumPack Integration Helpers
# =============================================================================

class ZeroCopyArray:
    """A wrapper that provides zero-copy interop for NumPack arrays.
    
    This class wraps a NumPy array and provides efficient zero-copy
    conversions to various frameworks via DLPack and buffer protocols.
    
    Examples
    --------
    >>> from numpack.io.zero_copy import ZeroCopyArray
    >>> arr = ZeroCopyArray(numpy_array)
    >>> tensor = arr.to_torch()  # Zero-copy!
    >>> arrow = arr.to_arrow()   # Zero-copy!
    """
    
    def __init__(self, array: np.ndarray):
        """Initialize with a NumPy array.
        
        Parameters
        ----------
        array : numpy.ndarray
            The underlying array. Will be made contiguous if needed.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(array)}")
        
        # Ensure contiguous
        if not array.flags['C_CONTIGUOUS']:
            warnings.warn(
                "Array is not C-contiguous. Creating a contiguous copy.",
                UserWarning
            )
            array = np.ascontiguousarray(array)
        
        self._array = array
    
    @property
    def array(self) -> np.ndarray:
        """Return the underlying NumPy array."""
        return self._array
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the array shape."""
        return self._array.shape
    
    @property
    def dtype(self):
        """Return the array dtype."""
        return self._array.dtype
    
    @property
    def nbytes(self) -> int:
        """Return the total bytes."""
        return self._array.nbytes
    
    def __dlpack__(self, *, stream: Optional[int] = None) -> Any:
        """DLPack protocol export."""
        if hasattr(self._array, '__dlpack__'):
            return self._array.__dlpack__(stream=stream)
        return DLPackBuffer(self._array).__dlpack__(stream=stream)
    
    def __dlpack_device__(self) -> Tuple[int, int]:
        """DLPack device info."""
        if hasattr(self._array, '__dlpack_device__'):
            return self._array.__dlpack_device__()
        return (DLPackBuffer.kDLCPU, 0)
    
    def __array__(self, dtype=None) -> np.ndarray:
        """NumPy array protocol."""
        if dtype is None:
            return self._array
        return self._array.astype(dtype)
    
    def to_torch(self, device: Optional[str] = None) -> "torch.Tensor":
        """Convert to PyTorch tensor (zero-copy for CPU)."""
        return numpy_to_torch_zero_copy(self._array, device)
    
    def to_arrow(self, name: str = "data") -> "pa.Array":
        """Convert to PyArrow array (zero-copy for 1D numeric)."""
        return numpy_to_arrow_zero_copy(self._array.ravel(), name)
    
    def to_dlpack(self) -> DLPackBuffer:
        """Get DLPack wrapper."""
        return DLPackBuffer(self._array)
    
    @classmethod
    def from_torch(cls, tensor: "torch.Tensor") -> "ZeroCopyArray":
        """Create from PyTorch tensor (zero-copy for CPU)."""
        return cls(torch_to_numpy_zero_copy(tensor))
    
    @classmethod
    def from_arrow(cls, arrow_array: "pa.Array") -> "ZeroCopyArray":
        """Create from PyArrow array (zero-copy when possible)."""
        return cls(arrow_to_numpy_zero_copy(arrow_array))
    
    @classmethod
    def from_dlpack(cls, capsule: Any) -> "ZeroCopyArray":
        """Create from DLPack capsule."""
        return cls(from_dlpack(capsule))


def wrap_for_zero_copy(array: np.ndarray) -> ZeroCopyArray:
    """Wrap a NumPy array for zero-copy interop.
    
    Parameters
    ----------
    array : numpy.ndarray
        Source array.
    
    Returns
    -------
    ZeroCopyArray
        Wrapper with zero-copy conversion methods.
    """
    return ZeroCopyArray(array)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # DLPack
    'DLPackBuffer',
    'to_dlpack',
    'from_dlpack',
    # PyArrow
    'numpy_to_arrow_zero_copy',
    'arrow_to_numpy_zero_copy',
    'table_to_numpy_zero_copy',
    # PyTorch
    'numpy_to_torch_zero_copy',
    'torch_to_numpy_zero_copy',
    # NumPack integration
    'ZeroCopyArray',
    'wrap_for_zero_copy',
]
