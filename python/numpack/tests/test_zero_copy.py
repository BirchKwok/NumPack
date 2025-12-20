"""Tests for zero-copy conversion utilities.

This module tests the zero-copy interoperability between NumPack and
various data frameworks (PyArrow, PyTorch, DLPack).
"""

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def sample_arrays():
    """Create sample arrays for testing."""
    return {
        'float32': np.random.randn(1000, 128).astype(np.float32),
        'float64': np.random.randn(500, 64).astype(np.float64),
        'int32': np.random.randint(0, 1000, (800, 32)).astype(np.int32),
        'int64': np.random.randint(0, 10000, (600, 16)).astype(np.int64),
        'bool': np.random.choice([True, False], size=(400, 8)),
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path, ignore_errors=True)


# =============================================================================
# DLPack Tests
# =============================================================================

class TestDLPack:
    """Tests for DLPack protocol support."""
    
    def test_dlpack_buffer_creation(self, sample_arrays):
        """Test DLPackBuffer can be created from numpy arrays."""
        from numpack.io.zero_copy import DLPackBuffer
        
        for name, arr in sample_arrays.items():
            buffer = DLPackBuffer(arr)
            assert buffer.array is arr or np.array_equal(buffer.array, arr)
    
    def test_dlpack_device(self, sample_arrays):
        """Test DLPack device info returns CPU."""
        from numpack.io.zero_copy import DLPackBuffer
        
        arr = sample_arrays['float32']
        buffer = DLPackBuffer(arr)
        device_type, device_id = buffer.__dlpack_device__()
        
        assert device_type == DLPackBuffer.kDLCPU  # CPU device
        assert device_id == 0
    
    def test_to_dlpack(self, sample_arrays):
        """Test to_dlpack wrapper function."""
        from numpack.io.zero_copy import to_dlpack
        
        arr = sample_arrays['float64']
        buffer = to_dlpack(arr)
        
        assert hasattr(buffer, '__dlpack__')
        assert hasattr(buffer, '__dlpack_device__')
    
    @pytest.mark.skipif(
        not hasattr(np, 'from_dlpack'),
        reason="NumPy >= 1.22 required for DLPack"
    )
    def test_dlpack_roundtrip(self, sample_arrays):
        """Test DLPack export and import roundtrip."""
        from numpack.io.zero_copy import DLPackBuffer, from_dlpack
        
        arr = sample_arrays['float32']
        buffer = DLPackBuffer(arr)
        
        # Import back using the buffer object (which has __dlpack__)
        # np.from_dlpack expects an object with __dlpack__ method
        result = from_dlpack(buffer)
        
        np.testing.assert_array_equal(result, arr)
    
    def test_non_contiguous_array_warning(self):
        """Test that non-contiguous arrays trigger a warning."""
        from numpack.io.zero_copy import DLPackBuffer
        
        # Create a non-contiguous array (transposed view)
        arr = np.random.randn(10, 20).T
        assert not arr.flags['C_CONTIGUOUS']
        
        with pytest.warns(UserWarning, match="not C-contiguous"):
            buffer = DLPackBuffer(arr)
        
        # The buffer should have a contiguous copy
        assert buffer.array.flags['C_CONTIGUOUS']


# =============================================================================
# PyArrow Zero-Copy Tests
# =============================================================================

class TestPyArrowZeroCopy:
    """Tests for PyArrow zero-copy conversions."""
    
    @pytest.fixture
    def pyarrow(self):
        """Import pyarrow or skip if not available."""
        pytest.importorskip('pyarrow')
        import pyarrow as pa
        return pa
    
    def test_numpy_to_arrow_zero_copy(self, pyarrow, sample_arrays):
        """Test zero-copy conversion from numpy to arrow."""
        from numpack.io.zero_copy import numpy_to_arrow_zero_copy
        
        arr = sample_arrays['float32'].ravel()  # Must be 1D for zero-copy
        arrow_arr = numpy_to_arrow_zero_copy(arr)
        
        assert len(arrow_arr) == len(arr)
        np.testing.assert_array_almost_equal(arrow_arr.to_numpy(), arr)
    
    def test_arrow_to_numpy_zero_copy(self, pyarrow, sample_arrays):
        """Test zero-copy conversion from arrow to numpy."""
        from numpack.io.zero_copy import arrow_to_numpy_zero_copy
        import pyarrow as pa
        
        arr = sample_arrays['int64'].ravel()
        arrow_arr = pa.array(arr)
        
        result = arrow_to_numpy_zero_copy(arrow_arr)
        
        np.testing.assert_array_equal(result, arr)
    
    def test_arrow_to_numpy_with_nulls(self, pyarrow):
        """Test that arrays with nulls fall back to copy."""
        from numpack.io.zero_copy import arrow_to_numpy_zero_copy
        import pyarrow as pa
        
        # Create array with nulls
        arr = pa.array([1, 2, None, 4, 5])
        
        with pytest.warns(UserWarning, match="Zero-copy conversion from Arrow failed"):
            result = arrow_to_numpy_zero_copy(arr)
        
        # Result should still be valid (with NaN for nulls)
        assert len(result) == 5
    
    def test_table_to_numpy_zero_copy(self, pyarrow):
        """Test converting a PyArrow table to numpy."""
        from numpack.io.zero_copy import table_to_numpy_zero_copy
        import pyarrow as pa
        
        # Create a simple table
        table = pa.table({
            'col1': [1, 2, 3, 4, 5],
            'col2': [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        
        result = table_to_numpy_zero_copy(table)
        
        assert isinstance(result, dict)
        assert 'col1' in result
        assert 'col2' in result
        assert len(result['col1']) == 5


# =============================================================================
# PyTorch Zero-Copy Tests
# =============================================================================

class TestPyTorchZeroCopy:
    """Tests for PyTorch zero-copy conversions."""
    
    @pytest.fixture
    def torch(self):
        """Import torch or skip if not available."""
        pytest.importorskip('torch')
        import torch
        return torch
    
    def test_numpy_to_torch_zero_copy(self, torch, sample_arrays):
        """Test zero-copy conversion from numpy to torch."""
        from numpack.io.zero_copy import numpy_to_torch_zero_copy
        
        arr = sample_arrays['float32']
        tensor = numpy_to_torch_zero_copy(arr)
        
        assert tensor.shape == arr.shape
        np.testing.assert_array_almost_equal(tensor.numpy(), arr)
    
    def test_numpy_to_torch_shares_memory(self, torch, sample_arrays):
        """Test that numpy_to_torch_zero_copy shares memory."""
        from numpack.io.zero_copy import numpy_to_torch_zero_copy
        
        arr = np.ascontiguousarray(sample_arrays['float64'])
        tensor = numpy_to_torch_zero_copy(arr)
        
        # Modify the tensor
        tensor[0, 0] = 999.0
        
        # Original array should be modified too (shared memory)
        assert arr[0, 0] == 999.0
    
    def test_torch_to_numpy_zero_copy(self, torch, sample_arrays):
        """Test zero-copy conversion from torch to numpy."""
        from numpack.io.zero_copy import torch_to_numpy_zero_copy
        import torch as th
        
        arr = sample_arrays['int32']
        tensor = th.from_numpy(arr)
        
        result = torch_to_numpy_zero_copy(tensor)
        
        np.testing.assert_array_equal(result, arr)
    
    def test_torch_to_numpy_shares_memory(self, torch):
        """Test that torch_to_numpy_zero_copy shares memory for CPU tensors."""
        from numpack.io.zero_copy import torch_to_numpy_zero_copy
        import torch as th
        
        tensor = th.randn(10, 10)
        arr = torch_to_numpy_zero_copy(tensor)
        
        # Modify the array
        arr[0, 0] = 999.0
        
        # Original tensor should be modified too (shared memory)
        assert tensor[0, 0].item() == 999.0


# =============================================================================
# ZeroCopyArray Tests
# =============================================================================

class TestZeroCopyArray:
    """Tests for the ZeroCopyArray wrapper class."""
    
    def test_creation(self, sample_arrays):
        """Test ZeroCopyArray creation."""
        from numpack.io.zero_copy import ZeroCopyArray
        
        arr = sample_arrays['float32']
        wrapper = ZeroCopyArray(arr)
        
        assert wrapper.shape == arr.shape
        assert wrapper.dtype == arr.dtype
        assert wrapper.nbytes == arr.nbytes
    
    def test_array_protocol(self, sample_arrays):
        """Test numpy array protocol."""
        from numpack.io.zero_copy import ZeroCopyArray
        
        arr = sample_arrays['float64']
        wrapper = ZeroCopyArray(arr)
        
        # Should be convertible to numpy array
        result = np.asarray(wrapper)
        np.testing.assert_array_equal(result, arr)
    
    def test_dlpack_protocol(self, sample_arrays):
        """Test DLPack protocol on ZeroCopyArray."""
        from numpack.io.zero_copy import ZeroCopyArray
        
        arr = sample_arrays['int32']
        wrapper = ZeroCopyArray(arr)
        
        assert hasattr(wrapper, '__dlpack__')
        assert hasattr(wrapper, '__dlpack_device__')
        
        device_type, device_id = wrapper.__dlpack_device__()
        assert device_type == 1  # CPU
    
    @pytest.fixture
    def torch(self):
        pytest.importorskip('torch')
        import torch
        return torch
    
    def test_to_torch(self, torch, sample_arrays):
        """Test ZeroCopyArray.to_torch()."""
        from numpack.io.zero_copy import ZeroCopyArray
        
        arr = sample_arrays['float32']
        wrapper = ZeroCopyArray(arr)
        
        tensor = wrapper.to_torch()
        
        np.testing.assert_array_almost_equal(tensor.numpy(), arr)
    
    @pytest.fixture
    def pyarrow(self):
        pytest.importorskip('pyarrow')
        import pyarrow as pa
        return pa
    
    def test_to_arrow(self, pyarrow, sample_arrays):
        """Test ZeroCopyArray.to_arrow()."""
        from numpack.io.zero_copy import ZeroCopyArray
        
        arr = sample_arrays['float64']
        wrapper = ZeroCopyArray(arr)
        
        arrow_arr = wrapper.to_arrow()
        
        assert len(arrow_arr) == arr.size  # Arrow array is flattened
    
    def test_wrap_for_zero_copy(self, sample_arrays):
        """Test wrap_for_zero_copy convenience function."""
        from numpack.io.zero_copy import wrap_for_zero_copy, ZeroCopyArray
        
        arr = sample_arrays['int64']
        wrapper = wrap_for_zero_copy(arr)
        
        assert isinstance(wrapper, ZeroCopyArray)
        assert wrapper.shape == arr.shape


# =============================================================================
# Integration Tests with NumPack
# =============================================================================

class TestNumPackIntegration:
    """Integration tests with NumPack."""
    
    @pytest.fixture
    def numpack(self):
        """Import numpack or skip if not available."""
        pytest.importorskip('numpack')
        import numpack
        return numpack
    
    def test_save_and_load_with_zero_copy_wrapper(self, numpack, temp_dir, sample_arrays):
        """Test saving and loading with ZeroCopyArray wrapper."""
        from numpack.io.zero_copy import ZeroCopyArray
        from numpack import NumPack
        
        arr = sample_arrays['float32']
        npk_path = temp_dir / 'test.npk'
        
        # Save
        with NumPack(str(npk_path)) as npk:
            npk.save({'test_array': arr})
        
        # Load and wrap
        with NumPack(str(npk_path)) as npk:
            loaded = npk.load('test_array')
            wrapper = ZeroCopyArray(loaded)
            
            np.testing.assert_array_almost_equal(wrapper.array, arr)
    
    @pytest.fixture
    def torch(self):
        pytest.importorskip('torch')
        import torch
        return torch
    
    def test_numpack_to_torch_zero_copy(self, numpack, torch, temp_dir, sample_arrays):
        """Test zero-copy conversion from NumPack to PyTorch."""
        from numpack.io.zero_copy import numpy_to_torch_zero_copy
        from numpack import NumPack
        
        arr = sample_arrays['float32']
        npk_path = temp_dir / 'test.npk'
        
        # Save to NumPack
        with NumPack(str(npk_path)) as npk:
            npk.save({'data': arr})
        
        # Load and convert to tensor
        with NumPack(str(npk_path)) as npk:
            loaded = npk.load('data')
            tensor = numpy_to_torch_zero_copy(loaded)
            
            np.testing.assert_array_almost_equal(tensor.numpy(), arr)


# =============================================================================
# Performance Tests (optional)
# =============================================================================

class TestPerformance:
    """Performance tests for zero-copy conversions."""
    
    @pytest.mark.slow
    def test_zero_copy_vs_copy_performance(self):
        """Compare zero-copy vs regular copy performance."""
        import time
        from numpack.io.zero_copy import ZeroCopyArray
        
        # Create a large array
        large_arr = np.random.randn(10000, 1000).astype(np.float32)
        
        # Time zero-copy wrapper creation
        start = time.perf_counter()
        for _ in range(100):
            wrapper = ZeroCopyArray(large_arr)
        zero_copy_time = time.perf_counter() - start
        
        # Time regular copy
        start = time.perf_counter()
        for _ in range(100):
            copy = large_arr.copy()
        copy_time = time.perf_counter() - start
        
        # Zero-copy should be significantly faster
        assert zero_copy_time < copy_time, f"Zero-copy ({zero_copy_time:.4f}s) should be faster than copy ({copy_time:.4f}s)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
