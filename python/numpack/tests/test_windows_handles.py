"""
Windows-specific tests for file handle management and context managers

These tests primarily run on Windows to verify correct handle management and context manager behavior.
On non-Windows platforms, some tests are skipped, but generic context manager tests still run.
"""

import pytest
import numpy as np
import sys
import shutil
import tempfile
import time
from pathlib import Path

# Import NumPack
from numpack import NumPack, get_backend_info


# Run certain tests only on Windows
windows_only = pytest.mark.skipif(
    not sys.platform.startswith('win'),
    reason="Windows-specific tests"
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestContextManagerBasic:
    """Test basic context manager functionality (all platforms)."""
    
    def test_context_manager_basic_usage(self, temp_dir):
        """Verify basic context manager usage."""
        npk_path = temp_dir / "test.npk"
        test_data = np.arange(100, dtype=np.int32)
        
        # Use context manager
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save({'data': test_data})
            loaded = npk.load('data')
            assert np.array_equal(loaded, test_data)
        
        # Verify file was created
        assert npk_path.exists()
    
    def test_context_manager_auto_cleanup(self, temp_dir):
        """Verify context manager auto cleanup."""
        npk_path = temp_dir / "test.npk"
        test_data = np.arange(100, dtype=np.int32)
        
        # Use context manager
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save({'data': test_data})
        
        # After context exit, should be able to delete immediately (on supported platforms)
        # Note: Windows may need a brief delay
        if sys.platform.startswith('win'):
            time.sleep(0.1)
        
        try:
            shutil.rmtree(npk_path)
            assert not npk_path.exists()
        except PermissionError:
            # Windows may need more time
            time.sleep(0.5)
            shutil.rmtree(npk_path)
            assert not npk_path.exists()
    
    def test_nested_context_managers(self, temp_dir):
        """Test nested context managers."""
        npk_path1 = temp_dir / "test1.npk"
        npk_path2 = temp_dir / "test2.npk"
        
        with NumPack(str(npk_path1), warn_no_context=False) as npk1:
            npk1.save({'data1': np.arange(100)})
            
            with NumPack(str(npk_path2), warn_no_context=False) as npk2:
                npk2.save({'data2': np.arange(200)})
                loaded2 = npk2.load('data2')
                assert len(loaded2) == 200
            
            # npk2 should be closed, npk1 still open
            npk1.save({'data3': np.arange(50)})
            loaded1 = npk1.load('data1')
            assert len(loaded1) == 100


class TestWindowsHandleManagement:
    """Test Windows platform handle management (Windows priority)."""
    
    @windows_only
    def test_rapid_create_delete_windows(self, temp_dir):
        """Test rapid file creation and deletion (Windows-specific)."""
        for i in range(20):  # Reduce iterations to speed up test
            npk_path = temp_dir / f"test_{i}.npk"
            
            with NumPack(str(npk_path), warn_no_context=False) as npk:
                npk.save({'data': np.arange(100)})
            
            # Should be able to delete immediately on Windows
            time.sleep(0.05)  # Small delay to ensure cleanup completes
            shutil.rmtree(npk_path)
            assert not npk_path.exists()
    
    @windows_only
    def test_manual_close_releases_handles_windows(self, temp_dir):
        """Test that manual close() releases handles (Windows-specific)."""
        npk_path = temp_dir / "test.npk"
        
        npk = NumPack(str(npk_path), warn_no_context=False)
        npk.open()  # Must open first
        npk.save({'data': np.arange(100)})
        
        # Explicit close
        npk.close()
        
        # Should be able to delete immediately; retry if fails
        try:
            shutil.rmtree(npk_path)
        except PermissionError:
            # In rare cases on Windows may need brief delay
            time.sleep(0.1)
            shutil.rmtree(npk_path)
        
        assert not npk_path.exists()
    
    def test_lazy_array_context_manager(self, temp_dir):
        """Test LazyArray context manager (all platforms)."""
        npk_path = temp_dir / "test.npk"
        large_data = np.arange(10000, dtype=np.float64)
        
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save({'large': large_data})
            
            # Use LazyArray directly (it supports context manager itself)
            lazy_arr = npk.load('large', lazy=True)
            result = lazy_arr[0:100]
            assert len(result) == 100
        
        # Cleanup should succeed
        if sys.platform.startswith('win'):
            time.sleep(0.1)
        shutil.rmtree(npk_path)
        assert not npk_path.exists()


class TestStrictContextMode:
    """Test strict context mode."""
    
    def test_strict_mode_prevents_non_context_usage(self, temp_dir):
        """Test that strict mode prevents non-context usage."""
        npk_path = temp_dir / "test.npk"
        
        # Create strict mode instance
        npk = NumPack(str(npk_path), strict_context_mode=True, warn_no_context=False)
        npk.open()  # Must open first to test strict mode
        
        # Should raise error when used outside context
        with pytest.raises(RuntimeError, match="strict context mode"):
            npk.save({'data': np.array([1, 2, 3])})
        
        # Cleanup
        npk.close()
    
    def test_strict_mode_works_in_context(self, temp_dir):
        """Test that strict mode works normally in context."""
        npk_path = temp_dir / "test.npk"
        
        # Should work normally in context
        with NumPack(str(npk_path), strict_context_mode=True, warn_no_context=False) as npk:
            npk.save({'data': np.array([1, 2, 3])})
            result = npk.load('data')
            assert np.array_equal(result, np.array([1, 2, 3]))
    
    def test_operations_after_close_raise_error(self, temp_dir):
        """Test that operations after close raise error."""
        npk_path = temp_dir / "test.npk"
        
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save({'data': np.array([1, 2, 3])})
        
        # Should be closed after context exit
        with pytest.raises(RuntimeError, match="closed"):
            npk.save({'more': np.array([4, 5, 6])})
    
    def test_multiple_close_calls_safe(self, temp_dir):
        """Test that multiple close() calls are safe."""
        npk_path = temp_dir / "test.npk"
        
        npk = NumPack(str(npk_path), warn_no_context=False)
        npk.open()  # Manually open file
        npk.save({'data': np.arange(10)})
        
        # Multiple close calls should not raise error
        npk.close()
        npk.close()
        npk.close()


class TestExceptionHandling:
    """Test exception handling."""
    
    def test_exception_during_context_still_cleans_up(self, temp_dir):
        """Test that exceptions don't prevent cleanup."""
        npk_path = temp_dir / "test.npk"
        
        try:
            with NumPack(str(npk_path), warn_no_context=False) as npk:
                npk.save({'data': np.arange(100)})
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected exception
        
        # Should still be able to cleanup
        if sys.platform.startswith('win'):
            time.sleep(0.1)
        shutil.rmtree(npk_path)
        assert not npk_path.exists()
    
    def test_close_is_idempotent(self, temp_dir):
        """Test that close is idempotent."""
        npk_path = temp_dir / "test.npk"
        
        npk = NumPack(str(npk_path), warn_no_context=False)
        npk.open()  # Manually open file
        npk.save({'data': np.array([1, 2, 3])})
        
        # First close
        npk.close()
        
        # Subsequent close calls should not error
        for _ in range(5):
            npk.close()  # Should be safe


class TestBackendConsistency:
    """Test backend consistency."""
    
    def test_backend_available(self):
        """Verify NumPack backend is available."""
        info = get_backend_info()
        
        # Verify backend info is accessible
        assert 'backend_type' in info
        assert 'platform' in info
        assert 'version' in info
        
        # Verify backend type is consistent (now only Rust backend)
        assert info['backend_type'] == 'rust'
    
    def test_large_file_operations(self, temp_dir):
        """Test large file operation reliability."""
        npk_path = temp_dir / "large.npk"
        
        # Create larger dataset
        large_array = np.random.randn(1000, 100).astype(np.float64)
        
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save({'large': large_array})
            loaded = npk.load('large')
            
            assert np.allclose(loaded, large_array)
        
        # Verify cleanup
        if sys.platform.startswith('win'):
            time.sleep(0.1)
        shutil.rmtree(npk_path)


class TestResourceManagement:
    """Test resource management."""
    
    def test_multiple_instances_same_file(self, temp_dir):
        """Test multiple instances accessing the same file."""
        npk_path = temp_dir / "shared.npk"
        
        # First instance writes
        with NumPack(str(npk_path), warn_no_context=False) as npk1:
            npk1.save({'data': np.array([1, 2, 3])})
        
        # Second instance reads
        with NumPack(str(npk_path), warn_no_context=False) as npk2:
            result = npk2.load('data')
            assert np.array_equal(result, np.array([1, 2, 3]))
    
    def test_lazy_array_multiple_access(self, temp_dir):
        """Test multiple LazyArray accesses."""
        npk_path = temp_dir / "test.npk"
        data = np.arange(1000, dtype=np.float32)
        
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save({'data': data})
            
            lazy = npk.load('data', lazy=True)
            
            # Multiple accesses to the same LazyArray
            for _ in range(10):
                result = lazy[0:10]
                assert len(result) == 10
    
    @windows_only
    def test_windows_handle_cleanup_stress(self, temp_dir):
        """Windows handle cleanup stress test."""
        # Rapidly create and destroy multiple NumPack instances
        for i in range(50):
            npk_path = temp_dir / f"stress_{i}.npk"
            
            with NumPack(str(npk_path), warn_no_context=False) as npk:
                npk.save({'data': np.random.randn(100, 10)})
                loaded = npk.load('data')
                assert loaded.shape == (100, 10)
            
            # Cleanup every 10 iterations
            if i % 10 == 0:
                time.sleep(0.05)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

