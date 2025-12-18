"""
Cross-platform consistency tests.

Ensure NumPack behaves consistently on Windows, macOS, and Linux.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys

from numpack import NumPack, get_backend_info


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestCrossPlatformConsistency:
    """Verify cross-platform consistency."""
    
    def test_same_backend_all_platforms(self):
        """All platforms should use the same backend (Rust)."""
        info = get_backend_info()
        # Verify backend type
        assert info['backend_type'] == 'rust'
    
    def test_file_format_compatibility(self, temp_dir):
        """File format should be cross-platform compatible."""
        npk_path = temp_dir / "test.npk"
        
        test_arrays = {
            'int32': np.arange(100, dtype=np.int32),
            'float64': np.random.randn(50, 50),
            'bool': np.array([True, False, True, False]),
            'uint8': np.arange(256, dtype=np.uint8),
        }
        
        # Save
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save(test_arrays)
        
        # Load and verify
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            for name, expected in test_arrays.items():
                loaded = npk.load(name)
                assert np.array_equal(loaded, expected), \
                    f"Array '{name}' not equal after save/load"
    
    def test_context_manager_behavior(self, temp_dir):
        """Context manager should behave consistently on all platforms."""
        npk_path = temp_dir / "test.npk"
        
        # Create and use context manager
        with NumPack(str(npk_path), strict_context_mode=True, warn_no_context=False) as npk:
            npk.save({'data': np.arange(10)})
            result = npk.load('data')
            assert len(result) == 10
        
        # Operations after context exit should fail (all platforms)
        with pytest.raises(RuntimeError, match="closed"):
            npk.save({'more': np.arange(5)})
    
    def test_strict_mode_consistent(self, temp_dir):
        """Strict mode behaves consistently on all platforms."""
        npk_path = temp_dir / "test.npk"
        
        npk = NumPack(str(npk_path), strict_context_mode=True, warn_no_context=False)
        npk.open()  # Must open first to test strict mode
        
        # All platforms should block non-context usage
        with pytest.raises(RuntimeError, match="strict context mode"):
            npk.save({'data': np.array([1, 2, 3])})
        
        npk.close()
    
    def test_data_types_consistency(self, temp_dir):
        """All data types are consistent across all platforms."""
        npk_path = temp_dir / "types.npk"
        
        # Test various data types
        test_data = {
            'bool': np.array([True, False, True], dtype=np.bool_),
            'int8': np.array([-128, 0, 127], dtype=np.int8),
            'int16': np.array([-32768, 0, 32767], dtype=np.int16),
            'int32': np.array([-2147483648, 0, 2147483647], dtype=np.int32),
            'int64': np.array([-9223372036854775808, 0, 9223372036854775807], dtype=np.int64),
            'uint8': np.array([0, 128, 255], dtype=np.uint8),
            'uint16': np.array([0, 32768, 65535], dtype=np.uint16),
            'uint32': np.array([0, 2147483648, 4294967295], dtype=np.uint32),
            'float32': np.array([0.0, 3.14, -2.71], dtype=np.float32),
            'float64': np.array([0.0, 3.141592653589793, -2.718281828459045], dtype=np.float64),
        }
        
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save(test_data)
            
            for name, expected in test_data.items():
                loaded = npk.load(name)
                if np.issubdtype(expected.dtype, np.floating):
                    assert np.allclose(loaded, expected), f"{name} data type mismatch"
                else:
                    assert np.array_equal(loaded, expected), f"{name} data type mismatch"
    
    def test_lazy_loading_consistency(self, temp_dir):
        """LazyArray behaves consistently on all platforms."""
        npk_path = temp_dir / "lazy.npk"
        data = np.arange(1000).reshape(100, 10)
        
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save({'data': data})
            
            # Lazy load
            lazy = npk.load('data', lazy=True)
            
            # Various indexing methods
            assert np.array_equal(lazy[0], data[0])
            assert np.array_equal(lazy[0:10], data[0:10])
            assert np.array_equal(lazy[[1, 5, 10]], data[[1, 5, 10]])
    
    def test_metadata_consistency(self, temp_dir):
        """Metadata is consistent across all platforms."""
        npk_path = temp_dir / "meta.npk"
        
        test_data = {
            'array1': np.random.randn(100, 50),
            'array2': np.random.randn(200, 30),
        }
        
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save(test_data)
            
            # Check metadata
            members = npk.get_member_list()
            assert set(members) == {'array1', 'array2'}
            
            assert npk.get_shape('array1') == (100, 50)
            assert npk.get_shape('array2') == (200, 30)
            
            assert npk.has_array('array1')
            assert npk.has_array('array2')
            assert not npk.has_array('nonexistent')


class TestPerformanceConsistency:
    """Test performance consistency."""
    
    def test_backend_info(self):
        """Verify backend info is correct."""
        info = get_backend_info()
        
        assert 'backend_type' in info
        assert 'platform' in info
        assert 'is_windows' in info
        assert 'version' in info
        

        
        # Platform should be a valid value
        assert info['platform'] in ['Windows', 'Darwin', 'Linux']
        
        # is_windows should match platform
        assert info['is_windows'] == (info['platform'] == 'Windows')
    
    def test_batch_operations_work(self, temp_dir):
        """Verify batch operations work on all platforms."""
        npk_path = temp_dir / "batch.npk"
        data = np.arange(10000).reshape(1000, 10)
        
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save({'data': data})
            
            # Batch index access
            indices = np.array([0, 10, 50, 100, 500, 999])
            result = npk.load('data', lazy=True)[indices]
            
            expected = data[indices]
            assert np.array_equal(result, expected)


class TestWarningBehavior:
    """Test warning behavior."""
    
    def test_warning_on_windows_without_context(self, temp_dir):
        """Test warning on Windows when not using context manager."""
        npk_path = temp_dir / "warn_test.npk"
        
        # Should warn on Windows, not on other platforms (unless explicitly set)
        if sys.platform.startswith('win'):
            with pytest.warns(UserWarning, match="strict context mode"):
                npk = NumPack(str(npk_path))
                npk.close()
        else:
            # Non-Windows platforms do not warn by default
            npk = NumPack(str(npk_path))  # Should not warn
            npk.close()
    
    def test_warning_suppression(self, temp_dir):
        """Test that warnings can be suppressed."""
        npk_path = temp_dir / "no_warn.npk"
        
        # Explicitly setting warn_no_context=False should not warn
        npk = NumPack(str(npk_path), warn_no_context=False)
        npk.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

