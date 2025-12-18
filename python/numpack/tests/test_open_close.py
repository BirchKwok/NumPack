"""Tests for NumPack open and close methods."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from numpack import NumPack


class TestOpenClose:
    """Tests for open and close method functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp = tempfile.mkdtemp()
        yield temp
        if Path(temp).exists():
            shutil.rmtree(temp)
    
    def test_no_auto_open(self, temp_dir):
        """Test default behavior: file does not auto-open."""
        npk_path = Path(temp_dir) / "test_no_auto_open.npk"
        
        # Should not auto-open when creating instance
        npk = NumPack(npk_path)
        assert not npk.is_opened
        assert npk.is_closed
        
        # Must manually open before use
        npk.open()
        assert npk.is_opened
        assert not npk.is_closed
        
        # Now can save and load data
        test_data = {'array1': np.array([1, 2, 3])}
        npk.save(test_data)
        loaded = npk.load('array1')
        np.testing.assert_array_equal(loaded, test_data['array1'])
        
        npk.close()
        assert npk.is_closed
        assert not npk.is_opened
    
    def test_must_open_before_use(self, temp_dir):
        """Test that file must be opened before use."""
        npk_path = Path(temp_dir) / "test_must_open.npk"
        
        # Create instance
        npk = NumPack(npk_path)
        assert not npk.is_opened
        assert npk.is_closed
        
        # Attempting to use before opening should fail
        with pytest.raises(RuntimeError, match="not opened or has been closed"):
            npk.save({'array1': np.array([1, 2, 3])})
        
        with pytest.raises(RuntimeError, match="not opened or has been closed"):
            npk.get_member_list()
        
        # Can use after manually opening
        npk.open()
        assert npk.is_opened
        assert not npk.is_closed
        
        test_data = {'array1': np.array([1, 2, 3])}
        npk.save(test_data)
        loaded = npk.load('array1')
        np.testing.assert_array_equal(loaded, test_data['array1'])
        
        npk.close()
    
    def test_reopen_after_close(self, temp_dir):
        """Test reopening after close."""
        npk_path = Path(temp_dir) / "test_reopen.npk"
        
        # First use
        npk = NumPack(npk_path)
        npk.open()
        test_data = {'array1': np.array([1, 2, 3, 4, 5])}
        npk.save(test_data)
        npk.close()
        
        assert npk.is_closed
        assert not npk.is_opened
        
        # Reopen
        npk.open()
        assert npk.is_opened
        assert not npk.is_closed
        
        # Should be able to load previously saved data
        loaded = npk.load('array1')
        np.testing.assert_array_equal(loaded, test_data['array1'])
        
        # Save more data
        new_data = {'array2': np.array([10, 20, 30])}
        npk.save(new_data)
        
        # Verify both arrays exist
        assert 'array1' in npk.get_member_list()
        assert 'array2' in npk.get_member_list()
        
        npk.close()
    
    def test_multiple_open_calls(self, temp_dir):
        """Test that multiple open calls are idempotent."""
        npk_path = Path(temp_dir) / "test_multiple_open.npk"
        
        npk = NumPack(npk_path)
        npk.open()
        npk.open()  # Second call should not error
        npk.open()  # Third call should not error
        
        assert npk.is_opened
        
        test_data = {'array1': np.array([1, 2, 3])}
        npk.save(test_data)
        
        npk.close()
    
    def test_multiple_close_calls(self, temp_dir):
        """Test that multiple close calls are idempotent."""
        npk_path = Path(temp_dir) / "test_multiple_close.npk"
        
        npk = NumPack(npk_path)
        npk.open()
        test_data = {'array1': np.array([1, 2, 3])}
        npk.save(test_data)
        
        npk.close()
        npk.close()  # Second call should not error
        npk.close()  # Third call should not error
        
        assert npk.is_closed
    
    def test_context_manager_auto_open(self, temp_dir):
        """Test that context manager auto-opens the file."""
        npk_path = Path(temp_dir) / "test_context_auto_open.npk"
        
        # Does not auto-open when creating instance
        npk = NumPack(npk_path)
        assert not npk.is_opened
        
        # Using context manager will auto-open
        with npk as n:
            assert n.is_opened
            test_data = {'array1': np.array([1, 2, 3])}
            n.save(test_data)
        
        # Should close after exiting context
        assert npk.is_closed
    
    def test_context_manager_reopen(self, temp_dir):
        """Test that context manager can reopen a closed file."""
        npk_path = Path(temp_dir) / "test_context_reopen.npk"
        
        # First use
        with NumPack(npk_path) as npk:
            test_data = {'array1': np.array([1, 2, 3])}
            npk.save(test_data)
        
        # File should be closed
        assert npk.is_closed
        
        # Reuse the same instance
        with npk as n:
            assert n.is_opened
            loaded = n.load('array1')
            np.testing.assert_array_equal(loaded, test_data['array1'])
        
        assert npk.is_closed
    
    def test_open_close_cycle(self, temp_dir):
        """Test multiple open-close cycles."""
        npk_path = Path(temp_dir) / "test_open_close_cycle.npk"
        
        npk = NumPack(npk_path)
        
        for i in range(5):
            npk.open()
            assert npk.is_opened
            
            # Save data
            test_data = {f'array{i}': np.array([i, i+1, i+2])}
            npk.save(test_data)
            
            npk.close()
            assert npk.is_closed
        
        # Final open, verify all data exists
        npk.open()
        members = npk.get_member_list()
        assert len(members) == 5
        for i in range(5):
            assert f'array{i}' in members
        npk.close()
    
    def test_drop_if_exists_with_manual_open(self, temp_dir):
        """Test drop_if_exists with manual open."""
        npk_path = Path(temp_dir) / "test_drop.npk"
        
        # First file creation
        npk1 = NumPack(npk_path)
        npk1.open()
        npk1.save({'array1': np.array([1, 2, 3])})
        npk1.close()
        
        # Second open with drop_if_exists=True
        npk2 = NumPack(npk_path, drop_if_exists=True)
        npk2.open()
        
        # Should be empty
        assert len(npk2.get_member_list()) == 0
        
        npk2.save({'array2': np.array([4, 5, 6])})
        members = npk2.get_member_list()
        assert len(members) == 1
        assert 'array2' in members
        assert 'array1' not in members
        
        npk2.close()
    
    def test_error_after_close(self, temp_dir):
        """Test that calling methods after close raises errors."""
        npk_path = Path(temp_dir) / "test_error_after_close.npk"
        
        npk = NumPack(npk_path)
        npk.open()
        test_data = {'array1': np.array([1, 2, 3])}
        npk.save(test_data)
        npk.close()
        
        # All operations should fail
        with pytest.raises(RuntimeError, match="not opened or has been closed"):
            npk.save({'array2': np.array([4, 5, 6])})
        
        with pytest.raises(RuntimeError, match="not opened or has been closed"):
            npk.load('array1')
        
        with pytest.raises(RuntimeError, match="not opened or has been closed"):
            npk.get_member_list()
    
    def test_properties(self, temp_dir):
        """Test is_opened and is_closed properties."""
        npk_path = Path(temp_dir) / "test_properties.npk"
        
        # Unopened state
        npk = NumPack(npk_path)
        assert not npk.is_opened
        assert npk.is_closed
        
        # Opened state
        npk.open()
        assert npk.is_opened
        assert not npk.is_closed
        
        # Closed state
        npk.close()
        assert not npk.is_opened
        assert npk.is_closed
        
        # Reopen
        npk.open()
        assert npk.is_opened
        assert not npk.is_closed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

