"""Tests for drop API functionality in different scenarios."""
import numpy as np
import pytest
import tempfile
import shutil
import os
from numpack import NumPack
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import conftest
ALL_DTYPES = conftest.ALL_DTYPES
create_test_array = conftest.create_test_array


class TestDropScenarios:
    """Test drop API correctness in various scenarios."""
    
    @pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
    def test_drop_single_row_int_index(self, dtype, test_values):
        """Test dropping a single row (integer index) - all data types."""
        test_data = create_test_array(dtype, (10, 10))
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                original_shape = npk.get_shape('data')
                
                # Drop row 5
                npk.drop('data', 5)
                new_shape = npk.get_shape('data')
                
                assert original_shape == (10, 10)
                assert new_shape == (9, 10)
                
                # 验证数据正确性
                loaded = npk.load('data')
                expected = np.delete(test_data, 5, axis=0)
                
                if dtype == np.bool_:
                    assert np.array_equal(loaded, expected)
                elif np.issubdtype(dtype, np.complexfloating):
                    assert np.allclose(loaded, expected)
                else:
                    assert np.allclose(loaded, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    @pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
    def test_drop_multiple_rows_list_index(self, dtype, test_values):
        """Test dropping multiple rows (list index) - all data types."""
        test_data = create_test_array(dtype, (20, 10))
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # Drop multiple rows
                indices_to_drop = [0, 5, 10, 15]
                npk.drop('data', indices_to_drop)
                new_shape = npk.get_shape('data')
                
                assert new_shape == (16, 10)
                
                # 验证数据正确性
                loaded = npk.load('data')
                expected = np.delete(test_data, indices_to_drop, axis=0)
                
                if dtype == np.bool_:
                    assert np.array_equal(loaded, expected)
                elif np.issubdtype(dtype, np.complexfloating):
                    assert np.allclose(loaded, expected)
                else:
                    assert np.allclose(loaded, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    @pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
    def test_drop_multiple_rows_numpy_array_index(self, dtype, test_values):
        """Test dropping multiple rows (numpy array index) - all data types."""
        test_data = create_test_array(dtype, (30, 10))
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # Use numpy array as index
                indices_to_drop = np.array([2, 7, 12, 17, 22])
                npk.drop('data', indices_to_drop)
                new_shape = npk.get_shape('data')
                
                assert new_shape == (25, 10)
                
                # 验证数据正确性
                loaded = npk.load('data')
                expected = np.delete(test_data, indices_to_drop, axis=0)
                
                if dtype == np.bool_:
                    assert np.array_equal(loaded, expected)
                elif np.issubdtype(dtype, np.complexfloating):
                    assert np.allclose(loaded, expected)
                else:
                    assert np.allclose(loaded, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_single_array(self):
        """Test dropping a single array."""
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data1 = np.random.rand(100, 10).astype(np.float32)
                data2 = np.random.rand(200, 10).astype(np.float32)
                npk.save({'array1': data1, 'array2': data2})
                
                # Drop single array
                npk.drop('array1')
                members = npk.get_member_list()
                
                assert 'array1' not in members
                assert 'array2' in members
                assert len(members) == 1
                
                # Verify array2 can still be loaded normally
                loaded = npk.load('array2')
                assert np.allclose(loaded, data2)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_multiple_arrays(self):
        """Test dropping multiple arrays."""
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data1 = np.random.rand(100, 10).astype(np.float32)
                data2 = np.random.rand(200, 10).astype(np.float32)
                data3 = np.random.rand(300, 10).astype(np.float32)
                npk.save({'array1': data1, 'array2': data2, 'array3': data3})
                
                # Drop multiple arrays
                npk.drop(['array1', 'array3'])
                members = npk.get_member_list()
                
                assert 'array1' not in members
                assert 'array2' in members
                assert 'array3' not in members
                assert len(members) == 1
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_then_append(self):
        """Test appending data after drop."""
        test_data = np.arange(1000).reshape(100, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # Drop last 20 rows
                npk.drop('data', list(range(80, 100)))
                assert npk.get_shape('data') == (80, 10)
                
                # Append 30 new rows
                new_data = np.ones((30, 10), dtype=np.float32) * 999
                npk.append({'data': new_data})
                
                assert npk.get_shape('data') == (110, 10)
                
                # 验证数据
                loaded = npk.load('data')
                assert loaded.shape == (110, 10)
                # First 80 rows should be original data
                assert np.allclose(loaded[:80], test_data[:80])
                # Last 30 rows should be new data
                assert np.allclose(loaded[80:], new_data)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_then_replace(self):
        """Test replacing data after drop."""
        test_data = np.arange(1000).reshape(100, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # Drop indices [5, 10, 15]
                npk.drop('data', [5, 10, 15])
                assert npk.get_shape('data') == (97, 10)
                
                # Replace row 1 (logical index)
                replacement = np.ones((1, 10), dtype=np.float32) * 888
                npk.replace({'data': replacement}, 1)
                
                # 验证
                loaded = npk.load('data')
                assert np.allclose(loaded[1], 888)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_then_update_compact(self):
        """Test physical compaction after drop.
        
        Note: update operation physically removes marked-for-deletion rows, creating a new compact file.
        """
        test_data = np.random.rand(1000, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # Drop 500 rows
                indices_to_drop = list(range(0, 500))
                npk.drop('data', indices_to_drop)
                shape_after_drop = npk.get_shape('data')
                assert shape_after_drop == (500, 10)
                
                # Get data after drop
                data_before_compact = npk.load('data')
                assert data_before_compact.shape == (500, 10)
                
                # Physical compaction - this creates a new compact file
                npk.update('data')
                
                # Verify shape immediately after update
                shape_after_update = npk.get_shape('data')
                assert shape_after_update == (500, 10)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_with_batch_mode(self):
        """Test drop operation in batch_mode."""
        test_data = np.arange(500).reshape(50, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                with npk.batch_mode():
                    # Drop in batch mode
                    npk.drop('data', [5, 10, 15])
                    assert npk.get_shape('data') == (47, 10)
                    
                    # Load data
                    loaded = npk.load('data')
                    expected = np.delete(test_data, [5, 10, 15], axis=0)
                    assert np.allclose(loaded, expected)
                
                # Verify after batch mode exit
                loaded_after = npk.load('data')
                assert loaded_after.shape == (47, 10)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_all_rows(self):
        """Test dropping all rows.
        
        Note: When all rows are dropped, the array itself is removed.
        """
        test_data = np.arange(100).reshape(10, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # Drop all rows - this causes the array to be completely removed
                npk.drop('data', list(range(10)))
                
                # Verify array no longer exists
                members = npk.get_member_list()
                assert 'data' not in members
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_with_tuple_index(self):
        """Test dropping with tuple index."""
        test_data = np.arange(100).reshape(10, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # Use tuple index
                npk.drop('data', (2, 5, 8))
                
                assert npk.get_shape('data') == (7, 10)
                
                loaded = npk.load('data')
                expected = np.delete(test_data, [2, 5, 8], axis=0)
                assert np.allclose(loaded, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_consecutive_operations(self):
        """Test consecutive drop operations."""
        test_data = np.arange(1000).reshape(100, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # First drop
                npk.drop('data', [0, 1, 2])
                assert npk.get_shape('data') == (97, 10)
                
                # Second drop
                npk.drop('data', [5, 10])
                assert npk.get_shape('data') == (95, 10)
                
                # Third drop
                npk.drop('data', [20, 30, 40])
                assert npk.get_shape('data') == (92, 10)
                
                # Verify final result
                loaded = npk.load('data')
                assert loaded.shape == (92, 10)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    @pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
    def test_drop_different_dtypes(self, dtype, test_values):
        """Test drop operations for all data types."""
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                array_name = f'array_{dtype.__name__}'
                test_data = create_test_array(dtype, (10, 10))
                npk.save({array_name: test_data})
                
                # Drop row 5
                npk.drop(array_name, 5)
                
                # 验证
                assert npk.get_shape(array_name) == (9, 10)
                loaded = npk.load(array_name)
                expected = np.delete(test_data, 5, axis=0)
                
                if dtype == np.bool_:
                    assert np.array_equal(loaded, expected)
                elif np.issubdtype(dtype, np.complexfloating):
                    assert np.allclose(loaded, expected)
                else:
                    assert np.allclose(loaded, expected)
                assert loaded.dtype == dtype
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_lazy_load_consistency(self):
        """Test lazy load consistency after drop."""
        test_data = np.random.rand(1000, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # Drop some rows
                npk.drop('data', [10, 20, 30, 40, 50])
                
                # Compare eager and lazy loading
                eager = npk.load('data', lazy=False)
                lazy = npk.load('data', lazy=True)
                
                # Shapes should match
                assert eager.shape == lazy.shape == (995, 10)
                
                # Random sampling to verify data consistency
                indices = [0, 100, 500, 900]
                for idx in indices:
                    assert np.allclose(eager[idx], lazy[idx])
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_empty_list(self):
        """Test dropping with empty index list (should not delete anything)."""
        test_data = np.arange(100).reshape(10, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # Drop with empty list
                npk.drop('data', [])
                
                # Shape should remain unchanged
                assert npk.get_shape('data') == (10, 10)
                
                # Data should be completely identical
                loaded = npk.load('data')
                assert np.allclose(loaded, test_data)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_reopen_file(self):
        """Test reopening file after drop."""
        test_data = np.arange(1000).reshape(100, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # First open, drop and save
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                npk.drop('data', list(range(50)))
                assert npk.get_shape('data') == (50, 10)
            
            # Reopen, verify drop is persisted
            with NumPack(numpack_dir) as npk:
                assert npk.get_shape('data') == (50, 10)
                loaded = npk.load('data')
                expected = test_data[50:]
                assert np.allclose(loaded, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_multidimensional_arrays(self):
        """Test dropping rows from multidimensional arrays."""
        # 3D array
        test_data_3d = np.random.rand(50, 20, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data_3d': test_data_3d})
                
                # Drop some indices from first dimension
                npk.drop('data_3d', [5, 10, 15])
                
                assert npk.get_shape('data_3d') == (47, 20, 10)
                
                loaded = npk.load('data_3d')
                expected = np.delete(test_data_3d, [5, 10, 15], axis=0)
                assert np.allclose(loaded, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_getitem_after_deletion(self):
        """Test using getitem after deletion."""
        test_data = np.arange(1000).reshape(100, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # Drop some rows
                deleted_indices = [5, 10, 15, 20]
                npk.drop('data', deleted_indices)
                
                # Use getitem to access
                item_0 = npk.getitem('data', 0)
                item_5 = npk.getitem('data', 5)
                
                # 验证
                full_data = npk.load('data')
                assert np.allclose(item_0, full_data[0])
                assert np.allclose(item_5, full_data[5])
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


class TestDropErrorHandling:
    """Test drop API error handling.
    
    Note: Some error handling may be implemented in Rust backend; behavior may differ from expectations.
    """
    
    def test_drop_nonexistent_array_silent(self):
        """Test dropping non-existent array.
        
        Note: Current implementation may silently ignore non-existent arrays without raising errors.
        This is a design decision to avoid interrupting batch operations due to a single missing array.
        """
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                # Create an array
                npk.save({'array1': np.random.rand(10, 10).astype(np.float32)})
                
                # Try to drop non-existent array (may silently succeed)
                npk.drop('nonexistent')
                
                # Verify original array still exists
                assert 'array1' in npk.get_member_list()
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_duplicate_index(self):
        """Test dropping duplicate indices (should be deduplicated)."""
        test_data = np.arange(100).reshape(10, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
                
                # Drop duplicate indices (should be automatically deduplicated)
                npk.drop('data', [0, 0, 1, 1, 2])
                
                # Should only drop 3 rows
                shape = npk.get_shape('data')
                assert shape == (7, 10)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

