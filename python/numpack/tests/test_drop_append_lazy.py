"""Tests for consistency of drop and append operations with lazy loading."""
import numpy as np
import pytest
import tempfile
import shutil
import os
from numpack import NumPack


class TestDropAppendLazyConsistency:
    """Tests for shape consistency of drop and append operations with lazy loading."""
    
    def test_drop_single_row_shape_consistency(self):
        """Test shape consistency of lazy and eager loading after dropping a single row."""
        test_data = np.random.rand(1000, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # Save data
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # Drop first row
            with NumPack(numpack_dir) as npk:
                npk.drop('data', 0)
                shape_after_drop = npk.get_shape('data')
                assert shape_after_drop == (999, 10), f"Expected (999, 10), got {shape_after_drop}"
            
            # Verify shape consistency of lazy and eager loading
            with NumPack(numpack_dir) as npk:
                arr_eager = npk.load('data', lazy=False)
                arr_lazy = npk.load('data', lazy=True)
                
                assert arr_eager.shape == (999, 10), f"Eager shape: {arr_eager.shape}"
                assert arr_lazy.shape == (999, 10), f"Lazy shape: {arr_lazy.shape}"
                assert arr_eager.shape == arr_lazy.shape, "Lazy and eager shapes must match"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_multiple_rows_shape_consistency(self):
        """Test shape consistency of lazy and eager loading after dropping multiple rows."""
        test_data = np.random.rand(1000, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # Save data
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # Drop multiple rows
            with NumPack(numpack_dir) as npk:
                npk.drop('data', [0, 5, 10, 100])
                shape_after_drop = npk.get_shape('data')
                assert shape_after_drop == (996, 10), f"Expected (996, 10), got {shape_after_drop}"
            
            # Verify shape consistency of lazy and eager loading
            with NumPack(numpack_dir) as npk:
                arr_eager = npk.load('data', lazy=False)
                arr_lazy = npk.load('data', lazy=True)
                
                assert arr_eager.shape == (996, 10), f"Eager shape: {arr_eager.shape}"
                assert arr_lazy.shape == (996, 10), f"Lazy shape: {arr_lazy.shape}"
                assert arr_eager.shape == arr_lazy.shape, "Lazy and eager shapes must match"
                
                # Note: when deletion bitmap exists, buffer protocol may return physical data
                # Should verify data consistency via indexing rather than np.array conversion
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_append_after_drop_shape_consistency(self):
        """Test shape consistency of append after drop."""
        test_data = np.random.rand(1000000, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # Save data
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # Drop one row
            with NumPack(numpack_dir) as npk:
                npk.drop('data', 0)
                shape_after_drop = npk.get_shape('data')
                assert shape_after_drop == (999999, 10), f"Expected (999999, 10), got {shape_after_drop}"
            
            # Append data
            with NumPack(numpack_dir) as npk:
                npk.append({'data': test_data})
                shape_after_append = npk.get_shape('data')
                assert shape_after_append == (1999999, 10), f"Expected (1999999, 10), got {shape_after_append}"
                
                # Verify shape consistency of lazy and eager loading
                arr_eager = npk.load('data', lazy=False)
                arr_lazy = npk.load('data', lazy=True)
                
                assert arr_eager.shape == (1999999, 10), f"Eager shape: {arr_eager.shape}"
                assert arr_lazy.shape == (1999999, 10), f"Lazy shape: {arr_lazy.shape}"
                assert arr_eager.shape == arr_lazy.shape, "Lazy and eager shapes must match after append"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_lazy_array_indexing_after_drop(self):
        """Test lazy array indexing after drop."""
        test_data = np.arange(100 * 5, dtype=np.float32).reshape(100, 5)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # Save data
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # Drop indices [5, 10, 15]
            with NumPack(numpack_dir) as npk:
                npk.drop('data', [5, 10, 15])
            
            # Verify lazy array indexing
            with NumPack(numpack_dir) as npk:
                arr_lazy = npk.load('data', lazy=True)
                arr_eager = npk.load('data', lazy=False)
                
                # Shapes should match
                assert arr_lazy.shape == arr_eager.shape == (97, 5)
                
                # Access first row (logical index 0, physical index 0)
                assert np.allclose(arr_lazy[0], arr_eager[0])
                
                # Access 6th row (logical index 5, physical index 6, because physical index 5 was deleted)
                assert np.allclose(arr_lazy[5], arr_eager[5])
                
                # Batch indexing
                assert np.allclose(arr_lazy[[0, 5, 10]], arr_eager[[0, 5, 10]])
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_lazy_array_iteration_after_drop(self):
        """Test lazy array iteration after drop."""
        test_data = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # Save data
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # Drop indices [0, 5, 10]
            with NumPack(numpack_dir) as npk:
                npk.drop('data', [0, 5, 10])
            
            # Verify lazy array iteration
            with NumPack(numpack_dir) as npk:
                arr_lazy = npk.load('data', lazy=True)
                arr_eager = npk.load('data', lazy=False)
                
                # Shapes should match
                assert len(arr_lazy) == len(arr_eager) == 17
                
                # Iteration should produce the same number of rows
                lazy_rows = list(arr_lazy)
                assert len(lazy_rows) == 17, f"Expected 17 rows, got {len(lazy_rows)}"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_update_removes_deletion_bitmap(self):
        """Test that update operation removes deletion bitmap; lazy and eager shapes match."""
        test_data = np.random.rand(1000, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # Save data
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # Drop some rows
            with NumPack(numpack_dir) as npk:
                npk.drop('data', [0, 1, 2])
                shape_after_drop = npk.get_shape('data')
                assert shape_after_drop == (997, 10)
            
            # Physical compaction
            with NumPack(numpack_dir) as npk:
                npk.update('data')
                shape_after_update = npk.get_shape('data')
                assert shape_after_update == (997, 10)
                
                # Verify shape consistency of lazy and eager loading
                arr_eager = npk.load('data', lazy=False)
                arr_lazy = npk.load('data', lazy=True)
                
                assert arr_eager.shape == (997, 10), f"Eager shape: {arr_eager.shape}"
                assert arr_lazy.shape == (997, 10), f"Lazy shape: {arr_lazy.shape}"
                assert arr_eager.shape == arr_lazy.shape
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_no_bitmap_fast_path(self):
        """Test fast path performance when there is no deletion bitmap."""
        test_data = np.random.rand(10000, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # Save data
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # No deletion operations, load directly
            with NumPack(numpack_dir) as npk:
                import time
                
                # Test lazy load performance (should be fast because no bitmap)
                start = time.time()
                arr_lazy = npk.load('data', lazy=True)
                lazy_time = time.time() - start
                
                assert arr_lazy.shape == (10000, 10)
                assert lazy_time < 0.1, f"Lazy load too slow: {lazy_time}s"  # Should be <100ms
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


if __name__ == "__main__":
    import os
    # Run tests manually
    test_cls = TestDropAppendLazyConsistency()
    test_cls.test_drop_single_row_shape_consistency()
    print("✓ test_drop_single_row_shape_consistency passed")
    
    test_cls.test_drop_multiple_rows_shape_consistency()
    print("✓ test_drop_multiple_rows_shape_consistency passed")
    
    test_cls.test_append_after_drop_shape_consistency()
    print("✓ test_append_after_drop_shape_consistency passed")
    
    test_cls.test_lazy_array_indexing_after_drop()
    print("✓ test_lazy_array_indexing_after_drop passed")
    
    test_cls.test_lazy_array_iteration_after_drop()
    print("✓ test_lazy_array_iteration_after_drop passed")
    
    test_cls.test_update_removes_deletion_bitmap()
    print("✓ test_update_removes_deletion_bitmap passed")
    
    test_cls.test_no_bitmap_fast_path()
    print("✓ test_no_bitmap_fast_path passed")
    
    print("\nAll tests passed!")

