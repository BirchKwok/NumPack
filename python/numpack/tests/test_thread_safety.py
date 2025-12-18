"""Tests for NumPack API thread safety."""
import numpy as np
import pytest
import tempfile
import shutil
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from numpack import NumPack


class TestThreadSafety:
    """Test NumPack safety in multi-threaded environments."""
    
    def test_concurrent_reads_same_array(self):
        """Test multiple threads reading the same array concurrently."""
        test_data = np.random.rand(1000, 10).astype(np.float32)
        numpack_dir = tempfile.mkdtemp()
        num_threads = 10
        
        try:
            # Save data first
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # Multi-threaded read
            def read_worker(thread_id):
                with NumPack(numpack_dir) as npk:
                    loaded = npk.load('data')
                    # Verify data integrity
                    return np.allclose(loaded, test_data) and loaded.shape == test_data.shape
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(read_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            # All reads should succeed
            assert all(results), "Some threads failed to read correctly"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_concurrent_reads_different_arrays(self):
        """Test multiple threads reading different arrays."""
        num_arrays = 20
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # Create multiple arrays
            test_arrays = {}
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                for i in range(num_arrays):
                    array_name = f'array_{i}'
                    test_arrays[array_name] = np.random.rand(100, 10).astype(np.float32)
                npk.save(test_arrays)
            
            # Multi-threaded read of different arrays
            def read_worker(array_id):
                array_name = f'array_{array_id}'
                with NumPack(numpack_dir) as npk:
                    loaded = npk.load(array_name)
                    expected = test_arrays[array_name]
                    return np.allclose(loaded, expected)
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(read_worker, i) for i in range(num_arrays)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some threads failed to read correctly"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_concurrent_writes_different_arrays(self):
        """Test multiple threads writing different arrays.
        
        Note: Concurrent writes require proper synchronization.
        In practice, use locks to serialize write operations.
        """
        num_threads = 10
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # Initialize empty file
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                pass
            
            # Multi-threaded write of different arrays - use locks for thread safety
            test_data = {}
            data_lock = threading.Lock()
            write_lock = threading.Lock()  # Serialize write operations
            
            def write_worker(thread_id):
                array_name = f'array_{thread_id}'
                data = np.random.rand(100, 10).astype(np.float32)
                
                with data_lock:  # Protect shared test_data dict
                    test_data[array_name] = data.copy()
                
                # Use lock to serialize writes and avoid concurrent conflicts
                with write_lock:
                    with NumPack(numpack_dir) as npk:
                        npk.save({array_name: data})
                return True
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(write_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some writes failed"
            
            # Verify all data was written correctly
            with NumPack(numpack_dir) as npk:
                members = npk.get_member_list()
                assert len(members) == num_threads, f"Expected {num_threads} arrays, got {len(members)}: {members}"
                
                for array_name, expected_data in test_data.items():
                    loaded = npk.load(array_name)
                    assert np.allclose(loaded, expected_data)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_concurrent_append_same_array(self):
        """Test multiple threads appending to the same array."""
        num_threads = 5
        rows_per_thread = 20
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # Initialize array
            initial_data = np.random.rand(100, 10).astype(np.float32)
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': initial_data})
            
            # Multi-threaded append (requires lock to serialize append operations)
            append_lock = threading.Lock()
            
            def append_worker(thread_id):
                data = np.ones((rows_per_thread, 10), dtype=np.float32) * thread_id
                
                with append_lock:  # Serialize append operations
                    with NumPack(numpack_dir) as npk:
                        npk.append({'data': data})
                return True
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(append_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some appends failed"
            
            # Verify final size
            with NumPack(numpack_dir) as npk:
                final_shape = npk.get_shape('data')
                expected_rows = 100 + num_threads * rows_per_thread
                assert final_shape == (expected_rows, 10)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_concurrent_drop_different_rows(self):
        """Test multiple threads dropping different rows."""
        numpack_dir = tempfile.mkdtemp()
        num_threads = 5
        
        try:
            # Create sufficiently large array
            initial_data = np.arange(1000).reshape(100, 10).astype(np.float32)
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': initial_data})
            
            # Multi-threaded drop of different rows (requires lock to serialize)
            drop_lock = threading.Lock()
            
            def drop_worker(thread_id):
                # Each thread drops a different index range
                start_idx = thread_id * 10
                end_idx = start_idx + 5
                indices = list(range(start_idx, end_idx))
                
                with drop_lock:  # Serialize drop operations
                    with NumPack(numpack_dir) as npk:
                        npk.drop('data', indices)
                return True
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(drop_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some drops failed"
            
            # Verify drop results
            with NumPack(numpack_dir) as npk:
                final_shape = npk.get_shape('data')
                expected_rows = 100 - (num_threads * 5)
                assert final_shape == (expected_rows, 10)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_concurrent_replace_operations(self):
        """Test multiple threads performing replace operations."""
        numpack_dir = tempfile.mkdtemp()
        num_threads = 10
        
        try:
            # Initialize array
            initial_data = np.zeros((100, 10), dtype=np.float32)
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': initial_data})
            
            # Multi-threaded replace of different rows
            replace_lock = threading.Lock()
            
            def replace_worker(thread_id):
                # Each thread replaces a different row
                index = thread_id
                replacement = np.ones((1, 10), dtype=np.float32) * (thread_id + 1)
                
                with replace_lock:  # Serialize replace operations
                    with NumPack(numpack_dir) as npk:
                        npk.replace({'data': replacement}, index)
                return True
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(replace_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some replaces failed"
            
            # Verify replace results
            with NumPack(numpack_dir) as npk:
                loaded = npk.load('data')
                for i in range(num_threads):
                    expected_value = i + 1
                    assert np.allclose(loaded[i], expected_value)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_concurrent_mixed_operations(self):
        """Test mixed concurrent operations (read, write, update)."""
        numpack_dir = tempfile.mkdtemp()
        num_operations = 30
        
        try:
            # Initialize multiple arrays
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                for i in range(10):
                    data = np.random.rand(100, 10).astype(np.float32)
                    npk.save({f'array_{i}': data})
            
            operation_lock = threading.Lock()
            
            def mixed_worker(op_id):
                op_type = op_id % 3
                array_id = op_id % 10
                array_name = f'array_{array_id}'
                
                try:
                    with NumPack(numpack_dir) as npk:
                        if op_type == 0:  # Read
                            loaded = npk.load(array_name)
                            return ('read', loaded.shape[0] > 0)
                        elif op_type == 1:  # Get shape
                            shape = npk.get_shape(array_name)
                            return ('shape', shape[1] == 10)
                        else:  # Get member list
                            members = npk.get_member_list()
                            return ('list', len(members) == 10)
                except Exception as e:
                    return ('error', False)
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(mixed_worker, i) for i in range(num_operations)]
                results = [f.result() for f in as_completed(futures)]
            
            # All operations should succeed
            for op_type, success in results:
                assert success, f"Operation {op_type} failed"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_concurrent_lazy_load(self):
        """Test multi-threaded lazy loading."""
        numpack_dir = tempfile.mkdtemp()
        num_threads = 10
        
        try:
            # Create test data
            test_data = np.random.rand(1000, 10).astype(np.float32)
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # Multi-threaded lazy load
            def lazy_load_worker(thread_id):
                with NumPack(numpack_dir) as npk:
                    lazy_array = npk.load('data', lazy=True)
                    # Access some data
                    row = lazy_array[thread_id % 100]
                    return row.shape == (10,)
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(lazy_load_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some lazy loads failed"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_concurrent_batch_mode(self):
        """Test multi-threaded use of batch_mode."""
        numpack_dir = tempfile.mkdtemp()
        
        try:
            # Initialize data
            test_data = np.random.rand(100, 10).astype(np.float32)
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # Note: batch_mode itself is not thread-safe; each thread should have its own NumPack instance
            def batch_mode_worker(thread_id):
                try:
                    with NumPack(numpack_dir) as npk:
                        with npk.batch_mode():
                            loaded = npk.load('data')
                            # Simple verification
                            return loaded.shape == (100, 10)
                except Exception as e:
                    return False
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(batch_mode_worker, i) for i in range(5)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some batch mode operations failed"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_concurrent_getitem(self):
        """Test multi-threaded getitem access."""
        numpack_dir = tempfile.mkdtemp()
        num_threads = 10
        
        try:
            # Create test data
            test_data = np.arange(1000).reshape(100, 10).astype(np.float32)
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # Multi-threaded getitem
            def getitem_worker(thread_id):
                with NumPack(numpack_dir) as npk:
                    # Access different indices
                    indices = [thread_id, (thread_id + 10) % 100]
                    items = npk.getitem('data', indices)
                    return items.shape == (2, 10)
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(getitem_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some getitem operations failed"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_thread_safety_with_context_manager(self):
        """Test thread safety when using context manager."""
        numpack_dir = tempfile.mkdtemp()
        num_threads = 10
        
        try:
            # Initialize
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                initial_data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': initial_data})
            
            # Multi-threaded use of context manager
            def context_worker(thread_id):
                try:
                    with NumPack(numpack_dir) as npk:
                        # Read
                        loaded = npk.load('data')
                        # Get shape
                        shape = npk.get_shape('data')
                        # Verify
                        return shape == (100, 10) and loaded.shape == shape
                except Exception as e:
                    return False
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(context_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some context manager operations failed"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_multiple_numpack_instances_same_file(self):
        """Test multiple NumPack instances accessing the same file simultaneously."""
        numpack_dir = tempfile.mkdtemp()
        num_instances = 5
        
        try:
            # Initialize data
            test_data = np.random.rand(100, 10).astype(np.float32)
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # Multiple instances reading simultaneously
            def instance_worker(instance_id):
                try:
                    # Each thread creates its own NumPack instance
                    with NumPack(numpack_dir) as npk:
                        loaded = npk.load('data')
                        return np.allclose(loaded, test_data)
                except Exception as e:
                    return False
            
            with ThreadPoolExecutor(max_workers=num_instances) as executor:
                futures = [executor.submit(instance_worker, i) for i in range(num_instances)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some instances failed to read correctly"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_concurrent_stream_load(self):
        """Test multi-threaded use of stream_load."""
        numpack_dir = tempfile.mkdtemp()
        num_threads = 5
        
        try:
            # Create test data
            test_data = np.arange(1000).reshape(100, 10).astype(np.float32)
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # Multi-threaded stream_load
            def stream_load_worker(thread_id):
                try:
                    with NumPack(numpack_dir) as npk:
                        count = 0
                        for chunk in npk.stream_load('data', buffer_size=10):
                            count += chunk.shape[0]
                        return count == 100
                except Exception as e:
                    return False
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(stream_load_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            assert all(results), "Some stream_load operations failed"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


class TestThreadSafetyStress:
    """Stress test thread safety."""
    
    def test_high_concurrency_reads(self):
        """High concurrency read stress test."""
        numpack_dir = tempfile.mkdtemp()
        num_threads = 50
        iterations_per_thread = 10
        
        try:
            # Create test data
            test_data = np.random.rand(1000, 10).astype(np.float32)
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': test_data})
            
            # High concurrency reads
            def stress_read_worker(thread_id):
                success_count = 0
                for i in range(iterations_per_thread):
                    try:
                        with NumPack(numpack_dir) as npk:
                            loaded = npk.load('data')
                            if loaded.shape == test_data.shape:
                                success_count += 1
                    except Exception:
                        pass
                return success_count
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(stress_read_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            # At least most operations should succeed
            total_operations = num_threads * iterations_per_thread
            successful_operations = sum(results)
            success_rate = successful_operations / total_operations
            
            assert success_rate > 0.95, f"Success rate too low: {success_rate}"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_rapid_open_close(self):
        """Rapid open-close stress test."""
        numpack_dir = tempfile.mkdtemp()
        num_threads = 20
        
        try:
            # Initialize data
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                npk.save({'data': np.random.rand(100, 10).astype(np.float32)})
            
            # Rapid open-close
            def rapid_open_close_worker(thread_id):
                success_count = 0
                for i in range(5):
                    try:
                        with NumPack(numpack_dir) as npk:
                            members = npk.get_member_list()
                            if 'data' in members:
                                success_count += 1
                    except Exception:
                        pass
                return success_count
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(rapid_open_close_worker, i) for i in range(num_threads)]
                results = [f.result() for f in as_completed(futures)]
            
            # Most operations should succeed
            assert sum(results) >= num_threads * 4  # At least 80% success
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

