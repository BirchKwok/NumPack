import os
import sys
import time
import logging
import numpy as np
from functools import wraps
from numpack import NumPack

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_file_when_finished(*filenames):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            finally:
                for filename in filenames:
                    try:
                        if os.path.isdir(filename):
                            for f in os.listdir(filename):
                                os.remove(os.path.join(filename, f))
                            os.rmdir(filename)
                        else:
                            os.remove(filename)
                        logger.info(f"Clean test file: {filename}")
                    except FileNotFoundError:
                        pass
        return wrapper
    return decorator

@clean_file_when_finished('test_large', 'test_large.npz', 'test_large_array1.npy', 'test_large_array2.npy')
def test_large_data():
    """Test large data processing"""
    logger.info("=== Test large data processing ===")
    
    try:
        # Create large data
        size = 1000000  
        arrays = {
            'array1': np.random.rand(size, 10).astype(np.float32),
            'array2': np.random.rand(size // 2, 5).astype(np.float32)
        }
        
        # Test NumPack save
        logger.info(f"Test NumPack save large arrays (array1: {arrays['array1'].shape}, array2: {arrays['array2'].shape})...")
        start_time = time.time()
        npk = NumPack('test_large', drop_if_exists=True)
        npk.save(arrays)
        save_time = time.time() - start_time
        logger.info(f"NumPack save time: {save_time:.2f} seconds")
        
        # Test NumPy npz save
        logger.info("Test NumPy npz save...")
        start_time = time.time()
        np.savez('test_large.npz', **arrays)
        npz_save_time = time.time() - start_time
        logger.info(f"NumPy npz save time: {npz_save_time:.2f} seconds")
        logger.info(f"Save performance comparison (npz): NumPack/NumPy = {save_time/npz_save_time:.2f}x")
        
        # Test NumPy npy save
        logger.info("Test NumPy npy save...")
        start_time = time.time()
        np.save('test_large_array1.npy', arrays['array1'])
        np.save('test_large_array2.npy', arrays['array2'])
        npy_save_time = time.time() - start_time
        logger.info(f"NumPy npy save time: {npy_save_time:.2f} seconds")
        logger.info(f"Save performance comparison (npy): NumPack/NumPy = {save_time/npy_save_time:.2f}x")
        
        logger.info("\n\nTest NumPack full load...")
        start_time = time.time()
        loaded = npk.load(mmap_mode=False)
        _, _ = loaded['array1'], loaded['array2']
        load_time = time.time() - start_time
        logger.info(f"NumPack load time: {load_time:.2f} seconds")
        
        logger.info("Test NumPack selective load...")
        start_time = time.time()
        loaded_partial = npk.load(mmap_mode=False)['array1']
        load_partial_time = time.time() - start_time
        logger.info(f"NumPack selective load time: {load_partial_time:.2f} seconds")
        
        # Test NumPy npz load
        logger.info("Test NumPy npz load...")
        start_time = time.time()
        npz_loaded = np.load('test_large.npz')
        _, _ = npz_loaded['array1'], npz_loaded['array2']
        npz_load_time = time.time() - start_time
        logger.info(f"NumPy npz load time: {npz_load_time:.2f} seconds")
        logger.info(f"Load performance comparison (npz): NumPack/NumPy = {load_time/npz_load_time:.2f}x")
        
        # Test NumPy npz selective load
        logger.info("Test NumPy npz selective load...")
        start_time = time.time()
        npz_loaded = np.load('test_large.npz')
        npz_array1 = npz_loaded['array1']
        npz_array1_load_time = time.time() - start_time
        logger.info(f"NumPy npz selective load single array time: {npz_array1_load_time:.2f} seconds")
        logger.info(f"Selective load performance comparison (npz): NumPack/NumPy = {load_partial_time/npz_array1_load_time:.2f}x")
        
        # Test NumPy npy load
        logger.info("\n\nTest NumPy npy load...")
        start_time = time.time()
        npy_loaded = {
            'array1': np.load('test_large_array1.npy'),
            'array2': np.load('test_large_array2.npy')
        }
        npy_load_time = time.time() - start_time
        logger.info(f"NumPy npy load time: {npy_load_time:.2f} seconds")
        logger.info(f"Load performance comparison (npy): NumPack/NumPy = {load_time/npy_load_time:.2f}x")
        
        # Test NumPack mmap load
        logger.info("\n\nTest NumPack mmap load...")
        start_time = time.time()
        lazy_loaded = npk.load(mmap_mode=True)
        _, _ = lazy_loaded['array1'], lazy_loaded['array2']
        lazy_load_time = time.time() - start_time
        logger.info(f"NumPack mmap load time: {lazy_load_time:.2f} seconds")
        logger.info(f"Mmap load performance comparison (npy): NumPack/NumPy = {lazy_load_time/npy_load_time:.2f}x")
        
        # Test NumPy npz mmap load
        logger.info("Test NumPy npz mmap load...")
        start_time = time.time()
        npz_mmap = np.load('test_large.npz', mmap_mode='r')
        _, _ = npz_mmap['array1'], npz_mmap['array2']
        npz_mmap_time = time.time() - start_time
        logger.info(f"NumPy npz mmap load time: {npz_mmap_time:.2f} seconds")
        logger.info(f"Mmap load performance comparison (npz): NumPack/NumPy = {lazy_load_time/npz_mmap_time:.2f}x")
        
        logger.info("Test NumPy npy mmap load...")
        start_time = time.time()
        npy_mmap = {
            'array1': np.load('test_large_array1.npy', mmap_mode='r'),
            'array2': np.load('test_large_array2.npy', mmap_mode='r')
        }
        npy_mmap_time = time.time() - start_time
        logger.info(f"NumPy npy mmap load time: {npy_mmap_time:.2f} seconds")
        logger.info(f"Mmap load performance comparison (npy): NumPack/NumPy = {lazy_load_time/npy_mmap_time:.2f}x")
        
        # Verify NumPack data
        for name, array in arrays.items():
            assert np.allclose(array, loaded[name])
            logger.info(f"NumPack array '{name}' verified")
            
        # Verify NumPy npz data
        for name, array in arrays.items():
            assert np.allclose(array, npz_loaded[name])
            logger.info(f"NumPy npz array '{name}' verified")
            
        # Verify NumPy npy data
        for name, array in arrays.items():
            assert np.allclose(array, npy_loaded[name])
            logger.info(f"NumPy npy array '{name}' verified")
        
        # Compare file size
        npk_size = sum(os.path.getsize(os.path.join('test_large', f)) for f in os.listdir('test_large')) / (1024 * 1024)  # MB
        npz_size = os.path.getsize('test_large.npz') / (1024 * 1024)  # MB
        npy_size = sum(os.path.getsize(f'test_large_{name}.npy') / (1024 * 1024) 
                      for name in ['array1', 'array2'])  # MB
        logger.info(f"\nFile size comparison:")
        logger.info(f"NumPack: {npk_size:.2f} MB")
        logger.info(f"NumPy npz: {npz_size:.2f} MB")
        logger.info(f"NumPy npy: {npy_size:.2f} MB")
        logger.info(f"Size comparison (vs npz): NumPack/NumPy = {npk_size/npz_size:.2f}x")
        logger.info(f"Size comparison (vs npy): NumPack/NumPy = {npk_size/npy_size:.2f}x")
        
        logger.info("Large data test completed")
        
    except Exception as e:
        logger.error(f"Large data test failed: {str(e)}")
        raise

@clean_file_when_finished('test_append', 'test_append.npz')
def test_append_operations():
    """Test append operations"""
    logger.info("=== Test append operations ===")
    
    try:
        # Create initial data
        size = 1000000
        arrays = {
            'array1': np.random.rand(size, 10).astype(np.float32),
            'array2': np.random.rand(size // 2, 5).astype(np.float32)
        }
        
        # Save initial data
        npk = NumPack('test_append', drop_if_exists=True)
        npk.save(arrays)
        np.savez('test_append.npz', **arrays)
        
        # Create data to append
        append_data = {
            'array3': np.random.rand(size // 4, 8).astype(np.float32),
            'array4': np.random.rand(size // 8, 3).astype(np.float32)
        }
        
        # Test NumPack append
        logger.info("Test NumPack append...")
        start_time = time.time()
        npk.save(append_data)
        append_time = time.time() - start_time
        logger.info(f"NumPack append time: {append_time:.2f} seconds")
        
        # NumPy npz does not support append, so the entire file needs to be saved again
        logger.info("Test NumPy npz append...")
        npz_data = dict(np.load('test_append.npz'))
        npz_data.update(append_data)
        start_time = time.time()
        np.savez('test_append.npz', **npz_data)
        npz_append_time = time.time() - start_time
        logger.info(f"NumPy npz append time: {npz_append_time:.2f} seconds")
        logger.info(f"Append performance comparison: NumPack/NumPy = {append_time/npz_append_time:.2f}x")
        
        # Load and verify
        loaded = npk.load(mmap_mode=False)
        
        # Verify original data
        for name, array in arrays.items():
            assert np.allclose(array, loaded[name])
            logger.info(f"Original array '{name}' verified")
        
        # Verify appended data
        for name, array in append_data.items():
            assert np.allclose(array, loaded[name])
            logger.info(f"Appended array '{name}' verified")
        
        logger.info("Append operations test completed")
        
    except Exception as e:
        logger.error(f"Append operations test failed: {str(e)}")
        raise

@clean_file_when_finished('test_random_access', 'test_random_access.npz', 'test_random_access_array1.npy', 'test_random_access_array2.npy')
def test_random_access():
    """Test random access performance"""
    logger.info("=== Test random access performance ===")
    
    try:
        # Create test data
        size = 1000000
        arrays = {
            'array1': np.random.rand(size, 10).astype(np.float32),
            'array2': np.random.rand(size, 5).astype(np.float32)
        }
        
        npk = NumPack('test_random_access', drop_if_exists=True)    
        npk.save(arrays)
        np.savez('test_random_access.npz', **arrays)
        np.save('test_random_access_array1.npy', arrays['array1'])
        np.save('test_random_access_array2.npy', arrays['array2'])
        
        random_indices = np.random.randint(0, size, 10000).tolist()
        logger.info("\nTest full random access performance...")
        
        start_time = time.time()
        numpack_random = {
            'array1': npk.getitem(random_indices, "array1"),
            'array2': npk.getitem(random_indices, "array2")
        }
        numpack_random_time = time.time() - start_time
        logger.info(f"NumPack random access time: {numpack_random_time:.2f} seconds")
        
        # NumPy npz random access
        start_time = time.time()
        npz_data = np.load('test_random_access.npz', mmap_mode='r')
        npz_random = {
            'array1': npz_data['array1'][random_indices],
            'array2': npz_data['array2'][random_indices]
        }
        npz_random_time = time.time() - start_time
        logger.info(f"NumPy npz random access time: {npz_random_time:.2f} seconds")
        logger.info(f"Random access performance comparison (npz): NumPack/NumPy = {numpack_random_time/npz_random_time:.2f}x")
        
        # NumPy npy random access
        start_time = time.time()
        npy_random = {
            'array1': np.load('test_random_access_array1.npy', mmap_mode='r')[random_indices],
            'array2': np.load('test_random_access_array2.npy', mmap_mode='r')[random_indices]
        }
        npy_random_time = time.time() - start_time
        logger.info(f"NumPy npy random access time: {npy_random_time:.2f} seconds")
        logger.info(f"Random access performance comparison (npy): NumPack/NumPy = {numpack_random_time/npy_random_time:.2f}x")
        
        for name in arrays:
            assert np.allclose(numpack_random[name], npz_random[name])
            assert np.allclose(numpack_random[name], npy_random[name])
            logger.info(f"Random access array '{name}' verified")
        
        logger.info("Random access performance test completed")
        
    except Exception as e:
        logger.error(f"Random access performance test failed: {str(e)}")
        raise

@clean_file_when_finished('test_replace', 'test_replace.npz', 'test_replace_array.npy')
def test_replace_operations():
    """Test replace operations performance"""
    logger.info("=== Test replace operations performance ===")
    
    try:
        # Create test data
        size = 1000000
        array = np.random.rand(size, 10).astype(np.float32)
        
        # Save initial data
        npk = NumPack('test_replace', drop_if_exists=True)
        npk.save({'array': array})
        np.savez('test_replace.npz', array=array)
        np.save('test_replace_array.npy', array)
        
        # Test scenario 1: Replace single row
        logger.info("\nTest scenario 1: Replace single row")
        single_row = np.random.rand(1, 10).astype(np.float32)
        idx = size // 2  # Replace middle row
        
        # NumPack replace
        start_time = time.time()
        npk.replace({'array': single_row}, [idx])
        replace_time = time.time() - start_time
        logger.info(f"NumPack single row replace time: {replace_time:.2f} seconds")
        
        # NumPy npz replace
        start_time = time.time()
        npz_data = dict(np.load('test_replace.npz'))
        npz_data['array'][idx] = single_row
        np.savez('test_replace.npz', **npz_data)
        npz_replace_time = time.time() - start_time
        logger.info(f"NumPy npz single row replace time: {npz_replace_time:.2f} seconds")
        logger.info(f"Single row replace performance comparison (npz): NumPack/NumPy = {replace_time/npz_replace_time:.2f}x")
        
        # NumPy npy replace
        start_time = time.time()
        npy_data = np.load('test_replace_array.npy')
        npy_data[idx] = single_row
        np.save('test_replace_array.npy', npy_data)
        npy_replace_time = time.time() - start_time
        logger.info(f"NumPy npy single row replace time: {npy_replace_time:.2f} seconds")
        logger.info(f"Single row replace performance comparison (npy): NumPack/NumPy = {replace_time/npy_replace_time:.2f}x")
        
        # Test scenario 2: Replace continuous rows
        logger.info("\nTest scenario 2: Replace continuous rows")
        continuous_rows = 10000  # Replace 10,000 rows
        multi_rows = np.random.rand(continuous_rows, 10).astype(np.float32)
        start_idx = size // 4
        
        # NumPack replace
        start_time = time.time()
        npk.replace({'array': multi_rows}, slice(start_idx, start_idx + continuous_rows))
        replace_time = time.time() - start_time
        logger.info(f"NumPack continuous rows replace time: {replace_time:.2f} seconds")
        
        # NumPy npz replace
        start_time = time.time()
        npz_data = dict(np.load('test_replace.npz'))
        npz_data['array'][start_idx:start_idx + continuous_rows] = multi_rows
        np.savez('test_replace.npz', **npz_data)
        npz_replace_time = time.time() - start_time
        logger.info(f"NumPy npz continuous rows replace time: {npz_replace_time:.2f} seconds")
        logger.info(f"Continuous rows replace performance comparison (npz): NumPack/NumPy = {replace_time/npz_replace_time:.2f}x")
        
        # NumPy npy replace
        start_time = time.time()
        npy_data = np.load('test_replace_array.npy')
        npy_data[start_idx:start_idx + continuous_rows] = multi_rows
        np.save('test_replace_array.npy', npy_data)
        npy_replace_time = time.time() - start_time
        logger.info(f"NumPy npy continuous rows replace time: {npy_replace_time:.2f} seconds")
        logger.info(f"Continuous rows replace performance comparison (npy): NumPack/NumPy = {replace_time/npy_replace_time:.2f}x")
        
        # Test scenario 3: Replace random distributed rows
        logger.info("\nTest scenario 3: Replace random distributed rows")
        random_count = 10000  # Replace 10,000 rows
        random_rows = np.random.rand(random_count, 10).astype(np.float32)
        random_indices = np.random.choice(size, random_count, replace=False)
        
        # NumPack replace
        start_time = time.time()
        npk.replace({'array': random_rows}, random_indices.tolist())
        replace_time = time.time() - start_time
        logger.info(f"NumPack random distributed rows replace time: {replace_time:.2f} seconds")
        
        # NumPy npz replace
        start_time = time.time()
        npz_data = dict(np.load('test_replace.npz'))
        npz_data['array'][random_indices] = random_rows
        np.savez('test_replace.npz', **npz_data)
        npz_replace_time = time.time() - start_time
        logger.info(f"NumPy npz random distributed rows replace time: {npz_replace_time:.2f} seconds")
        logger.info(f"Random distributed rows replace performance comparison (npz): NumPack/NumPy = {replace_time/npz_replace_time:.2f}x")
        
        start_time = time.time()
        npy_data = np.load('test_replace_array.npy')
        npy_data[random_indices] = random_rows
        np.save('test_replace_array.npy', npy_data)
        npy_replace_time = time.time() - start_time
        logger.info(f"NumPy npy random distributed rows replace time: {npy_replace_time:.2f} seconds")
        logger.info(f"Random distributed rows replace performance comparison (npy): NumPack/NumPy = {replace_time/npy_replace_time:.2f}x")
        
        # Test scenario 4: Replace large data
        logger.info("\nTest scenario 4: Replace large data")
        large_size = size // 2  # Replace 500,000 rows
        large_rows = np.random.rand(large_size, 10).astype(np.float32)
        
        # NumPack replace
        start_time = time.time()
        npk.replace({'array': large_rows}, slice(0, large_size))
        replace_time = time.time() - start_time
        logger.info(f"NumPack large data replace time: {replace_time:.2f} seconds")
        
        # NumPy npz replace
        start_time = time.time()
        npz_data = dict(np.load('test_replace.npz'))
        npz_data['array'][:large_size] = large_rows
        np.savez('test_replace.npz', **npz_data)
        npz_replace_time = time.time() - start_time
        logger.info(f"NumPy npz large data replace time: {npz_replace_time:.2f} seconds")
        logger.info(f"Large data replace performance comparison (npz): NumPack/NumPy = {replace_time/npz_replace_time:.2f}x")
        
        # NumPy npy replace
        start_time = time.time()
        npy_data = np.load('test_replace_array.npy')
        npy_data[:large_size] = large_rows
        np.save('test_replace_array.npy', npy_data)
        npy_replace_time = time.time() - start_time
        logger.info(f"NumPy npy large data replace time: {npy_replace_time:.2f} seconds")
        logger.info(f"Large data replace performance comparison (npy): NumPack/NumPy = {replace_time/npy_replace_time:.2f}x")
        
        # Verify data correctness
        loaded = npk.load(mmap_mode=False)['array']
        npz_loaded = np.load('test_replace.npz')['array']
        npy_loaded = np.load('test_replace_array.npy')
        
        assert np.allclose(loaded[:large_size], large_rows)
        assert np.allclose(npz_loaded[:large_size], large_rows)
        assert np.allclose(npy_loaded[:large_size], large_rows)
        logger.info("Data verification passed")
        
        logger.info("Replace operations test completed")
        
    except Exception as e:
        logger.error(f"Replace operations test failed: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        logger.info("Starting performance tests...")
        test_large_data()
        test_append_operations()
        test_random_access()
        test_replace_operations()
        logger.info("All tests completed!")
    except Exception as e:
        logger.error(f"Error occurred during tests: {str(e)}")
        sys.exit(1) 