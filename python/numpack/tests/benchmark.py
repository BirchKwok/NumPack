import os
import sys
import time
import logging
import numpy as np
from functools import wraps
from numpack import NumPack
from typing import Dict, List, Callable
from statistics import mean

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
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
                    except FileNotFoundError:
                        pass
        return wrapper
    return decorator

class TimingStats:
    def __init__(self):
        self.times: Dict[str, List[float]] = {}
    
    def add_time(self, operation: str, time_taken: float):
        if operation not in self.times:
            self.times[operation] = []
        self.times[operation].append(time_taken)
    
    def get_average(self, operation: str) -> float:
        return mean(self.times[operation])
    
    def get_all_times(self, operation: str) -> List[float]:
        return self.times[operation]
    
    def get_summary(self) -> str:
        summary = []
        max_op_len = max(len(op) for op in self.times.keys())
        
        for operation in sorted(self.times.keys()):
            times = self.get_all_times(operation)
            avg_time = self.get_average(operation)
            min_time = min(times)
            max_time = max(times)
            std_dev = np.std(times)
            
            summary.append(
                f"{operation:<{max_op_len}} | "
                f"avg: {avg_time:6.3f}s | "
                f"min: {min_time:6.3f}s | "
                f"max: {max_time:6.3f}s | "
                f"std: {std_dev:6.3f}s"
            )
        
        return "\n".join(summary)

def run_multiple_times(runs: int = 7):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timing_stats = TimingStats()
            
            for i in range(runs):
                result = func(*args, timing_stats=timing_stats, **kwargs)
            
            # Print summary
            logger.info(f"\n{func.__name__} Performance Summary:")
            logger.info("-" * 80)
            logger.info(timing_stats.get_summary())
            logger.info("-" * 80)
            
            return result
        return wrapper
    return decorator

@run_multiple_times(runs=7)
@clean_file_when_finished('test_large', 'test_large.npz', 'test_large_array1.npy', 'test_large_array2.npy')
def test_large_data(timing_stats: TimingStats):
    """Test large data processing"""
    try:
        # Create large data
        size = 1000000  
        arrays = {
            'array1': np.random.rand(size, 10).astype(np.float32),
            'array2': np.random.rand(size // 2, 5).astype(np.float32)
        }
        
        # Test NumPack save
        start_time = time.time()
        npk = NumPack('test_large', drop_if_exists=True)
        npk.save(arrays)
        save_time = time.time() - start_time
        timing_stats.add_time("NumPack save", save_time)
        
        # Test NumPy npz save
        start_time = time.time()
        np.savez('test_large.npz', **arrays)
        npz_save_time = time.time() - start_time
        timing_stats.add_time("NumPy npz save", npz_save_time)
        
        # Test NumPy npy save
        start_time = time.time()
        np.save('test_large_array1.npy', arrays['array1'])
        np.save('test_large_array2.npy', arrays['array2'])
        npy_save_time = time.time() - start_time
        timing_stats.add_time("NumPy npy save", npy_save_time)
        
        # Test NumPack full load
        start_time = time.time()
        arr1 = npk.load('array1')
        arr2 = npk.load('array2')
        load_time = time.time() - start_time
        timing_stats.add_time("NumPack load", load_time)
        
        # Test NumPack selective load
        start_time = time.time()
        loaded_partial = npk.load('array1')
        load_partial_time = time.time() - start_time
        timing_stats.add_time("NumPack selective load", load_partial_time)
        
        # Test NumPy npz load
        start_time = time.time()
        npz_loaded = np.load('test_large.npz')
        _, _ = npz_loaded['array1'], npz_loaded['array2']
        npz_load_time = time.time() - start_time
        timing_stats.add_time("NumPy npz load", npz_load_time)
        
        # Test NumPy npz selective load
        start_time = time.time()
        npz_loaded = np.load('test_large.npz')
        npz_array1 = npz_loaded['array1']
        npz_array1_load_time = time.time() - start_time
        timing_stats.add_time("NumPy npz selective load", npz_array1_load_time)
        
        # Test NumPy npy load
        start_time = time.time()
        npy_loaded = {
            'array1': np.load('test_large_array1.npy'),
            'array2': np.load('test_large_array2.npy')
        }
        npy_load_time = time.time() - start_time
        timing_stats.add_time("NumPy npy load", npy_load_time)
        
        # Test NumPack mmap load
        start_time = time.time()
        with npk.mmap_mode() as mmap_npk:
            lazy_arr1 = mmap_npk.load('array1')
            lazy_arr2 = mmap_npk.load('array2')
        lazy_load_time = time.time() - start_time
        timing_stats.add_time("NumPack mmap load", lazy_load_time)
        
        # Test NumPy npz mmap load
        start_time = time.time()
        npz_mmap = np.load('test_large.npz', mmap_mode='r')
        _, _ = npz_mmap['array1'], npz_mmap['array2']
        npz_mmap_time = time.time() - start_time
        timing_stats.add_time("NumPy npz mmap load", npz_mmap_time)
        
        # Test NumPy npy mmap load
        start_time = time.time()
        npy_mmap = {
            'array1': np.load('test_large_array1.npy', mmap_mode='r'),
            'array2': np.load('test_large_array2.npy', mmap_mode='r')
        }
        npy_mmap_time = time.time() - start_time
        timing_stats.add_time("NumPy npy mmap load", npy_mmap_time)
        
        # Verify data
        for name, array in arrays.items():
            if name == 'array1':
                assert np.allclose(array, arr1)
            else:
                assert np.allclose(array, arr2)
            assert np.allclose(array, npz_loaded[name])
            assert np.allclose(array, npy_loaded[name])
        
        # Compare file size
        npk_size = sum(os.path.getsize(os.path.join('test_large', f)) for f in os.listdir('test_large')) / (1024 * 1024)  # MB
        npz_size = os.path.getsize('test_large.npz') / (1024 * 1024)  # MB
        npy_size = sum(os.path.getsize(f'test_large_{name}.npy') / (1024 * 1024) 
                      for name in ['array1', 'array2'])  # MB
        timing_stats.add_time("File size (MB) - NumPack", npk_size)
        timing_stats.add_time("File size (MB) - NumPy npz", npz_size)
        timing_stats.add_time("File size (MB) - NumPy npy", npy_size)
        
    except Exception as e:
        logger.error(f"Large data test failed: {str(e)}")
        raise

@run_multiple_times(runs=7)
@clean_file_when_finished('test_append', 'test_append.npz')
def test_append_operations(timing_stats: TimingStats):
    """Test append operations"""
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
        start_time = time.time()
        npk.save(append_data)
        append_time = time.time() - start_time
        timing_stats.add_time("NumPack append", append_time)
        
        # NumPy npz append
        npz_data = dict(np.load('test_append.npz'))
        npz_data.update(append_data)
        start_time = time.time()
        np.savez('test_append.npz', **npz_data)
        npz_append_time = time.time() - start_time
        timing_stats.add_time("NumPy npz append", npz_append_time)
        
        # Verify data
        npz_data = dict(np.load('test_append.npz'))
        for name in {**arrays, **append_data}:
            if name in arrays:
                assert np.allclose(arrays[name], npk.load(name))
            else:
                assert np.allclose(append_data[name], npk.load(name))
        
    except Exception as e:
        logger.error(f"Append operations test failed: {str(e)}")
        raise

@run_multiple_times(runs=7)
@clean_file_when_finished('test_random_access', 'test_random_access.npz', 'test_random_access_array1.npy', 'test_random_access_array2.npy')
def test_random_access(timing_stats: TimingStats):
    """Test random access performance"""
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
        
        # NumPack random access
        start_time = time.time()
        numpack_random = {
            'array1': npk.getitem("array1", random_indices),
            'array2': npk.getitem("array2", random_indices)
        }
        numpack_random_time = time.time() - start_time
        timing_stats.add_time("NumPack random access", numpack_random_time)
        
        # NumPy npz random access
        start_time = time.time()
        npz_data = np.load('test_random_access.npz', mmap_mode='r')
        npz_random = {
            'array1': npz_data['array1'][random_indices],
            'array2': npz_data['array2'][random_indices]
        }
        npz_random_time = time.time() - start_time
        timing_stats.add_time("NumPy npz random access", npz_random_time)
        
        # NumPy npy random access
        start_time = time.time()
        npy_random = {
            'array1': np.load('test_random_access_array1.npy', mmap_mode='r')[random_indices],
            'array2': np.load('test_random_access_array2.npy', mmap_mode='r')[random_indices]
        }
        npy_random_time = time.time() - start_time
        timing_stats.add_time("NumPy npy random access", npy_random_time)
        
        # Verify data
        for name in arrays:
            assert np.allclose(numpack_random[name], npz_random[name])
            assert np.allclose(numpack_random[name], npy_random[name])
        
    except Exception as e:
        logger.error(f"Random access performance test failed: {str(e)}")
        raise

@run_multiple_times(runs=7)
@clean_file_when_finished('test_replace', 'test_replace.npz', 'test_replace_array.npy')
def test_replace_operations(timing_stats: TimingStats):
    """Test replace operations performance"""
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
        single_row = np.random.rand(1, 10).astype(np.float32)
        idx = size // 2  # Replace middle row
        
        # NumPack replace
        start_time = time.time()
        npk.replace({'array': single_row}, [idx])
        replace_time = time.time() - start_time
        timing_stats.add_time("NumPack single row replace", replace_time)
        
        # NumPy npz replace
        start_time = time.time()
        npz_data = dict(np.load('test_replace.npz'))
        npz_data['array'][idx] = single_row
        np.savez('test_replace.npz', **npz_data)
        npz_replace_time = time.time() - start_time
        timing_stats.add_time("NumPy npz single row replace", npz_replace_time)
        
        # NumPy npy replace
        start_time = time.time()
        npy_data = np.load('test_replace_array.npy')
        npy_data[idx] = single_row
        np.save('test_replace_array.npy', npy_data)
        npy_replace_time = time.time() - start_time
        timing_stats.add_time("NumPy npy single row replace", npy_replace_time)
        
        # Test scenario 2: Replace continuous rows
        continuous_rows = 10000  # Replace 10,000 rows
        multi_rows = np.random.rand(continuous_rows, 10).astype(np.float32)
        start_idx = size // 4
        
        # NumPack replace
        start_time = time.time()
        npk.replace({'array': multi_rows}, slice(start_idx, start_idx + continuous_rows))
        replace_time = time.time() - start_time
        timing_stats.add_time("NumPack continuous rows replace", replace_time)
        
        # NumPy npz replace
        start_time = time.time()
        npz_data = dict(np.load('test_replace.npz'))
        npz_data['array'][start_idx:start_idx + continuous_rows] = multi_rows
        np.savez('test_replace.npz', **npz_data)
        npz_replace_time = time.time() - start_time
        timing_stats.add_time("NumPy npz continuous rows replace", npz_replace_time)
        
        # NumPy npy replace
        start_time = time.time()
        npy_data = np.load('test_replace_array.npy')
        npy_data[start_idx:start_idx + continuous_rows] = multi_rows
        np.save('test_replace_array.npy', npy_data)
        npy_replace_time = time.time() - start_time
        timing_stats.add_time("NumPy npy continuous rows replace", npy_replace_time)
        
        # Test scenario 3: Replace random distributed rows
        random_count = 10000  # Replace 10,000 rows
        random_rows = np.random.rand(random_count, 10).astype(np.float32)
        random_indices = np.random.choice(size, random_count, replace=False)
        
        # NumPack replace
        start_time = time.time()
        npk.replace({'array': random_rows}, random_indices.tolist())
        replace_time = time.time() - start_time
        timing_stats.add_time("NumPack random rows replace", replace_time)
        
        # NumPy npz replace
        start_time = time.time()
        npz_data = dict(np.load('test_replace.npz'))
        npz_data['array'][random_indices] = random_rows
        np.savez('test_replace.npz', **npz_data)
        npz_replace_time = time.time() - start_time
        timing_stats.add_time("NumPy npz random rows replace", npz_replace_time)
        
        # NumPy npy replace
        start_time = time.time()
        npy_data = np.load('test_replace_array.npy')
        npy_data[random_indices] = random_rows
        np.save('test_replace_array.npy', npy_data)
        npy_replace_time = time.time() - start_time
        timing_stats.add_time("NumPy npy random rows replace", npy_replace_time)
        
        # Test scenario 4: Replace large data
        large_size = size // 2  # Replace 500,000 rows
        large_rows = np.random.rand(large_size, 10).astype(np.float32)
        
        # NumPack replace
        start_time = time.time()
        npk.replace({'array': large_rows}, slice(0, large_size))
        replace_time = time.time() - start_time
        timing_stats.add_time("NumPack large data replace", replace_time)
        
        # NumPy npz replace
        start_time = time.time()
        npz_data = dict(np.load('test_replace.npz'))
        npz_data['array'][:large_size] = large_rows
        np.savez('test_replace.npz', **npz_data)
        npz_replace_time = time.time() - start_time
        timing_stats.add_time("NumPy npz large data replace", npz_replace_time)
        
        # NumPy npy replace
        start_time = time.time()
        npy_data = np.load('test_replace_array.npy')
        npy_data[:large_size] = large_rows
        np.save('test_replace_array.npy', npy_data)
        npy_replace_time = time.time() - start_time
        timing_stats.add_time("NumPy npy large data replace", npy_replace_time)
        
        # Verify data correctness
        loaded = npk.load('array')
        npz_loaded = np.load('test_replace.npz')['array']
        npy_loaded = np.load('test_replace_array.npy')
        
        assert np.allclose(loaded[:large_size], large_rows)
        assert np.allclose(npz_loaded[:large_size], large_rows)
        assert np.allclose(npy_loaded[:large_size], large_rows)
        
    except Exception as e:
        logger.error(f"Replace operations test failed: {str(e)}")
        raise

@run_multiple_times(runs=7)
@clean_file_when_finished('test_drop', 'test_drop.npz', 'test_drop_array1.npy', 'test_drop_array2.npy')
def test_drop_operations(timing_stats: TimingStats):
    """Test drop operations performance"""
    try:
        # Create test data
        size = 1000000
        arrays = {
            'array1': np.random.rand(size, 10).astype(np.float32),
            'array2': np.random.rand(size, 5).astype(np.float32)
        }
        
        # Save initial data
        npk = NumPack('test_drop', drop_if_exists=True)
        npk.save(arrays)
        np.savez('test_drop.npz', **arrays)
        np.save('test_drop_array1.npy', arrays['array1'])
        np.save('test_drop_array2.npy', arrays['array2'])
        
        # Test scenario 1: Drop entire array
        # NumPack drop array
        start_time = time.time()
        npk.drop('array1')
        drop_time = time.time() - start_time
        timing_stats.add_time("NumPack drop array", drop_time)

        # NumPy npz drop array (need to reload and save)
        start_time = time.time()
        npz_data = dict(np.load('test_drop.npz'))
        del npz_data['array1']
        np.savez('test_drop.npz', **npz_data)
        npz_drop_time = time.time() - start_time
        timing_stats.add_time("NumPy npz drop array", npz_drop_time)

        # NumPy npy drop array (just delete file)
        start_time = time.time()
        os.remove('test_drop_array1.npy')
        npy_drop_time = time.time() - start_time
        timing_stats.add_time("NumPy npy drop array", npy_drop_time)

        # Restore data for next test
        npk.save({'array1': arrays['array1']})
        np.savez('test_drop.npz', **arrays)
        np.save('test_drop_array1.npy', arrays['array1'])

        # Test scenario 2: Drop specific rows
        drop_indices = list(range(0, size, 2))  # Drop every other row
        
        # NumPack drop rows
        start_time = time.time()
        # npk.drop('array1', drop_indices)
        with npk.mmap_mode() as mmap_npk:
            mask = np.ones(size, dtype=bool)
            mask[drop_indices] = False
            _arrays = mmap_npk.load('array1')
            npk.save({'array1': _arrays[mask]})
        drop_rows_time = time.time() - start_time
        timing_stats.add_time("NumPack drop rows", drop_rows_time)

        # NumPy npz drop rows (need to reload, modify and save)
        start_time = time.time()
        npz_data = dict(np.load('test_drop.npz'))
        mask = np.ones(size, dtype=bool)
        mask[drop_indices] = False
        npz_data['array1'] = npz_data['array1'][mask]
        np.savez('test_drop.npz', **npz_data)
        npz_drop_rows_time = time.time() - start_time
        timing_stats.add_time("NumPy npz drop rows", npz_drop_rows_time)

        # NumPy npy drop rows
        start_time = time.time()
        npy_data = np.load('test_drop_array1.npy')
        mask = np.ones(size, dtype=bool)
        mask[drop_indices] = False
        npy_data = npy_data[mask]
        np.save('test_drop_array1.npy', npy_data)
        npy_drop_rows_time = time.time() - start_time
        timing_stats.add_time("NumPy npy drop rows", npy_drop_rows_time)

        # Verify data after row dropping
        loaded_arr1 = npk.load('array1')
        loaded_arr2 = npk.load('array2')
        npz_loaded = dict(np.load('test_drop.npz'))
        npy_loaded = np.load('test_drop_array1.npy')

        # Verify array1 has correct shape after dropping rows
        expected_shape = (size // 2, arrays['array1'].shape[1])
        assert loaded_arr1.shape == expected_shape
        assert npz_loaded['array1'].shape == expected_shape
        assert npy_loaded.shape == expected_shape
        
        # Verify array2 remains unchanged
        assert np.allclose(loaded_arr2, arrays['array2'])
        assert np.allclose(npz_loaded['array2'], arrays['array2'])
        
    except Exception as e:
        logger.error(f"Drop operations test failed: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        logger.info("Performance Test Results")
        logger.info("=" * 80)
        test_large_data()
        test_append_operations()
        test_random_access()
        test_replace_operations()
        test_drop_operations()
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Error occurred during tests: {str(e)}")
        sys.exit(1) 