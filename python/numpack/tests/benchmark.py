import os
import sys
import time
import logging
import numpy as np
from functools import wraps
from numpack import NumPack
from typing import Dict, List, Callable
from statistics import mean
import psutil
import pytest

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

def format_size(size_bytes):
    """Format size in bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

def get_memory_usage():
    """Get memory usage of the current process"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

@pytest.mark.benchmark
@run_multiple_times(runs=3)
@clean_file_when_finished('benchmark_data')
def test_very_large_array(timing_stats: TimingStats):
    """Test performance of 100 million rows, 128-dimensional float32 arrays"""
    try:
        rows = 100_000_000
        cols = 128
        chunk_size = 1_000_000
        
        test_dir = "benchmark_data"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        
        np_pack = NumPack(test_dir)
        
        # 1. Test save performance
        start_time = time.time()
        total_chunks = rows // chunk_size
        
        for i in range(total_chunks):
            if i == 0:
                # Create array for the first time
                data = np.random.randn(chunk_size, cols).astype(np.float32)
                np_pack.save({"large_array": data})
            else:
                # Append data
                data = np.random.randn(chunk_size, cols).astype(np.float32)
                np_pack.append({"large_array": data})
            
            if (i + 1) % 10 == 0:
                logger.info(f"Appended: {(i + 1) / total_chunks * 100:.2f}%")
            
        save_time = time.time() - start_time
        timing_stats.add_time("Save 100M rows array", save_time)
        
        # 2. Test load performance (normal mode)
        # start_time = time.time()
        # mem_before = get_memory_usage()
        # array = np_pack.load("large_array")
        # load_time = time.time() - start_time
        # mem_after = get_memory_usage()
        # timing_stats.add_time("Load 100M rows array", load_time)
        # timing_stats.add_time("Load memory increase(GB)", (mem_after - mem_before) / (1024**3))
        
        # 3. Test load performance (lazy mode)
        start_time = time.time()
        mem_before = get_memory_usage()
        lazy_array = np_pack.load("large_array", lazy=True)
        load_time = time.time() - start_time
        mem_after = get_memory_usage()
        timing_stats.add_time("Load 100M rows array", load_time)
        timing_stats.add_time("Lazy load memory increase(MB)", (mem_after - mem_before) / (1024**2))
        
        # 4. Test random access performance
        n_reads = 1000
        indices = np.random.randint(0, rows, n_reads)
        
        start_time = time.time()
        for i in range(n_reads):
            _ = np_pack.getitem("large_array", indices[i:i+1])
        random_read_time = time.time() - start_time
        timing_stats.add_time("Average random read time(ms)", random_read_time/n_reads*1000)
        
        # Batch random read
        start_time = time.time()
        _ = np_pack.getitem("large_array", indices)
        batch_read_time = time.time() - start_time
        timing_stats.add_time("Batch random read", batch_read_time)
        
        # 5. Test append performance
        append_size = 1_000_000  # Append 1000000 rows
        data = np.random.randn(append_size, cols).astype(np.float32)
        
        start_time = time.time()
        np_pack.append({"large_array": data})
        append_time = time.time() - start_time
        timing_stats.add_time("Append 1000000 rows", append_time)
        
        # 6. Test drop performance
        delete_size = 1_000_000  # Delete 1000000 rows
        start_time = time.time()
        np_pack.drop("large_array", list(range(delete_size)))
        delete_time = time.time() - start_time
        timing_stats.add_time("Delete 1000000 rows", delete_time)
        
        # 7. Test replace performance
        replace_size = 1_000_000  # Replace 1000000 rows
        data = np.random.randn(replace_size, cols).astype(np.float32)
        indices = list(range(replace_size))
        
        start_time = time.time()
        np_pack.replace({"large_array": data}, indices)
        replace_time = time.time() - start_time
        timing_stats.add_time("Replace 1000000 rows", replace_time)
        
    finally:
        # Clean test data
        np_pack.reset()
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir)

@pytest.mark.benchmark
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
        
        # Test NumPack lazy load (mmap substitute)
        start_time = time.time()
        lazy_arr1 = npk.load('array1', lazy=True)
        lazy_arr2 = npk.load('array2', lazy=True)
        lazy_load_time = time.time() - start_time
        timing_stats.add_time("NumPack lazy load", lazy_load_time)
        
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

@pytest.mark.benchmark
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

@pytest.mark.benchmark
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

@pytest.mark.benchmark
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

@pytest.mark.benchmark
@run_multiple_times(runs=7)
@clean_file_when_finished('test_drop', 'test_drop.npz', 'test_drop_array1.npy', 'test_drop_array2.npy')
def test_drop_operations(timing_stats: TimingStats):
    """Test drop operations performance for various scenarios"""
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
        
        # 1.1 Drop first row
        start_time = time.time()
        npk.drop('array1', [0])
        timing_stats.add_time("NumPack drop first row", time.time() - start_time)
        
        # 1.2 Drop middle row
        middle_idx = size // 2
        start_time = time.time()
        npk.drop('array1', [middle_idx])
        timing_stats.add_time("NumPack drop middle row", time.time() - start_time)
        
        # 1.3 Drop last row
        start_time = time.time()
        npk.drop('array1', [size - 1])
        timing_stats.add_time("NumPack drop last row", time.time() - start_time)
        
        # Reload data for next test
        npk.reset()
        npk.save(arrays)
        
        # 2.1 Drop front continuous rows
        chunk_size = 10000  # Drop 10000 rows each time
        
        # 2.1 Drop front continuous rows
        front_indices = list(range(chunk_size))
        start_time = time.time()
        npk.drop('array1', front_indices)
        timing_stats.add_time("NumPack drop front continuous rows", time.time() - start_time)
        
        # 2.2 Drop middle continuous rows
        middle_start = size // 2 - chunk_size // 2
        middle_indices = list(range(middle_start, middle_start + chunk_size))
        start_time = time.time()
        npk.drop('array1', middle_indices)
        timing_stats.add_time("NumPack drop middle continuous rows", time.time() - start_time)
        
        # 2.3 Drop end continuous rows
        end_indices = list(range(size - chunk_size, size))
        start_time = time.time()
        npk.drop('array1', end_indices)
        timing_stats.add_time("NumPack drop end continuous rows", time.time() - start_time)
        
        # Reload data for next test
        npk.reset()
        npk.save(arrays)
        
        # 3.1 Drop near but non-continuous rows
        step = 2
        near_indices = list(range(0, chunk_size * step, step))  # Drop every other row
        start_time = time.time()
        npk.drop('array1', near_indices)
        timing_stats.add_time("NumPack drop near but non-continuous rows", time.time() - start_time)
        
        # Reload data for next test
        npk.reset()
        npk.save(arrays)
        
        # 4.1 Drop random distributed rows
        random_indices = np.random.choice(size, chunk_size, replace=False).tolist()
        start_time = time.time()
        npk.drop('array1', random_indices)
        timing_stats.add_time("NumPack drop random rows", time.time() - start_time)
        
        # Reload data for next test
        npk.reset()
        npk.save(arrays)
        
        # 5.1 Drop entire array
        start_time = time.time()
        npk.drop('array1')
        timing_stats.add_time("NumPack drop entire array", time.time() - start_time)
        
        # Compare with NumPy performance
        # For fair comparison, we also perform the same operations on npz and npy files
        
        # Test drop random rows performance (as a representative scenario)
        # NumPy npz
        start_time = time.time()
        npz_data = dict(np.load('test_drop.npz'))
        mask = np.ones(size, dtype=bool)
        mask[random_indices] = False
        npz_data['array1'] = npz_data['array1'][mask]
        np.savez('test_drop.npz', **npz_data)
        timing_stats.add_time("NumPy npz drop random rows", time.time() - start_time)
        
        # NumPy npy
        start_time = time.time()
        npy_data = np.load('test_drop_array1.npy')
        npy_data = npy_data[mask]
        np.save('test_drop_array1.npy', npy_data)
        timing_stats.add_time("NumPy npy drop random rows", time.time() - start_time)
        
        # Test drop entire array performance
        # NumPy npz
        start_time = time.time()
        npz_data = dict(np.load('test_drop.npz'))
        del npz_data['array1']
        np.savez('test_drop.npz', **npz_data)
        timing_stats.add_time("NumPy npz drop entire array", time.time() - start_time)
        
        # NumPy npy
        start_time = time.time()
        os.remove('test_drop_array1.npy')
        timing_stats.add_time("NumPy npy drop entire array", time.time() - start_time)
        
        # Verify data consistency
        # Reload data for verification
        npk.reset()
        npk.save(arrays)
        
        # Verify single row drop
        npk.drop('array1', [0])
        loaded = npk.load('array1')
        assert loaded.shape[0] == size - 1
        assert np.allclose(loaded[0], arrays['array1'][1])
        
        # Verify continuous rows drop
        npk.reset()
        npk.save(arrays)
        npk.drop('array1', list(range(chunk_size)))
        loaded = npk.load('array1')
        assert loaded.shape[0] == size - chunk_size
        assert np.allclose(loaded[0], arrays['array1'][chunk_size])
        
        # Verify random rows drop
        npk.reset()
        npk.save(arrays)
        npk.drop('array1', random_indices)
        loaded = npk.load('array1')
        assert loaded.shape[0] == size - len(random_indices)
        
    except Exception as e:
        logger.error(f"Drop operations test failed: {str(e)}")
        raise

@pytest.mark.benchmark
@run_multiple_times(runs=7)
@clean_file_when_finished('test_append_rows', 'test_append_rows.npz', 'test_append_rows_array.npy')
def test_append_rows_operations(timing_stats: TimingStats):
    """Test performance of appending rows to existing arrays"""
    try:
        # Create initial data
        size = 1000000
        initial_array = np.random.rand(size, 10).astype(np.float32)
        
        # Save initial data
        npk = NumPack('test_append_rows', drop_if_exists=True)
        npk.save({'array': initial_array})
        np.savez('test_append_rows.npz', array=initial_array)
        np.save('test_append_rows_array.npy', initial_array)
        
        # Test scenario 1: Append small number of rows
        small_append = np.random.rand(1000, 10).astype(np.float32)
        
        # NumPack append rows
        start_time = time.time()
        npk.append({'array': small_append})
        append_time = time.time() - start_time
        timing_stats.add_time("NumPack small rows append", append_time)
        
        # NumPy npz append rows
        start_time = time.time()
        npz_data = dict(np.load('test_append_rows.npz'))
        npz_data['array'] = np.vstack([npz_data['array'], small_append])
        np.savez('test_append_rows.npz', **npz_data)
        npz_append_time = time.time() - start_time
        timing_stats.add_time("NumPy npz small rows append", npz_append_time)
        
        # NumPy npy append rows
        start_time = time.time()
        npy_data = np.load('test_append_rows_array.npy')
        npy_data = np.vstack([npy_data, small_append])
        np.save('test_append_rows_array.npy', npy_data)
        npy_append_time = time.time() - start_time
        timing_stats.add_time("NumPy npy small rows append", npy_append_time)
        
        # Test scenario 2: Append large number of rows
        large_append = np.random.rand(size // 2, 10).astype(np.float32)
        
        # NumPack append rows
        start_time = time.time()
        npk.append({'array': large_append})
        append_time = time.time() - start_time
        timing_stats.add_time("NumPack large rows append", append_time)
        
        # NumPy npz append rows
        start_time = time.time()
        npz_data = dict(np.load('test_append_rows.npz'))
        npz_data['array'] = np.vstack([npz_data['array'], large_append])
        np.savez('test_append_rows.npz', **npz_data)
        npz_append_time = time.time() - start_time
        timing_stats.add_time("NumPy npz large rows append", npz_append_time)
        
        # NumPy npy append rows
        start_time = time.time()
        npy_data = np.load('test_append_rows_array.npy')
        npy_data = np.vstack([npy_data, large_append])
        np.save('test_append_rows_array.npy', npy_data)
        npy_append_time = time.time() - start_time
        timing_stats.add_time("NumPy npy large rows append", npy_append_time)
        
        # Verify data
        expected_rows = size + 1000 + size // 2
        loaded = npk.load('array')
        npz_loaded = np.load('test_append_rows.npz')['array']
        npy_loaded = np.load('test_append_rows_array.npy')
        
        assert loaded.shape == (expected_rows, 10)
        assert npz_loaded.shape == (expected_rows, 10)
        assert npy_loaded.shape == (expected_rows, 10)
        
        # Verify the appended data is correct
        assert np.allclose(loaded[size:size+1000], small_append)
        assert np.allclose(loaded[size+1000:], large_append)
        assert np.allclose(npz_loaded[size:size+1000], small_append)
        assert np.allclose(npz_loaded[size+1000:], large_append)
        assert np.allclose(npy_loaded[size:size+1000], small_append)
        assert np.allclose(npy_loaded[size+1000:], large_append)
        
    except Exception as e:
        logger.error(f"Append rows operations test failed: {str(e)}")
        raise

@pytest.mark.benchmark
@run_multiple_times(runs=7)
@clean_file_when_finished('test_matrix', 'test_matrix.npz', 'test_matrix_array.npy')
def test_matrix_computation(timing_stats: TimingStats):
    """Test matrix computation performance with different storage methods"""
    try:
        rows = 1000000
        cols = 128
        array = np.random.random((rows, cols)).astype(np.float32)
        
        npk = NumPack('test_matrix', drop_if_exists=True)
        npk.save({'array': array})
        np.savez('test_matrix.npz', array=array)
        np.save('test_matrix_array.npy', array)
        
        _ = npk.load('array', lazy=True)
        with np.load('test_matrix.npz', mmap_mode='r') as npz_data:
            _ = npz_data['array']
        _ = np.load('test_matrix_array.npy', mmap_mode='r')
        
        n_inner_runs = 10
        
        lazy_array = npk.load('array', lazy=True)
        query_vector = np.random.random((1, cols)).astype(np.float32)
        start_time = time.time()
        for _ in range(n_inner_runs):
            result_numpack = np.inner(query_vector, lazy_array)
        numpack_time = (time.time() - start_time) / n_inner_runs
        timing_stats.add_time("NumPack lazy mode matrix computation", numpack_time)
        
        with np.load('test_matrix.npz', mmap_mode='r') as npz_data:
            start_time = time.time()
            for _ in range(n_inner_runs):
                result_npz = np.inner(query_vector, npz_data['array'])
            npz_time = (time.time() - start_time) / n_inner_runs
        timing_stats.add_time("NumPy npz mmap matrix computation", npz_time)
        
        npy_mmap = np.load('test_matrix_array.npy', mmap_mode='r')
        start_time = time.time()
        for _ in range(n_inner_runs):
            result_npy = np.inner(query_vector, npy_mmap)
        npy_time = (time.time() - start_time) / n_inner_runs
        timing_stats.add_time("NumPy npy mmap matrix computation", npy_time)
        
        start_time = time.time()
        for _ in range(n_inner_runs):
            result_memory = np.inner(query_vector, array)
        memory_time = (time.time() - start_time) / n_inner_runs
        timing_stats.add_time("In-memory matrix computation", memory_time)
        
        result_numpack = np.inner(query_vector, lazy_array)
        result_memory = np.inner(query_vector, array)
        assert np.allclose(result_numpack, result_memory, rtol=1e-5)
        
        process = psutil.Process(os.getpid())
        timing_stats.add_time("Memory usage (MB)", process.memory_info().rss / (1024 * 1024))
        
    except Exception as e:
        logger.error(f"Matrix computation test failed: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        logger.info("Performance Test Results")
        logger.info("=" * 80)
        test_large_data()
        test_append_operations()
        test_append_rows_operations()
        test_random_access()
        test_replace_operations()
        test_drop_operations()
        test_matrix_computation()
        # test_very_large_array()
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Error occurred during tests: {str(e)}")
        sys.exit(1) 