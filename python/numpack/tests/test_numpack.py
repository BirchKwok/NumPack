import numpy as np
import pytest
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
from numpack import NumPack

@pytest.fixture
def temp_dir():
    """Create a temporary directory fixture"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def numpack(temp_dir):
    """Create a NumPack instance fixture"""
    npk = NumPack(temp_dir)
    npk.reset()
    return npk

def test_basic_save_load(numpack):
    """Test basic save and load functionality"""
    # Create test data
    array1 = np.random.rand(100, 100).astype(np.float32)
    array2 = np.random.rand(50, 200).astype(np.float32)
    arrays = {'array1': array1, 'array2': array2}
    
    # Save arrays
    numpack.save(arrays)
    
    # Test normal load
    loaded_arrays = numpack.load(mmap_mode=False)
    assert np.array_equal(array1, loaded_arrays['array1'])
    assert np.array_equal(array2, loaded_arrays['array2'])
    
    # Test shape
    assert array1.shape == loaded_arrays['array1'].shape
    assert array2.shape == loaded_arrays['array2'].shape

def test_mmap_load(numpack):
    """Test mmap load functionality"""
    array = np.random.rand(100, 100).astype(np.float32)
    numpack.save({'array': array})
    
    # Test mmap load
    mmap_arrays = numpack.load(mmap_mode=True)
    assert np.array_equal(array, mmap_arrays['array'])

def test_mmap_load_after_row_deletion(numpack):
    """Test mmap load functionality after row deletion"""
    # Create test data
    array = np.random.rand(100, 50).astype(np.float32)
    numpack.save({'array': array})
    
    # Delete some rows
    deleted_indices = [10, 20, 30, 40, 50]  # Delete 5 rows
    numpack.drop('array', deleted_indices)
    
    # Load with mmap mode
    loaded = numpack.load(mmap_mode=True)['array']
    
    # Verify data correctness
    expected = np.delete(array, deleted_indices, axis=0)
    assert loaded.shape == (95, 50)  
    assert np.array_equal(loaded, expected)
    
    # Test random access to some rows
    test_indices = [0, 25, 50, 75]  # Test some random positions
    for idx in test_indices:
        assert np.array_equal(loaded[idx], expected[idx])

def test_selective_load(numpack):
    """Test selective load functionality"""
    arrays = {
        'array1': np.random.rand(10, 10).astype(np.float32),
        'array2': np.random.rand(10, 10).astype(np.float32),
        'array3': np.random.rand(10, 10).astype(np.float32)
    }
    numpack.save(arrays)
    
    # Load only some arrays
    loaded = numpack.load(mmap_mode=False)
    assert set(loaded.keys()) == {'array1', 'array2', 'array3'}
    assert np.array_equal(arrays['array1'], loaded['array1'])
    assert np.array_equal(arrays['array3'], loaded['array3'])


def test_replace_with_indices(numpack):
    """Test replacing array content with index list"""
    original = np.random.rand(100, 50).astype(np.float32)
    numpack.save({'array': original})
    
    indices = [0, 10, 20, 30]
    replacement = np.random.rand(len(indices), 50).astype(np.float32)
    
    numpack.replace({'array': replacement}, indices)
    
    loaded = numpack.load(mmap_mode=False)['array']
    assert np.array_equal(replacement, loaded[indices])

@pytest.mark.parametrize("dtype,test_values", [
    (np.bool_, [[True, False], [False, True]]),
    (np.uint8, [[0, 255], [128, 64]]),
    (np.uint16, [[0, 65535], [32768, 16384]]),
    (np.uint32, [[0, 4294967295], [2147483648, 1073741824]]),
    (np.uint64, [[0, 18446744073709551615], [9223372036854775808, 4611686018427387904]]),
    (np.int8, [[-128, 127], [0, -64]]),
    (np.int16, [[-32768, 32767], [0, -16384]]),
    (np.int32, [[-2147483648, 2147483647], [0, -1073741824]]),
    (np.int64, [[-9223372036854775808, 9223372036854775807], [0, -4611686018427387904]]),
    (np.float32, [[-1.0, 1.0], [0.0, 0.5]]),
    (np.float64, [[-1.0, 1.0], [0.0, 0.5]])
])
def test_data_types(numpack, dtype, test_values):
    """Test saving and loading different data types"""
    array = np.array(test_values, dtype=dtype)
    numpack.save({'array': array})
    
    loaded = numpack.load(mmap_mode=False)['array']
    assert loaded.dtype == dtype
    assert np.array_equal(array, loaded)
    
    if np.issubdtype(dtype, np.floating):
        assert np.allclose(array, loaded, rtol=1e-6)

def test_large_array_handling(numpack):
    """Test handling large arrays"""
    large_array = np.random.rand(10000, 1000).astype(np.float32)
    numpack.save({'large': large_array})
    
    loaded = numpack.load(mmap_mode=True)['large']
    assert np.array_equal(large_array, loaded)

def test_metadata_operations(numpack):
    """Test metadata operations"""
    array = np.random.rand(100, 50).astype(np.float32)
    numpack.save({'array': array})
    
    # Test shape retrieval
    shape = numpack.get_shape('array')
    assert shape == (100, 50)
    
    # Test member list
    members = numpack.get_member_list()
    assert members == ['array']
    
    # Test modify time
    mtime = numpack.get_modify_time('array')
    assert isinstance(mtime, int)
    assert mtime > 0

def test_array_deletion(numpack):
    """Test array deletion functionality"""
    arrays = {
        'array1': np.random.rand(10, 10).astype(np.float32),
        'array2': np.random.rand(10, 10).astype(np.float32)
    }
    numpack.save(arrays)
    
    # Delete single array
    numpack.drop('array1')
    loaded = numpack.load(mmap_mode=False)
    assert 'array1' not in loaded
    assert 'array2' in loaded
    
    # Delete multiple arrays
    numpack.save({'array1': arrays['array1']})
    numpack.drop(['array1', 'array2'])
    loaded = numpack.load(mmap_mode=False)
    assert len(loaded) == 0

def test_partial_row_deletion(numpack):
    """Test partial row deletion functionality"""
    array = np.random.rand(100, 50).astype(np.float32)
    numpack.save({'array': array})
    
    # Delete partial rows
    numpack.drop('array', list(range(10, 20)))
    loaded = numpack.load(mmap_mode=False)['array']
    assert loaded.shape[0] == 90
    assert np.array_equal(array[:10], loaded[:10])
    assert np.array_equal(array[20:], loaded[10:])

def test_concurrent_operations(numpack):
    """Test concurrent operations"""
    def worker(thread_id):
        array = np.random.rand(100, 50).astype(np.float32)
        name = f'array_{thread_id}'
        numpack.save({name: array})
        loaded = numpack.load(mmap_mode=False)[name]
        return np.array_equal(array, loaded)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(worker, range(4)))
    
    assert all(results)
    loaded = numpack.load(mmap_mode=False)
    assert len(loaded) == 4

def test_error_handling(numpack):
    """Test error handling"""
    # Test loading non-existent array
    with pytest.raises(KeyError):
        numpack.load(mmap_mode=False)['nonexistent']
    
    # Test saving unsupported data type
    with pytest.raises(Exception):
        numpack.save({'array': np.array([1+2j, 3+4j])})  # Complex type not supported
    
    # Test invalid slice operation
    array = np.random.rand(10, 10).astype(np.float32)
    numpack.save({'array': array})
    with pytest.raises(Exception):
        numpack.replace({'array': np.random.rand(5, 10)}, slice(20, 25))  # Slice out of range

def test_append_operations(numpack):
    """Test append operations"""
    # Create initial array
    array = np.random.rand(100, 50).astype(np.float32)
    numpack.save({'array': array})
    
    # Append new data
    append_data = np.random.rand(50, 50).astype(np.float32)
    numpack.append({'array': append_data})
    
    # Verify append result
    loaded = numpack.load(mmap_mode=False)['array']
    assert loaded.shape[0] == 150
    assert np.array_equal(array, loaded[:100])
    assert np.array_equal(append_data, loaded[100:])
    
    # Test append dimension mismatch
    with pytest.raises(ValueError):
        numpack.append({'array': np.random.rand(10, 30)})  # Column number mismatch

if __name__ == '__main__':
    pytest.main([__file__, '-v'])