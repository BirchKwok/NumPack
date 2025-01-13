import numpy as np
import pytest
import tempfile
from concurrent.futures import ThreadPoolExecutor
from numpack import NumPack

ALL_DTYPES = [
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
]

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

def create_test_array(dtype, shape):
    """Helper function to create test arrays of different types"""
    if dtype == np.bool_:
        return np.random.choice([True, False], size=shape).astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.random.randint(info.min // 2, info.max // 2, size=shape, dtype=dtype)
    else:  # floating point
        return np.random.rand(*shape).astype(dtype)

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
def test_basic_save_load(numpack, dtype, test_values):
    """Test basic save and load functionality for all data types"""
    array1 = create_test_array(dtype, (100, 100))
    array2 = create_test_array(dtype, (50, 200))
    arrays = {'array1': array1, 'array2': array2}
    
    numpack.save(arrays)
    
    arr1 = numpack.load('array1')
    arr2 = numpack.load('array2')
    assert np.array_equal(array1, arr1)
    assert np.array_equal(array2, arr2)
    assert array1.dtype == arr1.dtype
    assert array2.dtype == arr2.dtype
    
    assert array1.shape == arr1.shape
    assert array2.shape == arr2.shape

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
def test_mmap_load(numpack, dtype, test_values):
    """Test mmap load functionality for all data types"""
    array = create_test_array(dtype, (100, 100))
    numpack.save({'array': array})

    with numpack.mmap_mode() as mmap_npk:
        mmap_array = mmap_npk.load('array')
        assert np.array_equal(array, mmap_array)
        assert array.dtype == mmap_array.dtype

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
def test_mmap_load_after_row_deletion(numpack, dtype, test_values):
    """Test mmap load functionality after row deletion for all data types"""
    array = create_test_array(dtype, (100, 50))
    numpack.save({'array': array})
    
    deleted_indices = [10, 20, 30, 40, 50]
    numpack.drop('array', deleted_indices)
    
    with numpack.mmap_mode() as mmap_npk:
        loaded = mmap_npk.load('array')
        expected = np.delete(array, deleted_indices, axis=0)
        
        assert loaded.shape == (95, 50)
        assert loaded.dtype == dtype
        assert np.array_equal(loaded, expected)
        
        test_indices = [0, 25, 50, 75]
        for idx in test_indices:
            assert np.array_equal(loaded[idx], expected[idx])

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
def test_selective_load(numpack, dtype, test_values):
    """Test selective load functionality for all data types"""
    arrays = {
        'array1': create_test_array(dtype, (10, 10)),
        'array2': create_test_array(dtype, (10, 10)),
        'array3': create_test_array(dtype, (10, 10))
    }
    numpack.save(arrays)
    
    loaded1 = numpack.load('array1')
    loaded2 = numpack.load('array2')
    loaded3 = numpack.load('array3')
    
    assert loaded1.dtype == dtype
    assert loaded2.dtype == dtype
    assert loaded3.dtype == dtype
    assert np.array_equal(arrays['array1'], loaded1)
    assert np.array_equal(arrays['array2'], loaded2)
    assert np.array_equal(arrays['array3'], loaded3)

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
def test_metadata_operations(numpack, dtype, test_values):
    """Test metadata operations for all data types"""
    array = create_test_array(dtype, (100, 50))
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

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
def test_array_deletion(numpack, dtype, test_values):
    """Test array deletion functionality for all data types"""
    arrays = {
        'array1': create_test_array(dtype, (10, 10)),
        'array2': create_test_array(dtype, (10, 10))
    }
    numpack.save(arrays)
    
    # Delete single array
    numpack.drop('array1')
    with pytest.raises(KeyError):
        numpack.load('array1')
    loaded2 = numpack.load('array2')
    assert loaded2.dtype == dtype
    assert np.array_equal(arrays['array2'], loaded2)
    
    # Delete multiple arrays
    numpack.save({'array1': arrays['array1']})
    numpack.drop(['array1', 'array2'])
    member_list = numpack.get_member_list()
    assert len(member_list) == 0

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
def test_concurrent_operations(numpack, dtype, test_values):
    """Test concurrent operations for all data types"""
    def worker(thread_id):
        array = create_test_array(dtype, (100, 50))
        name = f'array_{thread_id}'
        numpack.save({name: array})
        loaded = numpack.load(name)
        return np.array_equal(array, loaded) and loaded.dtype == dtype
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(worker, range(4)))
    
    assert all(results)
    member_list = numpack.get_member_list()
    assert len(member_list) == 4
    for i in range(4):
        array_name = f'array_{i}'
        loaded = numpack.load(array_name)
        assert loaded.dtype == dtype

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
def test_error_handling(numpack, dtype, test_values):
    """Test error handling for all data types"""
    # Test loading non-existent array
    with pytest.raises(KeyError):
        numpack.load('nonexistent')
    
    # Test saving unsupported data type
    with pytest.raises(Exception):
        numpack.save({'array': np.array([1+2j, 3+4j])})  # Complex type not supported
    
    # Test invalid slice operation
    array = create_test_array(dtype, (10, 10))
    numpack.save({'array': array})
    with pytest.raises(Exception):
        replacement = create_test_array(dtype, (5, 10))
        numpack.replace({'array': replacement}, slice(20, 25))  # Slice out of range

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
def test_append_operations(numpack, dtype, test_values):
    """Test append operations for all data types"""
    array = create_test_array(dtype, (100, 50))
    numpack.save({'array': array})
    
    append_data = create_test_array(dtype, (50, 50))
    numpack.append({'array': append_data})
    
    loaded = numpack.load('array')
    assert loaded.dtype == dtype
    assert loaded.shape[0] == 150
    assert np.array_equal(array, loaded[:100])
    assert np.array_equal(append_data, loaded[100:])
    
    with pytest.raises(ValueError):
        numpack.append({'array': create_test_array(dtype, (10, 30))})

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
def test_getitem(numpack, dtype, test_values):
    """Test getitem functionality for all data types"""
    array = create_test_array(dtype, (100, 50))
    numpack.save({'array': array})
    
    loaded = numpack.getitem('array', [10, 20, 30])
    assert np.array_equal(array[[10, 20, 30]], loaded)
    
    loaded = numpack.getitem('array', [15, 25, 35])
    assert np.array_equal(array[[15, 25, 35]], loaded)

    loaded = numpack.getitem('array', [10, 20, 30, 40, 50])
    assert np.array_equal(array[[10, 20, 30, 40, 50]], loaded)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])