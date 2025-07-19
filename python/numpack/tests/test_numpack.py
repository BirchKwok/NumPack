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
    (np.float16, [[-1.0, 1.0], [0.0, 0.5]]),
    (np.float32, [[-1.0, 1.0], [0.0, 0.5]]),
    (np.float64, [[-1.0, 1.0], [0.0, 0.5]]),
    (np.complex64, [[1+2j, 3+4j], [0+0j, -1-2j]]),
    (np.complex128, [[1+2j, 3+4j], [0+0j, -1-2j]])
]

ARRAY_DIMS = [
    (1, (100,)),                           # 1 dimension
    (2, (50, 40)),                         # 2 dimension
    (3, (30, 20, 10)),                     # 3 dimension
    (4, (20, 15, 10, 5)),                  # 4 dimension
    (5, (10, 8, 6, 4, 2))                  # 5 dimension
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
    elif np.issubdtype(dtype, np.complexfloating):
        # Generate complex numbers with random real and imaginary parts
        real_part = np.random.rand(*shape) * 10 - 5  # random values between -5 and 5
        imag_part = np.random.rand(*shape) * 10 - 5  # random values between -5 and 5
        return (real_part + 1j * imag_part).astype(dtype)
    else:  # floating point
        return np.random.rand(*shape).astype(dtype)

def handle_complex_test(dtype, test_func):
    """Helper function to handle complex type tests with proper error handling"""
    if np.issubdtype(dtype, np.complexfloating):
        try:
            return test_func()
        except NotImplementedError as e:
            assert "Complex number support is currently under development" in str(e)
            pytest.skip(f"Complex type {dtype} not yet fully implemented in current backend")
    else:
        return test_func()

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
@pytest.mark.parametrize("ndim,shape", ARRAY_DIMS)
def test_basic_save_load(numpack, dtype, test_values, ndim, shape):
    """Test basic save and load functionality for all data types and dimensions"""
    array1 = create_test_array(dtype, shape)
    array2 = create_test_array(dtype, shape)
    arrays = {'array1': array1, 'array2': array2}
    
    # Handle complex types that might not be fully supported in Rust backend
    if np.issubdtype(dtype, np.complexfloating):
        # For complex types, test might raise NotImplementedError in Rust backend
        try:
            numpack.save(arrays)
            arr1 = numpack.load('array1')
            arr2 = numpack.load('array2')
            # If successful, verify the data
            assert np.array_equal(array1, arr1)
            assert np.array_equal(array2, arr2)
            assert array1.dtype == arr1.dtype
            assert array2.dtype == arr2.dtype
        except NotImplementedError as e:
            # Expected for Rust backend with complex types under development
            assert "Complex number support is currently under development" in str(e)
            pytest.skip(f"Complex type {dtype} not yet fully implemented in current backend")
    else:
        # Standard test for non-complex types
        numpack.save(arrays)
        arr1 = numpack.load('array1')
        arr2 = numpack.load('array2')
        assert np.array_equal(array1, arr1)
        assert np.array_equal(array2, arr2)
        assert array1.dtype == arr1.dtype
        assert array2.dtype == arr2.dtype
    assert array1.shape == arr1.shape
    assert array2.shape == arr2.shape

# mmap_mode 功能已移除，上述测试用例不再适用

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
@pytest.mark.parametrize("ndim,shape", ARRAY_DIMS)
def test_selective_load(numpack, dtype, test_values, ndim, shape):
    """Test selective load functionality for all data types and dimensions"""
    def run_test():
        arrays = {
            'array1': create_test_array(dtype, shape),
            'array2': create_test_array(dtype, shape),
            'array3': create_test_array(dtype, shape)
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
    
    handle_complex_test(dtype, run_test)

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
@pytest.mark.parametrize("ndim,shape", ARRAY_DIMS)
def test_metadata_operations(numpack, dtype, test_values, ndim, shape):
    """Test metadata operations for all data types and dimensions"""
    array = create_test_array(dtype, shape)
    numpack.save({'array': array})
    
    # Test shape retrieval
    saved_shape = numpack.get_shape('array')
    assert saved_shape == shape  # Direct comparison of tuples
    
    # Test member list
    members = numpack.get_member_list()
    assert members == ['array']
    
    # Test modify time
    mtime = numpack.get_modify_time('array')
    assert isinstance(mtime, int)
    assert mtime > 0

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
@pytest.mark.parametrize("ndim,shape", ARRAY_DIMS)
def test_array_deletion(numpack, dtype, test_values, ndim, shape):
    """Test array deletion functionality for all data types and dimensions"""
    arrays = {
        'array1': create_test_array(dtype, shape),
        'array2': create_test_array(dtype, shape)
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
@pytest.mark.parametrize("ndim,shape", ARRAY_DIMS)
def test_concurrent_operations(numpack, dtype, test_values, ndim, shape):
    """Test concurrent operations for all data types and dimensions"""
    def worker(thread_id):
        array = create_test_array(dtype, shape)
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
@pytest.mark.parametrize("ndim,shape", ARRAY_DIMS)
def test_error_handling(numpack, dtype, test_values, ndim, shape):
    """Test error handling for all data types and dimensions"""
    def run_test():
        # Test loading non-existent array
        with pytest.raises(KeyError):
            numpack.load('nonexistent')
        
        # Test saving unsupported data type (complex types are now supported)
        # Skip this test since complex types are now fully supported
        pass
        
        # Test invalid slice operation
        array = create_test_array(dtype, shape)
        numpack.save({'array': array})
        with pytest.raises(Exception):
            replacement = create_test_array(dtype, shape)
            numpack.replace({'array': replacement}, slice(shape[0] + 10, shape[0] + 15))  # Slice out of range
    
    handle_complex_test(dtype, run_test)

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
@pytest.mark.parametrize("ndim,shape", ARRAY_DIMS)
def test_append_operations(numpack, dtype, test_values, ndim, shape):
    """Test append operations for all data types and dimensions"""
    array = create_test_array(dtype, shape)
    append_data = create_test_array(dtype, shape)
    
    # Handle complex types that might not be fully supported in Rust backend
    if np.issubdtype(dtype, np.complexfloating):
        try:
            numpack.save({'array': array})
            numpack.append({'array': append_data})
            loaded = numpack.load('array')
            assert loaded.dtype == dtype
            assert loaded.shape[0] == 2 * shape[0]  # The first dimension should double
            assert np.array_equal(array, loaded[:shape[0]])
            assert np.array_equal(append_data, loaded[shape[0]:])
        except NotImplementedError as e:
            assert "Complex number support is currently under development" in str(e)
            pytest.skip(f"Complex type {dtype} not yet fully implemented in current backend")
    else:
        numpack.save({'array': array})
        numpack.append({'array': append_data})
        loaded = numpack.load('array')
        assert loaded.dtype == dtype
        assert loaded.shape[0] == 2 * shape[0]  # The first dimension should double
        assert np.array_equal(array, loaded[:shape[0]])
        assert np.array_equal(append_data, loaded[shape[0]:])

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
@pytest.mark.parametrize("ndim,shape", ARRAY_DIMS)
def test_getitem(numpack, dtype, test_values, ndim, shape):
    """Test getitem functionality for all data types and dimensions"""
    def run_test():
        array = create_test_array(dtype, shape)
        numpack.save({'array': array})
        
        indices = [1, 2, 3]
        loaded = numpack.getitem('array', indices)
        assert np.array_equal(array[indices], loaded)
    
    handle_complex_test(dtype, run_test)

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
@pytest.mark.parametrize("ndim,shape", ARRAY_DIMS)
def test_get_metadata(numpack, dtype, test_values, ndim, shape):
    """Test get_metadata functionality for all dimensions"""
    def run_test():
        array = create_test_array(dtype, shape)
        numpack.save({'array': array})
        
        metadata = numpack.get_metadata()
        assert isinstance(metadata, dict)
        assert 'arrays' in metadata
        assert 'array' in metadata['arrays']
        assert 'shape' in metadata['arrays']['array']
        assert metadata['arrays']['array']['shape'] == list(shape)
    
    handle_complex_test(dtype, run_test)

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
@pytest.mark.parametrize("ndim,shape", ARRAY_DIMS)
def test_magic_methods(numpack, dtype, test_values, ndim, shape):
    """Test magic methods (__getitem__ and __iter__) for all dimensions"""
    def run_test():
        arrays = {
            'array1': create_test_array(dtype, shape),
            'array2': create_test_array(dtype, shape)
        }
        numpack.save(arrays)
        
        # Test __getitem__
        loaded1 = numpack['array1']
        assert np.array_equal(arrays['array1'], loaded1)
        
        # Test __iter__
        member_list = list(numpack)
        assert set(member_list) == set(['array1', 'array2'])
    
    handle_complex_test(dtype, run_test)

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
@pytest.mark.parametrize("ndim,shape", ARRAY_DIMS)
def test_stream_load(numpack, dtype, test_values, ndim, shape):
    """Test stream_load functionality"""
    array = create_test_array(dtype, shape)
    
    # Handle complex types that might not be fully supported in Rust backend
    if np.issubdtype(dtype, np.complexfloating):
        try:
            numpack.save({'array': array})
            
            # Test with buffer_size=None
            for i, chunk in enumerate(numpack.stream_load('array', buffer_size=None)):
                assert np.array_equal(array[i:i+1], chunk)
            
            # Test with specific buffer_size
            buffer_size = 10
            for i, chunk in enumerate(numpack.stream_load('array', buffer_size=buffer_size)):
                start_idx = i * buffer_size
                end_idx = min(start_idx + buffer_size, shape[0])
                assert np.array_equal(array[start_idx:end_idx], chunk)
            
            # Test invalid buffer_size
            with pytest.raises(ValueError):
                next(numpack.stream_load('array', buffer_size=0))
        except NotImplementedError as e:
            assert "Complex number support is currently under development" in str(e)
            pytest.skip(f"Complex type {dtype} not yet fully implemented in current backend")
    else:
        numpack.save({'array': array})
        
        # Test with buffer_size=None
        for i, chunk in enumerate(numpack.stream_load('array', buffer_size=None)):
            assert np.array_equal(array[i:i+1], chunk)
        
        # Test with specific buffer_size
        buffer_size = 10
        for i, chunk in enumerate(numpack.stream_load('array', buffer_size=buffer_size)):
            start_idx = i * buffer_size
            end_idx = min(start_idx + buffer_size, shape[0])
            assert np.array_equal(array[start_idx:end_idx], chunk)
        
        # Test invalid buffer_size
        with pytest.raises(ValueError):
            next(numpack.stream_load('array', buffer_size=0))

if __name__ == '__main__':
    pytest.main([__file__, '-v'])