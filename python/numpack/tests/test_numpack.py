import numpy as np
import pytest
import tempfile
from concurrent.futures import ThreadPoolExecutor
from numpack import NumPack
import sys
from pathlib import Path

# Import constants from conftest
sys.path.insert(0, str(Path(__file__).parent))
import conftest
ALL_DTYPES = conftest.ALL_DTYPES
ARRAY_DIMS = conftest.ARRAY_DIMS
create_test_array = conftest.create_test_array

@pytest.fixture
def temp_dir():
    """Create a temporary directory fixture"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def numpack(temp_dir):
    """Create a NumPack instance fixture"""
    npk = NumPack(temp_dir)
    npk.open()  # Open explicitly
    npk.reset()
    return npk

def handle_complex_test(dtype, test_func):
    """Helper function to run tests - simplified since all types are now supported"""
    return test_func()

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
@pytest.mark.parametrize("ndim,shape", ARRAY_DIMS)
def test_basic_save_load(numpack, dtype, test_values, ndim, shape):
    """Test basic save and load functionality for all data types and dimensions"""
    array1 = create_test_array(dtype, shape)
    array2 = create_test_array(dtype, shape)
    arrays = {'array1': array1, 'array2': array2}
    
    # Save and load
    numpack.save(arrays)
    arr1 = numpack.load('array1')
    arr2 = numpack.load('array2')
    
    # Verify
    assert np.array_equal(array1, arr1)
    assert np.array_equal(array2, arr2)
    assert array1.dtype == arr1.dtype
    assert array2.dtype == arr2.dtype
    assert array1.shape == arr1.shape
    assert array2.shape == arr2.shape

# mmap_mode support has been removed; the test case above no longer applies

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


def test_drop_various_index_types_1d(numpack):
    """Ensure drop works correctly for multiple index representations on 1D arrays."""
    def run_case(indexes, expected_indexes):
        with tempfile.TemporaryDirectory() as tmp:
            npk = NumPack(tmp)
            npk.open()
            base = np.arange(6, dtype=np.int32)
            npk.save({'array': base})
            npk.drop('array', indexes)
            loaded = npk.load('array')
            expected = np.delete(base, expected_indexes)
            assert np.array_equal(loaded, expected)
            assert npk.get_shape('array') == expected.shape

    run_case(0, 0)
    run_case([1, 3], [1, 3])
    run_case((2,), [2])
    run_case(np.array([4, -1]), [4, 5])
    run_case(slice(1, 4), [1, 2, 3])

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
    
    # Save and append
    numpack.save({'array': array})
    numpack.append({'array': append_data})
    loaded = numpack.load('array')
    
    # Verify
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
    
    # Save
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

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
@pytest.mark.parametrize("ndim,shape", ARRAY_DIMS)
def test_clone_basic(numpack, dtype, test_values, ndim, shape):
    """Test basic clone functionality for all data types and dimensions"""
    def run_test():
        # Create and save original array
        original_array = create_test_array(dtype, shape)
        numpack.save({'original': original_array})
        
        # Clone the array
        numpack.clone('original', 'cloned')
        
        # Load both arrays
        loaded_original = numpack.load('original')
        loaded_cloned = numpack.load('cloned')
        
        # Verify both arrays are identical
        assert np.array_equal(original_array, loaded_original)
        assert np.array_equal(original_array, loaded_cloned)
        assert loaded_original.dtype == loaded_cloned.dtype
        assert loaded_original.shape == loaded_cloned.shape
        
        # Verify both arrays exist in member list
        member_list = numpack.get_member_list()
        assert 'original' in member_list
        assert 'cloned' in member_list
    
    handle_complex_test(dtype, run_test)

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
@pytest.mark.parametrize("ndim,shape", ARRAY_DIMS)
def test_clone_independence(numpack, dtype, test_values, ndim, shape):
    """Test that cloned arrays are independent and can be modified separately"""
    def run_test():
        # Create and save original array
        original_array = create_test_array(dtype, shape)
        numpack.save({'original': original_array})
        
        # Clone the array
        numpack.clone('original', 'cloned')
        
        # Modify the cloned array
        cloned_array = numpack.load('cloned')
        if dtype == np.bool_:
            cloned_array = ~cloned_array  # Invert boolean values
        elif np.issubdtype(dtype, np.complexfloating):
            cloned_array = cloned_array * 2 + 1j
        else:
            cloned_array = cloned_array * 2
        
        numpack.save({'cloned': cloned_array})
        
        # Verify original array is unchanged
        loaded_original = numpack.load('original')
        assert np.array_equal(original_array, loaded_original)
        
        # Verify cloned array was modified
        loaded_cloned = numpack.load('cloned')
        assert not np.array_equal(original_array, loaded_cloned)
        assert np.array_equal(cloned_array, loaded_cloned)
    
    handle_complex_test(dtype, run_test)

def test_clone_errors(numpack):
    """Test error handling for clone operations"""
    # Test cloning non-existent array
    with pytest.raises(KeyError):
        numpack.clone('non_existent', 'target')
    
    # Test cloning to existing array name
    array = create_test_array(np.float64, (10, 5))
    numpack.save({'existing': array})
    numpack.clone('existing', 'clone1')
    
    with pytest.raises(ValueError):
        numpack.clone('existing', 'clone1')  # clone1 already exists

@pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
def test_clone_with_metadata(numpack, dtype, test_values):
    """Test that metadata is correctly cloned"""
    shape = (20, 10)
    original_array = create_test_array(dtype, shape)
    numpack.save({'original': original_array})
    
    # Clone the array
    numpack.clone('original', 'cloned')
    
    # Get metadata for both arrays
    metadata = numpack.get_metadata()
    
    original_meta = metadata['arrays']['original']
    cloned_meta = metadata['arrays']['cloned']
    
    # Verify metadata matches
    assert original_meta['shape'] == cloned_meta['shape']
    assert original_meta['dtype'] == cloned_meta['dtype']
    # Note: size_bytes should be the same since arrays are identical
    assert original_meta['size_bytes'] == cloned_meta['size_bytes']

if __name__ == '__main__':
    pytest.main([__file__, '-v'])