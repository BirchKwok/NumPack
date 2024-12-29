import numpy as np
import pytest
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
from numpack import NumPack

@pytest.fixture
def temp_dir():
    """创建临时目录的 fixture"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def numpack(temp_dir):
    """创建 NumPack 实例的 fixture"""
    npk = NumPack(temp_dir)
    npk.reset()
    return npk

def test_basic_save_load(numpack):
    """测试基本的保存和加载功能"""
    # 创建测试数据
    array1 = np.random.rand(100, 100).astype(np.float32)
    array2 = np.random.rand(50, 200).astype(np.float32)
    arrays = {'array1': array1, 'array2': array2}
    
    # 保存数组
    numpack.save_arrays(arrays)
    
    # 测试普通加载
    loaded_arrays = numpack.load_arrays(mmap_mode=False)
    assert np.array_equal(array1, loaded_arrays['array1'])
    assert np.array_equal(array2, loaded_arrays['array2'])
    
    # 测试形状
    assert array1.shape == loaded_arrays['array1'].shape
    assert array2.shape == loaded_arrays['array2'].shape

def test_mmap_load(numpack):
    """测试内存映射加载功能"""
    array = np.random.rand(100, 100).astype(np.float32)
    numpack.save_arrays({'array': array})
    
    # 测试 mmap 加载
    mmap_arrays = numpack.load_arrays(mmap_mode=True)
    assert np.array_equal(array, mmap_arrays['array'])
    assert isinstance(mmap_arrays['array'], np.memmap)

def test_selective_load(numpack):
    """测试选择性加载功能"""
    arrays = {
        'array1': np.random.rand(10, 10).astype(np.float32),
        'array2': np.random.rand(10, 10).astype(np.float32),
        'array3': np.random.rand(10, 10).astype(np.float32)
    }
    numpack.save_arrays(arrays)
    
    # 只加载部分数组
    loaded = numpack.load_arrays(['array1', 'array3'], mmap_mode=False)
    assert set(loaded.keys()) == {'array1', 'array3'}
    assert np.array_equal(arrays['array1'], loaded['array1'])
    assert np.array_equal(arrays['array3'], loaded['array3'])


def test_replace_with_indices(numpack):
    """测试使用索引列表替换数组部分内容"""
    original = np.random.rand(100, 50).astype(np.float32)
    numpack.save_arrays({'array': original})
    
    indices = [0, 10, 20, 30]
    replacement = np.random.rand(len(indices), 50).astype(np.float32)
    
    numpack.replace_arrays({'array': replacement}, indices)
    
    loaded = numpack.load_arrays(mmap_mode=False)['array']
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
    """测试不同数据类型的保存和加载"""
    array = np.array(test_values, dtype=dtype)
    numpack.save_arrays({'array': array})
    
    loaded = numpack.load_arrays(mmap_mode=False)['array']
    assert loaded.dtype == dtype
    assert np.array_equal(array, loaded)
    
    if np.issubdtype(dtype, np.floating):
        assert np.allclose(array, loaded, rtol=1e-6)

def test_large_array_handling(numpack):
    """测试大数组的处理"""
    large_array = np.random.rand(10000, 1000).astype(np.float32)
    numpack.save_arrays({'large': large_array})
    
    loaded = numpack.load_arrays(mmap_mode=True)['large']
    assert np.array_equal(large_array, loaded)

def test_metadata_operations(numpack):
    """测试元数据操作"""
    array = np.random.rand(100, 50).astype(np.float32)
    numpack.save_arrays({'array': array})
    
    # 测试形状获取
    shape = numpack.get_shape('array')
    assert shape == (100, 50)
    
    # 测试成员列表
    members = numpack.get_member_list()
    assert members == ['array']
    
    # 测试修改时间
    mtime = numpack.get_modify_time('array')
    assert isinstance(mtime, int)
    assert mtime > 0

def test_array_deletion(numpack):
    """测试数组删除功能"""
    arrays = {
        'array1': np.random.rand(10, 10).astype(np.float32),
        'array2': np.random.rand(10, 10).astype(np.float32)
    }
    numpack.save_arrays(arrays)
    
    # 删除单个数组
    numpack.drop_arrays('array1')
    loaded = numpack.load_arrays(mmap_mode=False)
    assert 'array1' not in loaded
    assert 'array2' in loaded
    
    # 删除多个数组
    numpack.save_arrays({'array1': arrays['array1']})
    numpack.drop_arrays(['array1', 'array2'])
    loaded = numpack.load_arrays(mmap_mode=False)
    assert len(loaded) == 0

def test_partial_row_deletion(numpack):
    """测试部分行删除功能"""
    array = np.random.rand(100, 50).astype(np.float32)
    numpack.save_arrays({'array': array})
    
    # 删除部分行
    numpack.drop_arrays('array', list(range(10, 20)))
    loaded = numpack.load_arrays(mmap_mode=False)['array']
    assert loaded.shape[0] == 90
    assert np.array_equal(array[:10], loaded[:10])
    assert np.array_equal(array[20:], loaded[10:])

def test_concurrent_operations(numpack):
    """测试并发操作"""
    def worker(thread_id):
        array = np.random.rand(100, 50).astype(np.float32)
        name = f'array_{thread_id}'
        numpack.save_arrays({name: array})
        loaded = numpack.load_arrays([name], mmap_mode=False)[name]
        return np.array_equal(array, loaded)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(worker, range(4)))
    
    assert all(results)
    loaded = numpack.load_arrays(mmap_mode=False)
    assert len(loaded) == 4

def test_error_handling(numpack):
    """测试错误处理"""
    # 测试加载不存在的数组
    with pytest.raises(KeyError):
        numpack.load_arrays(['nonexistent'], mmap_mode=False)
    
    # 测试保存不支持的数据类型
    with pytest.raises(Exception):
        numpack.save_arrays({'array': np.array([1+2j, 3+4j])})  # 复数类型不支持
    
    # 测试无效的切片操作
    array = np.random.rand(10, 10).astype(np.float32)
    numpack.save_arrays({'array': array})
    with pytest.raises(Exception):
        numpack.replace_arrays({'array': np.random.rand(5, 10)}, slice(20, 25))  # 超出范围的切片

if __name__ == '__main__':
    pytest.main([__file__, '-v'])