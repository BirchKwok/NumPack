"""测试 writable_batch_mode 的零拷贝批处理功能"""
import numpy as np
import pytest
import tempfile
import shutil
import os
from numpack import NumPack


class TestWritableBatchModeBasic:
    """测试 writable_batch_mode 的基础功能"""
    
    def test_basic_value_modification(self):
        """测试基本的值修改"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(1000, 10).astype(np.float32)
                npk.save({'data': data})
                
                # 使用 writable_batch_mode 修改值
                with npk.writable_batch_mode() as wb:
                    arr = wb.load('data')
                    arr *= 2.0
                    wb.save({'data': arr})  # save 是可选的
                
                # 验证修改生效
                result = npk.load('data')
                assert np.allclose(result, data * 2.0)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_modification_without_save(self):
        """测试不调用save也能生效（零拷贝特性）"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                # 不调用 save，直接修改
                with npk.writable_batch_mode() as wb:
                    arr = wb.load('data')
                    arr += 10.0
                    # 注意：没有调用 wb.save()
                
                # 验证修改仍然生效
                result = npk.load('data')
                expected = data + 10.0
                assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_inplace_operations(self):
        """测试各种原地操作"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.writable_batch_mode() as wb:
                    arr = wb.load('data')
                    
                    # 各种原地操作
                    arr *= 2.0
                    arr += 5.0
                    arr /= 3.0
                    arr -= 1.0
                
                # 验证
                result = npk.load('data')
                expected = ((data * 2.0 + 5.0) / 3.0) - 1.0
                assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_array_view_property(self):
        """测试返回的是 mmap 视图（OWNDATA=False）"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.writable_batch_mode() as wb:
                    arr = wb.load('data')
                    
                    # 验证是视图而非拷贝
                    assert arr.flags['OWNDATA'] == False, "应该是 mmap 视图"
                    
                    # 修改仍然有效
                    arr[0, 0] = 999.0
                
                # 验证修改生效
                result = npk.load('data')
                assert result[0, 0] == 999.0
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


class TestWritableBatchModeMultipleArrays:
    """测试 writable_batch_mode 处理多个数组"""
    
    def test_modify_multiple_arrays(self):
        """测试修改多个数组"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                arrays = {
                    f'array_{i}': np.random.rand(100, 10).astype(np.float32)
                    for i in range(5)
                }
                npk.save(arrays)
                
                with npk.writable_batch_mode() as wb:
                    for i in range(5):
                        arr = wb.load(f'array_{i}')
                        arr *= (i + 1)
                
                # 验证所有数组
                for i in range(5):
                    result = npk.load(f'array_{i}')
                    expected = arrays[f'array_{i}'] * (i + 1)
                    assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_cached_array_access(self):
        """测试缓存的数组访问"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.writable_batch_mode() as wb:
                    # 第一次加载
                    arr1 = wb.load('data')
                    arr1[0, 0] = 100
                    
                    # 第二次加载应该返回同一个视图
                    arr2 = wb.load('data')
                    assert arr2[0, 0] == 100, "应该返回缓存的视图"
                    
                    # 验证是同一个对象
                    assert arr1 is arr2, "应该是同一个 mmap 视图"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_selective_array_modification(self):
        """测试选择性修改数组"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                arrays = {
                    f'array_{i}': np.random.rand(100, 10).astype(np.float32)
                    for i in range(10)
                }
                npk.save(arrays)
                
                with npk.writable_batch_mode() as wb:
                    # 只修改部分数组
                    for i in [2, 5, 8]:
                        arr = wb.load(f'array_{i}')
                        arr *= 10.0
                
                # 验证：只有选定的数组被修改
                for i in range(10):
                    result = npk.load(f'array_{i}')
                    if i in [2, 5, 8]:
                        expected = arrays[f'array_{i}'] * 10.0
                    else:
                        expected = arrays[f'array_{i}']
                    assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


class TestWritableBatchModeDataTypes:
    """测试 writable_batch_mode 对不同数据类型的支持"""
    
    def test_float_types(self):
        """测试浮点类型"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                arrays = {
                    'float16': np.random.rand(100, 10).astype(np.float16),
                    'float32': np.random.rand(100, 10).astype(np.float32),
                    'float64': np.random.rand(100, 10).astype(np.float64),
                }
                npk.save(arrays)
                
                with npk.writable_batch_mode() as wb:
                    for name in ['float16', 'float32', 'float64']:
                        arr = wb.load(name)
                        arr *= 2.0
                
                # 验证所有浮点类型
                for name, original in arrays.items():
                    result = npk.load(name)
                    expected = original * 2.0
                    assert np.allclose(result, expected, rtol=1e-3)
                    assert result.dtype == original.dtype
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_integer_types(self):
        """测试整数类型"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                arrays = {
                    'int8': np.random.randint(0, 10, (100, 10), dtype=np.int8),
                    'int16': np.random.randint(0, 100, (100, 10), dtype=np.int16),
                    'int32': np.random.randint(0, 1000, (100, 10), dtype=np.int32),
                    'int64': np.random.randint(0, 1000, (100, 10), dtype=np.int64),
                }
                npk.save(arrays)
                
                with npk.writable_batch_mode() as wb:
                    for name in ['int8', 'int16', 'int32', 'int64']:
                        arr = wb.load(name)
                        arr += 5
                
                # 验证所有整数类型
                for name, original in arrays.items():
                    result = npk.load(name)
                    expected = original + 5
                    assert np.array_equal(result, expected)
                    assert result.dtype == original.dtype
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_unsigned_integer_types(self):
        """测试无符号整数类型"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                arrays = {
                    'uint8': np.random.randint(0, 10, (100, 10), dtype=np.uint8),
                    'uint16': np.random.randint(0, 100, (100, 10), dtype=np.uint16),
                    'uint32': np.random.randint(0, 1000, (100, 10), dtype=np.uint32),
                    'uint64': np.random.randint(0, 1000, (100, 10), dtype=np.uint64),
                }
                npk.save(arrays)
                
                with npk.writable_batch_mode() as wb:
                    for name in ['uint8', 'uint16', 'uint32', 'uint64']:
                        arr = wb.load(name)
                        arr += 3
                
                # 验证所有无符号整数类型
                for name, original in arrays.items():
                    result = npk.load(name)
                    expected = original + 3
                    assert np.array_equal(result, expected)
                    assert result.dtype == original.dtype
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_bool_type(self):
        """测试布尔类型"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10) > 0.5
                npk.save({'bool_data': data})
                
                with npk.writable_batch_mode() as wb:
                    arr = wb.load('bool_data')
                    # 反转所有布尔值
                    arr[:] = ~arr
                
                # 验证
                result = npk.load('bool_data')
                expected = ~data
                assert np.array_equal(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


class TestWritableBatchModePerformance:
    """测试 writable_batch_mode 的性能相关场景"""
    
    def test_frequent_modifications(self):
        """测试频繁修改（writable_batch_mode的优势场景）"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(1000, 100).astype(np.float32)
                npk.save({'data': data})
                
                with npk.writable_batch_mode() as wb:
                    arr = wb.load('data')
                    
                    # 100次修改（零拷贝，直接在 mmap 上操作）
                    for i in range(100):
                        arr[:10] *= 1.01
                
                # 验证修改生效
                result = npk.load('data')
                assert not np.allclose(result, data)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_large_array_modification(self):
        """测试大数组修改（零内存开销）"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                # 创建一个较大的数组（~40MB）
                data = np.random.rand(5000, 1000).astype(np.float32)
                npk.save({'large': data})
                
                with npk.writable_batch_mode() as wb:
                    arr = wb.load('large')
                    
                    # 直接修改（零拷贝）
                    arr *= 0.5
                    arr += 1.0
                
                # 验证
                result = npk.load('large')
                expected = data * 0.5 + 1.0
                assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_random_access_pattern(self):
        """测试随机访问模式"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                arrays = {
                    f'array_{i}': np.random.rand(100, 10).astype(np.float32)
                    for i in range(10)
                }
                npk.save(arrays)
                
                with npk.writable_batch_mode() as wb:
                    # 随机访问模式
                    import random
                    random.seed(42)
                    for _ in range(50):
                        idx = random.randint(0, 9)
                        arr = wb.load(f'array_{idx}')
                        arr += 0.1
                
                # 验证（每个数组被修改了不同次数）
                for i in range(10):
                    result = npk.load(f'array_{i}')
                    assert not np.allclose(result, arrays[f'array_{i}'])
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


class TestWritableBatchModeSlicing:
    """测试 writable_batch_mode 的切片操作"""
    
    def test_row_slicing(self):
        """测试行切片修改"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.writable_batch_mode() as wb:
                    arr = wb.load('data')
                    
                    # 修改特定行
                    arr[10:20] *= 2.0
                    arr[30:40] += 5.0
                
                # 验证
                result = npk.load('data')
                expected = data.copy()
                expected[10:20] *= 2.0
                expected[30:40] += 5.0
                assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_column_slicing(self):
        """测试列切片修改"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.writable_batch_mode() as wb:
                    arr = wb.load('data')
                    
                    # 修改特定列
                    arr[:, 0] = 100
                    arr[:, 5:7] *= 3.0
                
                # 验证
                result = npk.load('data')
                expected = data.copy()
                expected[:, 0] = 100
                expected[:, 5:7] *= 3.0
                assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_fancy_indexing(self):
        """测试花式索引修改"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.writable_batch_mode() as wb:
                    arr = wb.load('data')
                    
                    # 花式索引
                    indices = [10, 20, 30, 40, 50]
                    arr[indices] = 999.0
                
                # 验证
                result = npk.load('data')
                assert np.all(result[indices] == 999.0)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_boolean_indexing(self):
        """测试布尔索引修改"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.writable_batch_mode() as wb:
                    arr = wb.load('data')
                    
                    # 布尔索引：将所有小于0.5的值设为0
                    mask = arr < 0.5
                    arr[mask] = 0.0
                
                # 验证
                result = npk.load('data')
                assert np.all(result[data < 0.5] == 0.0)
                assert np.all(result[data >= 0.5] == data[data >= 0.5])
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


class TestWritableBatchModeCompatibility:
    """测试 writable_batch_mode 的兼容性"""
    
    def test_nested_context_managers(self):
        """测试嵌套使用"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                # 第一次使用
                with npk.writable_batch_mode() as wb:
                    arr = wb.load('data')
                    arr *= 2.0
                
                # 第二次使用
                with npk.writable_batch_mode() as wb:
                    arr = wb.load('data')
                    arr += 5.0
                
                # 验证两次修改都生效
                result = npk.load('data')
                expected = data * 2.0 + 5.0
                assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_exception_handling(self):
        """测试异常处理"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                try:
                    with npk.writable_batch_mode() as wb:
                        arr = wb.load('data')
                        arr *= 3.0
                        
                        # 故意抛出异常
                        raise ValueError("Test exception")
                except ValueError:
                    pass
                
                # 验证修改仍然生效（mmap 自动刷新）
                result = npk.load('data')
                expected = data * 3.0
                assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_reopen_after_writable_batch_mode(self):
        """测试 writable_batch_mode 后重新打开文件"""
        numpack_dir = tempfile.mkdtemp()
        try:
            # 第一次：使用 writable_batch_mode
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.writable_batch_mode() as wb:
                    arr = wb.load('data')
                    arr *= 4.0
            
            # 第二次：重新打开验证
            with NumPack(numpack_dir) as npk:
                result = npk.load('data')
                expected = data * 4.0
                assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_interleaved_with_batch_mode(self):
        """测试与 batch_mode 交替使用"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                # 使用 writable_batch_mode
                with npk.writable_batch_mode() as wb:
                    arr = wb.load('data')
                    arr *= 2.0
                
                # 使用 batch_mode
                with npk.batch_mode():
                    arr = npk.load('data')
                    arr += 10.0
                    npk.save({'data': arr})
                
                # 验证两种模式都生效
                result = npk.load('data')
                expected = data * 2.0 + 10.0
                assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


class TestWritableBatchModeLimitations:
    """测试 writable_batch_mode 的限制"""
    
    def test_no_append_support(self):
        """测试不支持 append 操作"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.writable_batch_mode() as wb:
                    arr = wb.load('data')
                    
                    # writable_batch_mode 没有 append 方法
                    assert not hasattr(wb, 'append'), "writable_batch_mode 不应支持 append"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_no_drop_support(self):
        """测试不支持 drop 操作"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.writable_batch_mode() as wb:
                    arr = wb.load('data')
                    
                    # writable_batch_mode 没有 drop 方法
                    assert not hasattr(wb, 'drop'), "writable_batch_mode 不应支持 drop"
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_shape_unchanged(self):
        """测试形状不能改变"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.writable_batch_mode() as wb:
                    arr = wb.load('data')
                    
                    # 只能修改值，不能改变形状
                    original_shape = arr.shape
                    arr[:] *= 2.0
                    
                    # 形状不应改变
                    assert arr.shape == original_shape
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


class TestWritableBatchModeEdgeCases:
    """测试边界情况"""
    
    def test_empty_writable_batch_mode(self):
        """测试空的 writable_batch_mode"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                # 进入但不做任何操作
                with npk.writable_batch_mode() as wb:
                    pass
                
                # 验证数据未变
                result = npk.load('data')
                assert np.allclose(result, data)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_load_nonexistent_array(self):
        """测试加载不存在的数组"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.writable_batch_mode() as wb:
                    # 尝试加载不存在的数组
                    with pytest.raises(KeyError):
                        wb.load('nonexistent')
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_single_element_modification(self):
        """测试单个元素修改"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.writable_batch_mode() as wb:
                    arr = wb.load('data')
                    
                    # 只修改一个元素
                    arr[50, 5] = 888.0
                
                # 验证
                result = npk.load('data')
                assert result[50, 5] == 888.0
                # 其他元素不变
                mask = np.ones((100, 10), dtype=bool)
                mask[50, 5] = False
                assert np.allclose(result[mask], data[mask])
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_very_large_array(self):
        """测试超大数组（零内存开销的优势）"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                # 创建一个较大的数组（~200MB）
                data = np.random.rand(25000, 1000).astype(np.float32)
                npk.save({'huge': data})
                
                with npk.writable_batch_mode() as wb:
                    arr = wb.load('huge')
                    
                    # 即使是超大数组，也能零拷贝修改
                    arr[:100] *= 2.0
                
                # 验证部分修改生效
                result = npk.load('huge')
                assert np.allclose(result[:100], data[:100] * 2.0)
                assert np.allclose(result[100:], data[100:])
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

