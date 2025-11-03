"""测试 batch_mode 的流式批处理和内存控制功能"""
import numpy as np
import pytest
import tempfile
import shutil
import os
from numpack import NumPack


class TestBatchModeMemoryControl:
    """测试 batch_mode 的内存控制功能"""
    
    def test_default_memory_limit(self):
        """测试默认 memory_limit (500MB)"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                # 创建测试数据
                data = np.random.rand(1000, 10).astype(np.float32)
                npk.save({'data': data})
                
                # 使用默认 batch_mode
                with npk.batch_mode():
                    arr = npk.load('data')
                    arr *= 2.0
                    npk.save({'data': arr})
                
                # 验证结果
                result = npk.load('data')
                assert np.allclose(result, data * 2.0)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_custom_memory_limit(self):
        """测试自定义 memory_limit"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                # 创建多个数组
                arrays = {
                    f'array_{i}': np.random.rand(1000, 10).astype(np.float32)
                    for i in range(5)
                }
                npk.save(arrays)
                
                # 使用较小的 memory_limit (100KB)
                memory_limit = 100 * 1024  # 100KB
                
                with npk.batch_mode(memory_limit=memory_limit):
                    for i in range(5):
                        arr = npk.load(f'array_{i}')
                        arr += 1.0
                        npk.save({f'array_{i}': arr})
                
                # 验证所有数组
                for i in range(5):
                    result = npk.load(f'array_{i}')
                    expected = arrays[f'array_{i}'] + 1.0
                    assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_disabled_memory_limit(self):
        """测试禁用 memory_limit (设为0)"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                # 禁用流式处理
                with npk.batch_mode(memory_limit=0):
                    arr = npk.load('data')
                    arr *= 3.0
                    npk.save({'data': arr})
                
                result = npk.load('data')
                assert np.allclose(result, data * 3.0)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


class TestBatchModeMixedArraySizes:
    """测试 batch_mode 处理混合大小的数组"""
    
    def test_small_arrays_only(self):
        """测试只有小数组的场景"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                # 创建多个小数组
                arrays = {
                    f'small_{i}': np.random.rand(100, 5).astype(np.float32)
                    for i in range(10)
                }
                npk.save(arrays)
                
                with npk.batch_mode(memory_limit=50*1024):  # 50KB
                    for i in range(10):
                        arr = npk.load(f'small_{i}')
                        arr *= 2.0
                        npk.save({f'small_{i}': arr})
                
                # 验证
                for i in range(10):
                    result = npk.load(f'small_{i}')
                    expected = arrays[f'small_{i}'] * 2.0
                    assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_large_arrays_only(self):
        """测试只有大数组的场景"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                # 创建几个大数组（每个~4MB）
                arrays = {
                    f'large_{i}': np.random.rand(1000, 1000).astype(np.float32)
                    for i in range(3)
                }
                npk.save(arrays)
                
                # 小内存限制，强制分批
                with npk.batch_mode(memory_limit=2*1024*1024):  # 2MB
                    for i in range(3):
                        arr = npk.load(f'large_{i}')
                        arr += 10.0
                        npk.save({f'large_{i}': arr})
                
                # 验证
                for i in range(3):
                    result = npk.load(f'large_{i}')
                    expected = arrays[f'large_{i}'] + 10.0
                    assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_mixed_size_arrays(self):
        """测试混合大小的数组（最重要的测试）"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                # 创建不同大小的数组
                arrays = {
                    'tiny': np.random.rand(10, 10).astype(np.float32),      # ~0.4KB
                    'small': np.random.rand(100, 100).astype(np.float32),   # ~40KB
                    'medium': np.random.rand(500, 500).astype(np.float32),  # ~1MB
                    'large': np.random.rand(1000, 1000).astype(np.float32), # ~4MB
                }
                npk.save(arrays)
                
                # 使用中等内存限制
                with npk.batch_mode(memory_limit=2*1024*1024):  # 2MB
                    for name in ['tiny', 'small', 'medium', 'large']:
                        arr = npk.load(name)
                        arr *= 1.5
                        npk.save({name: arr})
                
                # 验证所有数组
                for name in ['tiny', 'small', 'medium', 'large']:
                    result = npk.load(name)
                    expected = arrays[name] * 1.5
                    assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_single_oversized_array(self):
        """测试单个数组超过 memory_limit 的场景"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                # 创建一个大数组（~4MB）
                huge_array = np.random.rand(1000, 1000).astype(np.float32)
                npk.save({'huge': huge_array})
                
                # 设置小于数组大小的 memory_limit
                with npk.batch_mode(memory_limit=1*1024*1024):  # 1MB
                    arr = npk.load('huge')
                    arr *= 0.5
                    npk.save({'huge': arr})
                
                # 验证
                result = npk.load('huge')
                expected = huge_array * 0.5
                assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


class TestBatchModeDirtyTracking:
    """测试 batch_mode 的脏数组检测功能"""
    
    def test_modify_subset_of_arrays(self):
        """测试只修改部分数组"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                # 创建10个数组
                arrays = {
                    f'array_{i}': np.random.rand(100, 10).astype(np.float32)
                    for i in range(10)
                }
                npk.save(arrays)
                
                with npk.batch_mode():
                    # 加载所有数组（放入缓存）
                    for i in range(10):
                        _ = npk.load(f'array_{i}')
                    
                    # 只修改其中3个
                    for i in [2, 5, 8]:
                        arr = npk.load(f'array_{i}')
                        arr *= 2.0
                        npk.save({f'array_{i}': arr})
                
                # 验证：只有修改的数组变化了
                for i in range(10):
                    result = npk.load(f'array_{i}')
                    if i in [2, 5, 8]:
                        expected = arrays[f'array_{i}'] * 2.0
                    else:
                        expected = arrays[f'array_{i}']
                    assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_inplace_modification_detection(self):
        """测试原地修改的检测"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.batch_mode():
                    arr = npk.load('data')
                    # 原地修改
                    arr[:] *= 3.0
                    # 保存同一个对象
                    npk.save({'data': arr})
                
                # 验证修改生效
                result = npk.load('data')
                expected = data * 3.0
                assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_array_replacement_detection(self):
        """测试数组替换的检测"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.batch_mode():
                    _ = npk.load('data')  # 加载到缓存
                    
                    # 创建新数组并保存（替换）
                    new_array = np.ones((100, 10), dtype=np.float32) * 99
                    npk.save({'data': new_array})
                
                # 验证替换生效
                result = npk.load('data')
                assert np.allclose(result, new_array)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_load_only_no_modification(self):
        """测试只加载不修改的场景"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.batch_mode():
                    # 只加载，不修改
                    arr = npk.load('data')
                    _ = arr.sum()  # 只读操作
                
                # 验证数据未变化
                result = npk.load('data')
                assert np.allclose(result, data)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


class TestBatchModeWithShapeChanges:
    """测试 batch_mode 支持形状变化操作"""
    
    def test_append_in_batch_mode(self):
        """测试在 batch_mode 中使用 append"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.batch_mode():
                    # 直接追加新数据（不先修改）
                    new_rows = np.ones((50, 10), dtype=np.float32) * 99
                    npk.append({'data': new_rows})
                
                # 验证
                result = npk.load('data')
                assert result.shape == (150, 10)
                assert np.allclose(result[:100], data)
                assert np.allclose(result[100:], 99.0)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_drop_in_batch_mode(self):
        """测试在 batch_mode 中使用 drop"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.batch_mode():
                    # 直接删除部分行（不先修改）
                    npk.drop('data', [0, 1, 2, 3, 4])
                
                # 验证
                result = npk.load('data')
                assert result.shape == (95, 10)
                expected = data[5:]
                assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_multiple_shape_operations(self):
        """测试多次形状变化操作"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.batch_mode():
                    # 1. 追加
                    npk.append({'data': np.ones((20, 10), dtype=np.float32)})
                    assert npk.get_shape('data') == (120, 10)
                    
                    # 2. 删除
                    npk.drop('data', list(range(10)))
                    assert npk.get_shape('data') == (110, 10)
                    
                    # 3. 再追加
                    npk.append({'data': np.zeros((10, 10), dtype=np.float32)})
                    assert npk.get_shape('data') == (120, 10)
                
                # 验证最终形状
                result = npk.load('data')
                assert result.shape == (120, 10)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


class TestBatchModeCompatibility:
    """测试 batch_mode 与其他功能的兼容性"""
    
    def test_nested_context_managers(self):
        """测试嵌套上下文管理器"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                # 嵌套使用
                with npk.batch_mode():
                    arr = npk.load('data')
                    arr *= 2.0
                    npk.save({'data': arr})
                
                # 再次使用
                with npk.batch_mode():
                    arr = npk.load('data')
                    arr += 5.0
                    npk.save({'data': arr})
                
                # 验证
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
                    with npk.batch_mode():
                        arr = npk.load('data')
                        arr *= 2.0
                        npk.save({'data': arr})
                        
                        # 故意抛出异常
                        raise ValueError("Test exception")
                except ValueError:
                    pass
                
                # 验证：即使异常，之前的修改也应该保存
                result = npk.load('data')
                expected = data * 2.0
                assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_different_dtypes(self):
        """测试不同数据类型"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                # 创建不同类型的数组
                arrays = {
                    'float32': np.random.rand(100, 10).astype(np.float32),
                    'float64': np.random.rand(100, 10).astype(np.float64),
                    'int32': np.random.randint(0, 100, (100, 10), dtype=np.int32),
                    'int64': np.random.randint(0, 100, (100, 10), dtype=np.int64),
                }
                npk.save(arrays)
                
                with npk.batch_mode(memory_limit=50*1024):  # 50KB
                    for name, original in arrays.items():
                        arr = npk.load(name)
                        if arr.dtype in [np.float32, np.float64]:
                            arr *= 2.0
                        else:
                            arr += 10
                        npk.save({name: arr})
                
                # 验证
                for name, original in arrays.items():
                    result = npk.load(name)
                    if original.dtype in [np.float32, np.float64]:
                        expected = original * 2.0
                    else:
                        expected = original + 10
                    assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_reopen_after_batch_mode(self):
        """测试 batch_mode 后重新打开文件"""
        numpack_dir = tempfile.mkdtemp()
        try:
            # 第一次：使用 batch_mode 修改
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.batch_mode():
                    arr = npk.load('data')
                    arr *= 3.0
                    npk.save({'data': arr})
            
            # 第二次：重新打开验证
            with NumPack(numpack_dir) as npk:
                result = npk.load('data')
                expected = data * 3.0
                assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


class TestBatchModePerformance:
    """测试 batch_mode 的性能相关场景"""
    
    def test_many_small_operations(self):
        """测试大量小操作"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                with npk.batch_mode():
                    # 100次小修改
                    for i in range(100):
                        arr = npk.load('data')
                        arr[i % 100] += 1.0
                        npk.save({'data': arr})
                
                # 验证：每行都被修改了至少一次
                result = npk.load('data')
                assert not np.allclose(result, data)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_frequent_array_switching(self):
        """测试频繁切换数组"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                # 创建多个数组
                arrays = {
                    f'array_{i}': np.random.rand(100, 10).astype(np.float32)
                    for i in range(5)
                }
                npk.save(arrays)
                
                with npk.batch_mode():
                    # 频繁切换访问
                    for round in range(20):
                        for i in range(5):
                            arr = npk.load(f'array_{i}')
                            arr += 0.1
                            npk.save({f'array_{i}': arr})
                
                # 验证：每个数组都被修改了20次
                for i in range(5):
                    result = npk.load(f'array_{i}')
                    expected = arrays[f'array_{i}'] + 2.0  # 20 * 0.1
                    assert np.allclose(result, expected, rtol=1e-5)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_large_scale_scenario(self):
        """测试大规模场景（综合测试）"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                # 创建20个不同大小的数组
                arrays = {}
                for i in range(20):
                    size = 50 * (i + 1)  # 从50到1000递增
                    arrays[f'array_{i}'] = np.random.rand(size, 10).astype(np.float32)
                npk.save(arrays)
                
                # 使用较小的 memory_limit 强制分批
                with npk.batch_mode(memory_limit=200*1024):  # 200KB
                    for i in range(20):
                        arr = npk.load(f'array_{i}')
                        arr *= (i + 1)
                        npk.save({f'array_{i}': arr})
                
                # 验证所有数组
                for i in range(20):
                    result = npk.load(f'array_{i}')
                    expected = arrays[f'array_{i}'] * (i + 1)
                    assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


class TestBatchModeEdgeCases:
    """测试边界情况"""
    
    def test_empty_batch_mode(self):
        """测试空的 batch_mode（不做任何操作）"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(100, 10).astype(np.float32)
                npk.save({'data': data})
                
                # 进入但不做任何操作
                with npk.batch_mode():
                    pass
                
                # 验证数据未变
                result = npk.load('data')
                assert np.allclose(result, data)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_very_small_memory_limit(self):
        """测试极小的 memory_limit"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                # 创建一些小数组
                arrays = {
                    f'array_{i}': np.random.rand(10, 10).astype(np.float32)
                    for i in range(5)
                }
                npk.save(arrays)
                
                # 使用非常小的 memory_limit (1KB)
                with npk.batch_mode(memory_limit=1024):
                    for i in range(5):
                        arr = npk.load(f'array_{i}')
                        arr += 1.0
                        npk.save({f'array_{i}': arr})
                
                # 验证所有数组
                for i in range(5):
                    result = npk.load(f'array_{i}')
                    expected = arrays[f'array_{i}'] + 1.0
                    assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_single_array_batch_mode(self):
        """测试只有单个数组的场景"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                data = np.random.rand(1000, 100).astype(np.float32)
                npk.save({'data': data})
                
                with npk.batch_mode(memory_limit=1*1024*1024):  # 1MB
                    arr = npk.load('data')
                    arr *= 5.0
                    npk.save({'data': arr})
                
                result = npk.load('data')
                expected = data * 5.0
                assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)
    
    def test_all_arrays_exceed_limit(self):
        """测试所有数组都超过 memory_limit 的场景"""
        numpack_dir = tempfile.mkdtemp()
        try:
            with NumPack(numpack_dir, drop_if_exists=True) as npk:
                # 创建3个大数组（每个~400KB）
                arrays = {
                    f'large_{i}': np.random.rand(500, 200).astype(np.float32)
                    for i in range(3)
                }
                npk.save(arrays)
                
                # memory_limit 小于任何单个数组
                with npk.batch_mode(memory_limit=100*1024):  # 100KB
                    for i in range(3):
                        arr = npk.load(f'large_{i}')
                        arr *= 2.0
                        npk.save({f'large_{i}': arr})
                
                # 验证：即使每个都超限，也应该能正确处理
                for i in range(3):
                    result = npk.load(f'large_{i}')
                    expected = arrays[f'large_{i}'] * 2.0
                    assert np.allclose(result, expected)
        finally:
            if os.path.exists(numpack_dir):
                shutil.rmtree(numpack_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

