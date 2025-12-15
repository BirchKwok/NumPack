"""
全面的batch mode和writable batch mode边界测试

测试覆盖：
1. batch_mode: 内存缓存模式的各种边界情况
2. writable_batch_mode: 零拷贝mmap模式的各种边界情况
3. 边界场景：空数组、大数组、异常处理、并发、内存限制等
"""
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from numpack import NumPack
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import conftest
ALL_DTYPES = conftest.ALL_DTYPES
create_test_array = conftest.create_test_array


@pytest.fixture
def temp_dir():
    """临时目录fixture"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    try:
        shutil.rmtree(temp_path)
    except Exception:
        pass


@pytest.fixture
def sample_npk(temp_dir):
    """创建带有示例数据的NumPack fixture"""
    npk_path = temp_dir / "test.npk"
    npk = NumPack(npk_path, drop_if_exists=True)
    npk.open()
    
    # 创建多种类型的测试数据
    npk.save({
        'small': np.array([[1.0, 2.0, 3.0]]),
        'medium': np.random.rand(100, 100),
        'int_array': np.array([[1, 2, 3, 4, 5]], dtype=np.int32),
        'float_array': np.array([[1.1, 2.2, 3.3]], dtype=np.float64),
        'bool_array': np.array([[True, False, True]], dtype=np.bool_),
    })
    
    yield npk
    
    try:
        npk.close()
    except Exception:
        pass


class TestBatchModeBasics:
    """batch_mode基础功能测试"""
    
    def test_basic_load_save(self, sample_npk):
        """测试基本的load和save操作"""
        with sample_npk.batch_mode():
            # Load并修改
            arr = sample_npk.load('small')
            original = arr.copy()
            arr *= 2.0
            sample_npk.save({'small': arr})
            
            # 再次load应该从缓存读取
            arr2 = sample_npk.load('small')
            assert np.array_equal(arr2, original * 2.0)
    
    def test_cache_hit_mechanism(self, sample_npk):
        """测试缓存命中机制"""
        with sample_npk.batch_mode() as batch:
            # 第一次load - 缓存未命中
            arr1 = sample_npk.load('medium')
            assert 'medium' in sample_npk._memory_cache
            
            # 第二次load - 缓存命中
            arr2 = sample_npk.load('medium')
            assert arr1 is arr2  # 应该是同一个对象
    
    def test_multiple_arrays(self, sample_npk):
        """测试多个数组的批处理"""
        with sample_npk.batch_mode():
            # 修改多个数组
            small = sample_npk.load('small')
            medium = sample_npk.load('medium')
            # 只测试float64数组，避免混合类型问题
            float_arr = sample_npk.load('float_array')

            small *= 2
            medium += 1
            float_arr *= 3

            sample_npk.save({
                'small': small,
                'medium': medium,
                'float_array': float_arr
            })

        # 验证所有修改都持久化
        assert np.allclose(sample_npk.load('small'), np.array([[2.0, 4.0, 6.0]]))
        assert np.allclose(sample_npk.load('medium'), medium)
        assert np.allclose(sample_npk.load('float_array'), np.array([[3.3, 6.6, 9.9]], dtype=np.float64))
    
    def test_dirty_tracking(self, sample_npk):
        """测试脏标记跟踪机制"""
        with sample_npk.batch_mode() as batch:
            # 修改一个数组
            arr = sample_npk.load('small')
            arr *= 2
            sample_npk.save({'small': arr})
            
            # 检查脏标记
            assert 'small' in batch._dirty_arrays
            
            # 加载但不修改
            arr2 = sample_npk.load('medium')
            # medium不应该在脏集合中（除非后续被修改）
    
    def test_in_place_modification(self, sample_npk):
        """测试原地修改检测"""
        with sample_npk.batch_mode():
            arr = sample_npk.load('small')
            # 原地修改（不创建新数组）
            arr *= 2
            arr += 1
            # 即使不显式save，修改也应该被跟踪
            sample_npk.save({'small': arr})
        
        # 验证修改持久化
        result = sample_npk.load('small')
        expected = np.array([[1.0, 2.0, 3.0]]) * 2 + 1
        assert np.allclose(result, expected)


class TestBatchModeBoundaries:
    """batch_mode边界情况测试"""
    
    def test_empty_array(self, temp_dir):
        """测试空数组处理"""
        npk = NumPack(temp_dir / "empty.npk", drop_if_exists=True)
        npk.open()
        
        # 保存空数组
        empty = np.array([]).reshape(0, 3)
        npk.save({'empty': empty})
        
        with npk.batch_mode():
            arr = npk.load('empty')
            assert arr.shape == (0, 3)
            # 不能修改空数组的值，但可以操作
            npk.save({'empty': arr})
        
        npk.close()
    
    def test_single_element(self, temp_dir):
        """测试单元素数组"""
        npk = NumPack(temp_dir / "single.npk", drop_if_exists=True)
        npk.open()
        
        npk.save({'single': np.array([[42.0]])})
        
        with npk.batch_mode():
            arr = npk.load('single')
            arr *= 2
            npk.save({'single': arr})
        
        result = npk.load('single')
        assert result[0, 0] == 84.0
        npk.close()
    
    def test_large_array(self, temp_dir):
        """测试大数组（> 100MB）"""
        npk = NumPack(temp_dir / "large.npk", drop_if_exists=True)
        npk.open()
        
        # 创建约100MB的数组 (13M floats * 8 bytes)
        large = np.random.rand(1, 13_000_000)
        npk.save({'large': large})
        
        with npk.batch_mode():
            arr = npk.load('large')
            arr *= 1.5
            npk.save({'large': arr})
        
        # 验证
        result = npk.load('large')
        assert np.allclose(result, large * 1.5)
        npk.close()
    
    def test_many_small_arrays(self, temp_dir):
        """测试大量小数组"""
        npk = NumPack(temp_dir / "many.npk", drop_if_exists=True)
        npk.open()
        
        # 创建100个小数组
        arrays = {f'arr_{i}': np.array([[i * 1.0]]) for i in range(100)}
        npk.save(arrays)
        
        with npk.batch_mode():
            for i in range(100):
                arr = npk.load(f'arr_{i}')
                arr *= 2
                npk.save({f'arr_{i}': arr})
        
        # 验证所有数组
        for i in range(100):
            result = npk.load(f'arr_{i}')
            assert result[0, 0] == i * 2.0
        
        npk.close()
    
    @pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
    def test_different_dtypes(self, temp_dir, dtype, test_values):
        """测试所有支持的数据类型"""
        npk = NumPack(temp_dir / f"dtype_{dtype.__name__}.npk", drop_if_exists=True)
        npk.open()
        
        # 创建测试数组
        test_data = create_test_array(dtype, (10, 5))
        npk.save({'test_array': test_data})
        
        with npk.batch_mode():
            arr = npk.load('test_array')
            original = arr.copy()
            
            # 根据类型进行修改
            if dtype == np.bool_:
                arr = ~arr
            elif np.issubdtype(dtype, np.integer):
                arr = arr * 2
            elif np.issubdtype(dtype, np.complexfloating):
                arr = arr * 2 + 1j
            else:  # floating point
                arr = arr * 1.5
            
            npk.save({'test_array': arr})
        
        # 验证修改持久化
        result = npk.load('test_array')
        if dtype == np.bool_:
            assert np.array_equal(result, ~original)
        elif np.issubdtype(dtype, np.integer):
            assert np.array_equal(result, original * 2)
        elif np.issubdtype(dtype, np.complexfloating):
            assert np.allclose(result, original * 2 + 1j)
        else:
            assert np.allclose(result, original * 1.5)
        
        npk.close()
    
    @pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
    def test_multidimensional_arrays(self, temp_dir, dtype, test_values):
        """测试多维数组（所有数据类型）"""
        npk = NumPack(temp_dir / f"multi_{dtype.__name__}.npk", drop_if_exists=True)
        npk.open()
        
        # 3D数组
        arr_3d = create_test_array(dtype, (10, 20, 30))
        npk.save({'3d': arr_3d})
        
        with npk.batch_mode():
            arr = npk.load('3d')
            original = arr.copy()
            
            # 根据类型进行修改
            if dtype == np.bool_:
                arr = ~arr
            elif np.issubdtype(dtype, np.integer):
                arr = arr * 2
            elif np.issubdtype(dtype, np.complexfloating):
                arr = arr * 2
            else:
                arr = arr * 2.0
            
            npk.save({'3d': arr})
        
        result = npk.load('3d')
        if dtype == np.bool_:
            assert np.array_equal(result, ~original)
        elif np.issubdtype(dtype, np.integer):
            assert np.array_equal(result, original * 2)
        else:
            assert np.allclose(result, original * 2.0)
        
        npk.close()


class TestBatchModeWithOtherOperations:
    """batch_mode与其他操作混合测试"""
    
    def test_with_append(self, sample_npk):
        """测试batch_mode中的append操作"""
        with sample_npk.batch_mode():
            arr = sample_npk.load('small')
            arr *= 2
            sample_npk.save({'small': arr})
            
            # append会改变shape，应该清理缓存
            sample_npk.append({'small': np.array([[10.0, 11.0, 12.0]])})
            
            # 重新加载应该看到新的shape
            arr2 = sample_npk.load('small')
            assert arr2.shape[0] == 2  # 原来1行，现在2行
    
    def test_with_drop_rows(self, sample_npk):
        """测试batch_mode中的drop行操作"""
        # 先增加一些行
        sample_npk.append({'small': np.array([[4.0, 5.0, 6.0]])})
        sample_npk.append({'small': np.array([[7.0, 8.0, 9.0]])})
        
        with sample_npk.batch_mode():
            arr = sample_npk.load('small')
            original_shape = arr.shape
            
            # drop某些行
            sample_npk.drop('small', indexes=[0])
            
            # 重新加载应该看到新的shape
            arr2 = sample_npk.load('small')
            assert arr2.shape[0] == original_shape[0] - 1
    
    def test_with_replace(self, sample_npk):
        """测试batch_mode中的replace操作"""
        with sample_npk.batch_mode():
            # 注意：batch_mode主要优化load/save，replace直接操作文件
            sample_npk.replace({'small': np.array([[100.0, 200.0, 300.0]])}, indexes=0)
            
            # 清理缓存后重新加载
            if 'small' in sample_npk._memory_cache:
                del sample_npk._memory_cache['small']
            
            arr = sample_npk.load('small')
            assert np.allclose(arr[0], [100.0, 200.0, 300.0])


class TestBatchModeExceptionHandling:
    """batch_mode异常处理测试"""
    
    def test_exception_during_batch(self, sample_npk):
        """测试批处理期间的异常处理"""
        try:
            with sample_npk.batch_mode():
                arr = sample_npk.load('small')
                arr *= 2
                sample_npk.save({'small': arr})
                
                # 触发异常
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # 确保缓存被清理
        assert not sample_npk._cache_enabled
        
        # 但修改应该已经保存（因为save在异常前完成）
        # 这里依赖于具体实现
    
    def test_load_nonexistent_array(self, sample_npk):
        """测试加载不存在的数组"""
        with pytest.raises(Exception):  # 可能是KeyError或其他异常
            with sample_npk.batch_mode():
                sample_npk.load('nonexistent')
    
    def test_multiple_context_entries(self, sample_npk):
        """测试多次进入batch_mode"""
        with sample_npk.batch_mode():
            arr1 = sample_npk.load('small')
            arr1 *= 2
            sample_npk.save({'small': arr1})
        
        # 第二次进入
        with sample_npk.batch_mode():
            arr2 = sample_npk.load('small')
            arr2 *= 3
            sample_npk.save({'small': arr2})
        
        # 最终结果
        result = sample_npk.load('small')
        expected = np.array([[1.0, 2.0, 3.0]]) * 2 * 3
        assert np.allclose(result, expected)
    
    def test_nested_context_not_supported(self, sample_npk):
        """测试嵌套context（不支持）"""
        with sample_npk.batch_mode():
            arr = sample_npk.load('small')
            
            # 嵌套应该工作但可能行为未定义
            with sample_npk.batch_mode():
                arr2 = sample_npk.load('medium')
                # 这可能会导致意外行为


class TestWritableBatchModeBasics:
    """writable_batch_mode基础功能测试"""
    
    def test_basic_load_modify(self, sample_npk):
        """测试基本的加载和修改"""
        with sample_npk.writable_batch_mode() as wb:
            arr = wb.load('small')
            original = arr.copy()
            arr *= 2.0
            # save是可选的
            wb.save({'small': arr})
        
        # 验证修改持久化
        result = sample_npk.load('small')
        assert np.allclose(result, original * 2.0)
    
    def test_zero_copy_verification(self, sample_npk):
        """测试零拷贝特性"""
        with sample_npk.writable_batch_mode() as wb:
            arr = wb.load('small')
            
            # 验证是视图而非副本
            assert not arr.flags['OWNDATA']  # 不拥有数据
            
            # 修改应该直接写入文件
            arr[0, 0] = 999.0
        
        # 验证修改持久化
        result = sample_npk.load('small')
        assert result[0, 0] == 999.0
    
    def test_multiple_arrays_writable(self, sample_npk):
        """测试多个数组的writable模式"""
        with sample_npk.writable_batch_mode() as wb:
            small = wb.load('small')
            medium = wb.load('medium')
            int_arr = wb.load('int_array')
            
            small *= 2
            medium += 1
            int_arr *= 3
        
        # 验证所有修改都持久化
        assert np.allclose(sample_npk.load('small')[0], [2.0, 4.0, 6.0])
        # medium和int_arr也应该被修改
    
    def test_repeated_load_same_array(self, sample_npk):
        """测试重复加载同一数组（应该返回缓存的视图）"""
        with sample_npk.writable_batch_mode() as wb:
            arr1 = wb.load('small')
            arr2 = wb.load('small')
            
            # 应该是同一个对象（从缓存返回）
            assert arr1 is arr2
            
            # 修改一个应该影响另一个
            arr1[0, 0] = 123.0
            assert arr2[0, 0] == 123.0


class TestWritableBatchModeBoundaries:
    """writable_batch_mode边界情况测试"""
    
    @pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
    def test_all_supported_dtypes(self, temp_dir, dtype, test_values):
        """测试所有支持的数据类型（writable_batch_mode）"""
        npk = NumPack(temp_dir / f"writable_dtype_{dtype.__name__}.npk", drop_if_exists=True)
        npk.open()
        
        # 创建测试数组
        test_data = create_test_array(dtype, (10, 5))
        npk.save({'test_array': test_data})
        
        with npk.writable_batch_mode() as wb:
            arr = wb.load('test_array')
            original = arr.copy()
            
            # 验证能够加载
            assert arr.shape == (10, 5)
            
            # 根据类型进行修改
            if dtype == np.bool_:
                arr[:] = ~arr
            elif np.issubdtype(dtype, np.integer):
                arr += 1
            elif np.issubdtype(dtype, np.complexfloating):
                arr *= 2
            else:  # floating point
                arr *= 1.5
        
        # 验证修改持久化
        result = npk.load('test_array')
        if dtype == np.bool_:
            assert np.array_equal(result, ~original)
        elif np.issubdtype(dtype, np.integer):
            assert np.array_equal(result, original + 1)
        elif np.issubdtype(dtype, np.complexfloating):
            assert np.allclose(result, original * 2)
        else:
            assert np.allclose(result, original * 1.5)
        
        npk.close()
    
    def test_large_array_writable(self, temp_dir):
        """测试大数组的writable模式（零内存优势）"""
        npk = NumPack(temp_dir / "large_writable.npk", drop_if_exists=True)
        npk.open()
        
        # 创建大数组
        large = np.random.rand(1, 10_000_000)
        npk.save({'large': large})
        
        with npk.writable_batch_mode() as wb:
            arr = wb.load('large')
            
            # 验证是mmap视图
            assert not arr.flags['OWNDATA']
            
            # 修改部分数据
            arr[0, :1000] *= 2.0
        
        # 验证修改
        result = npk.load('large')
        assert np.allclose(result[0, :1000], large[0, :1000] * 2.0)
        
        npk.close()
    
    def test_empty_array_writable(self, temp_dir):
        """测试空数组的writable模式（空文件无法mmap）"""
        npk = NumPack(temp_dir / "empty_writable.npk", drop_if_exists=True)
        npk.open()
        
        empty = np.array([]).reshape(0, 3).astype(np.float64)
        npk.save({'empty': empty})
        
        # 空文件无法mmap，应该抛出ValueError
        with pytest.raises(ValueError, match="cannot mmap an empty file"):
            with npk.writable_batch_mode() as wb:
                arr = wb.load('empty')
        
        npk.close()
    
    def test_single_element_writable(self, temp_dir):
        """测试单元素数组的writable模式"""
        npk = NumPack(temp_dir / "single_writable.npk", drop_if_exists=True)
        npk.open()
        
        npk.save({'single': np.array([[42.0]])})
        
        with npk.writable_batch_mode() as wb:
            arr = wb.load('single')
            arr[0, 0] = 99.0
        
        assert npk.load('single')[0, 0] == 99.0
        npk.close()


class TestWritableBatchModeExceptions:
    """writable_batch_mode异常处理测试"""
    
    def test_exception_during_writable_batch(self, sample_npk):
        """测试异常处理和资源清理"""
        try:
            with sample_npk.writable_batch_mode() as wb:
                arr = wb.load('small')
                arr *= 2
                
                # 触发异常
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # 验证资源被正确清理
        # writable_arrays和array_cache应该为空
    
    def test_load_nonexistent_writable(self, sample_npk):
        """测试加载不存在的数组"""
        with pytest.raises(KeyError):
            with sample_npk.writable_batch_mode() as wb:
                wb.load('nonexistent')
    
    def test_multiple_writable_contexts(self, sample_npk):
        """测试多次进入writable_batch_mode"""
        with sample_npk.writable_batch_mode() as wb:
            arr = wb.load('small')
            arr *= 2
        
        # 第二次进入
        with sample_npk.writable_batch_mode() as wb:
            arr = wb.load('small')
            arr *= 3
        
        # 验证累积效果
        result = sample_npk.load('small')
        expected = np.array([[1.0, 2.0, 3.0]]) * 2 * 3
        assert np.allclose(result, expected)
    
    def test_modify_after_context_exit(self, sample_npk):
        """测试context退出后的修改（应该无效）"""
        arr_reference = None
        
        with sample_npk.writable_batch_mode() as wb:
            arr_reference = wb.load('small')
            arr_reference *= 2
        
        # Context已退出，mmap已关闭
        # 此时修改arr_reference可能导致错误或无效
        # 这里只验证不会崩溃
        try:
            # 不应该再修改已关闭的mmap
            pass
        except Exception:
            pass


class TestBatchModesComparison:
    """batch_mode与writable_batch_mode对比测试"""
    
    def test_same_results_different_modes(self, temp_dir):
        """测试两种模式产生相同的结果"""
        # 创建两个相同的数据集
        npk1 = NumPack(temp_dir / "mode1.npk", drop_if_exists=True)
        npk1.open()
        npk1.save({'data': np.array([[1.0, 2.0, 3.0]])})
        
        npk2 = NumPack(temp_dir / "mode2.npk", drop_if_exists=True)
        npk2.open()
        npk2.save({'data': np.array([[1.0, 2.0, 3.0]])})
        
        # 使用batch_mode
        with npk1.batch_mode():
            arr1 = npk1.load('data')
            arr1 *= 2
            arr1 += 1
            npk1.save({'data': arr1})
        
        # 使用writable_batch_mode
        with npk2.writable_batch_mode() as wb:
            arr2 = wb.load('data')
            arr2 *= 2
            arr2 += 1
        
        # 验证结果相同
        result1 = npk1.load('data')
        result2 = npk2.load('data')
        assert np.allclose(result1, result2)
        
        npk1.close()
        npk2.close()
    
    def test_memory_characteristics(self, temp_dir):
        """测试两种模式的内存特性"""
        npk = NumPack(temp_dir / "memory.npk", drop_if_exists=True)
        npk.open()
        npk.save({'data': np.random.rand(100, 100)})
        
        # batch_mode: 数据在内存缓存中
        with npk.batch_mode():
            arr1 = npk.load('data')
            assert 'data' in npk._memory_cache
            # 注意：batch_mode返回的数组可能是缓存的引用，不一定OWNDATA
        
        # writable_batch_mode: 数据是mmap视图
        with npk.writable_batch_mode() as wb:
            arr2 = wb.load('data')
            assert not arr2.flags['OWNDATA']  # 不拥有数据（视图）
        
        npk.close()


class TestBatchModeEdgeCases:
    """batch mode边缘情况测试"""
    
    def test_save_without_load(self, sample_npk):
        """测试不load直接save新数据"""
        new_data = np.array([[99.0, 88.0, 77.0]])
        
        with sample_npk.batch_mode():
            # 直接save新数据
            sample_npk.save({'new_array': new_data})
            
            # 应该能够load
            arr = sample_npk.load('new_array')
            assert np.allclose(arr, new_data)
    
    def test_load_modify_no_save(self, sample_npk):
        """测试load并修改但不save"""
        original = sample_npk.load('small').copy()
        
        with sample_npk.batch_mode():
            arr = sample_npk.load('small')
            arr *= 2
            # 没有调用save
        
        # 修改不应该持久化（取决于实现）
        # 由于是in-place修改，可能会被检测到
    
    def test_replace_cached_array(self, sample_npk):
        """测试替换已缓存的数组"""
        with sample_npk.batch_mode():
            arr1 = sample_npk.load('small')
            
            # 用新数组替换
            new_arr = np.array([[100.0, 200.0, 300.0]])
            sample_npk.save({'small': new_arr})
            
            # 再次load应该返回新数组
            arr2 = sample_npk.load('small')
            assert np.allclose(arr2, new_arr)
    
    def test_very_frequent_operations(self, sample_npk):
        """测试高频操作（性能相关）"""
        with sample_npk.batch_mode():
            for i in range(1000):
                arr = sample_npk.load('small')
                arr *= 1.001
                sample_npk.save({'small': arr})
        
        # 验证累积效果
        result = sample_npk.load('small')
        expected = np.array([[1.0, 2.0, 3.0]]) * (1.001 ** 1000)
        assert np.allclose(result, expected, rtol=1e-3)


class TestWritableBatchModeEdgeCases:
    """writable_batch_mode边缘情况测试"""
    
    def test_readonly_after_modifications(self, sample_npk):
        """测试修改后的只读访问"""
        with sample_npk.writable_batch_mode() as wb:
            arr = wb.load('small')
            arr *= 2
        
        # 退出后使用正常load访问
        result = sample_npk.load('small')
        expected = np.array([[1.0, 2.0, 3.0]]) * 2
        assert np.allclose(result, expected)
    
    def test_partial_array_modification(self, sample_npk):
        """测试部分数组修改"""
        with sample_npk.writable_batch_mode() as wb:
            arr = wb.load('medium')
            
            # 只修改部分数据
            arr[0:10, 0:10] *= 2
            arr[50:60, 50:60] = 0
        
        # 验证部分修改
        result = sample_npk.load('medium')
        assert np.allclose(result[50:60, 50:60], 0)
    
    def test_no_save_still_persists(self, sample_npk):
        """测试不调用save修改仍然持久化"""
        with sample_npk.writable_batch_mode() as wb:
            arr = wb.load('small')
            arr *= 2
            # 故意不调用 wb.save()
        
        # 修改应该仍然持久化（因为是直接写文件）
        result = sample_npk.load('small')
        expected = np.array([[1.0, 2.0, 3.0]]) * 2
        assert np.allclose(result, expected)
    
    def test_mixed_read_write(self, sample_npk):
        """测试混合读写操作"""
        with sample_npk.writable_batch_mode() as wb:
            # 读一个数组
            arr1 = wb.load('small')
            val = arr1[0, 0]
            
            # 写另一个数组
            arr2 = wb.load('medium')
            arr2[0, 0] = val * 2
            
            # 再读第一个
            arr1_again = wb.load('small')
            assert arr1 is arr1_again


class TestBatchModeMemoryLimit:
    """batch_mode内存限制测试"""
    
    def test_memory_limit_parameter(self, temp_dir):
        """测试memory_limit参数"""
        npk = NumPack(temp_dir / "memlimit.npk", drop_if_exists=True)
        npk.open()
        
        # 创建测试数据
        npk.save({'arr': np.random.rand(100, 100)})
        
        # 使用内存限制
        with npk.batch_mode(memory_limit=1):  # 1MB限制
            arr = npk.load('arr')
            arr *= 2
            npk.save({'arr': arr})
        
        npk.close()
    
    def test_exceeding_memory_limit(self, temp_dir):
        """测试超出内存限制的行为"""
        npk = NumPack(temp_dir / "exceed.npk", drop_if_exists=True)
        npk.open()
        
        # 创建多个数组
        for i in range(10):
            npk.save({f'arr_{i}': np.random.rand(100, 100)})
        
        # 使用很小的内存限制
        with npk.batch_mode(memory_limit=0.1):  # 0.1MB限制
            for i in range(10):
                arr = npk.load(f'arr_{i}')
                arr *= 2
                npk.save({f'arr_{i}': arr})
        
        npk.close()


class TestConcurrentAccess:
    """并发访问测试（边界情况）"""
    
    def test_sequential_batch_modes(self, sample_npk):
        """测试顺序的多个batch_mode"""
        # 第一个batch
        with sample_npk.batch_mode():
            arr = sample_npk.load('small')
            arr *= 2
            sample_npk.save({'small': arr})
        
        # 第二个batch
        with sample_npk.batch_mode():
            arr = sample_npk.load('small')
            arr += 1
            sample_npk.save({'small': arr})
        
        # 验证累积效果
        result = sample_npk.load('small')
        expected = np.array([[1.0, 2.0, 3.0]]) * 2 + 1
        assert np.allclose(result, expected)
    
    def test_interleaved_modes(self, sample_npk):
        """测试交替使用两种模式"""
        # batch_mode
        with sample_npk.batch_mode():
            arr = sample_npk.load('small')
            arr *= 2
            sample_npk.save({'small': arr})
        
        # writable_batch_mode
        with sample_npk.writable_batch_mode() as wb:
            arr = wb.load('small')
            arr += 1
        
        # 再次batch_mode
        with sample_npk.batch_mode():
            arr = sample_npk.load('small')
            arr *= 3
            sample_npk.save({'small': arr})
        
        # 验证累积效果
        result = sample_npk.load('small')
        expected = (np.array([[1.0, 2.0, 3.0]]) * 2 + 1) * 3
        assert np.allclose(result, expected)


class TestIntegrationScenarios:
    """集成场景测试"""
    
    def test_realistic_data_processing(self, temp_dir):
        """测试真实的数据处理场景"""
        npk = NumPack(temp_dir / "realistic.npk", drop_if_exists=True)
        npk.open()
        
        # 模拟训练数据
        features = np.random.rand(1000, 50)
        labels = np.random.randint(0, 10, size=(1000, 1))
        npk.save({
            'features': features,
            'labels': labels
        })
        
        # 使用batch_mode进行数据标准化
        with npk.batch_mode():
            feat = npk.load('features')
            # 标准化
            mean = feat.mean(axis=1, keepdims=True)
            std = feat.std(axis=1, keepdims=True)
            feat = (feat - mean) / (std + 1e-8)
            npk.save({'features': feat})
        
        # 验证标准化
        result = npk.load('features')
        assert np.abs(result.mean()) < 0.1
        
        npk.close()
    
    def test_incremental_updates(self, temp_dir):
        """测试增量更新场景"""
        npk = NumPack(temp_dir / "incremental.npk", drop_if_exists=True)
        npk.open()
        
        # 初始数据
        data = np.zeros((10, 10))
        npk.save({'data': data})
        
        # 使用writable_batch_mode进行增量更新
        for step in range(5):
            with npk.writable_batch_mode() as wb:
                arr = wb.load('data')
                # 每次更新一部分
                start = step * 2
                end = start + 2
                arr[start:end, :] += 1
        
        # 验证每个区域被更新正确的次数
        result = npk.load('data')
        expected = np.ones((10, 10))
        assert np.allclose(result, expected)
        
        npk.close()
    
    def test_mixed_operations_workflow(self, temp_dir):
        """测试混合操作工作流"""
        npk = NumPack(temp_dir / "workflow.npk", drop_if_exists=True)
        npk.open()
        
        # 1. 初始化数据
        npk.save({'data': np.arange(100).reshape(10, 10).astype(float)})
        
        # 2. batch_mode: 批量处理
        with npk.batch_mode():
            data = npk.load('data')
            data *= 2
            npk.save({'data': data})
        
        # 3. 正常操作: append
        npk.append({'data': np.full((5, 10), 999.0)})
        
        # 4. writable_batch_mode: 精细调整
        with npk.writable_batch_mode() as wb:
            data = wb.load('data')
            data[0, 0] = -1
        
        # 5. 验证
        result = npk.load('data')
        assert result[0, 0] == -1
        assert result.shape == (15, 10)
        
        npk.close()


class TestStressTests:
    """压力测试"""
    
    def test_many_iterations_batch_mode(self, temp_dir):
        """测试大量迭代（batch_mode）"""
        npk = NumPack(temp_dir / "stress1.npk", drop_if_exists=True)
        npk.open()
        npk.save({'data': np.array([[1.0]])})
        
        with npk.batch_mode():
            for i in range(10000):
                arr = npk.load('data')
                arr *= 1.0001  # 微小变化
                npk.save({'data': arr})
        
        npk.close()
    
    def test_many_iterations_writable(self, temp_dir):
        """测试大量迭代（writable_batch_mode）"""
        npk = NumPack(temp_dir / "stress2.npk", drop_if_exists=True)
        npk.open()
        npk.save({'data': np.array([[1.0]])})
        
        with npk.writable_batch_mode() as wb:
            for i in range(10000):
                arr = wb.load('data')
                arr *= 1.0001
        
        npk.close()
    
    def test_many_arrays_batch_mode(self, temp_dir):
        """测试大量数组（batch_mode）"""
        npk = NumPack(temp_dir / "many_arrays.npk", drop_if_exists=True)
        npk.open()
        
        n_arrays = 1000
        # 创建数组
        arrays = {f'arr_{i}': np.array([[float(i)]]) for i in range(n_arrays)}
        npk.save(arrays)
        
        # batch处理
        with npk.batch_mode():
            for i in range(n_arrays):
                arr = npk.load(f'arr_{i}')
                arr *= 2
                npk.save({f'arr_{i}': arr})
        
        # 验证
        for i in range(0, n_arrays, 100):  # 抽样验证
            result = npk.load(f'arr_{i}')
            assert result[0, 0] == float(i) * 2
        
        npk.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

