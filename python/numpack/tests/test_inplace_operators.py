"""测试 LazyArray 的原地操作符支持"""

import pytest
import numpy as np
import numpack as npk
from pathlib import Path
import shutil


class TestInPlaceOperators:
    """测试 LazyArray 原地操作符"""
    
    @pytest.fixture
    def test_data_dir(self, tmp_path):
        """创建测试数据目录"""
        test_dir = tmp_path / "test_inplace"
        test_dir.mkdir(exist_ok=True)
        
        # 创建测试数据
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        with npk.NumPack(test_dir) as pack:
            pack.save({'array': data})
        
        yield test_dir
        
        # 清理
        if test_dir.exists():
            shutil.rmtree(test_dir)
    
    def test_imul(self, test_data_dir):
        """测试 *= 操作符"""
        with npk.NumPack(test_data_dir) as pack:
            a = pack.load('array', lazy=True)
            a *= 4.1
            
            # 结果应该是 numpy 数组
            assert isinstance(a, np.ndarray)
            
            # 验证结果值
            expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32) * 4.1
            np.testing.assert_allclose(a, expected, rtol=1e-5)
    
    def test_iadd(self, test_data_dir):
        """测试 += 操作符"""
        with npk.NumPack(test_data_dir) as pack:
            a = pack.load('array', lazy=True)
            a += 10
            
            assert isinstance(a, np.ndarray)
            expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32) + 10
            np.testing.assert_array_equal(a, expected)
    
    def test_isub(self, test_data_dir):
        """测试 -= 操作符"""
        with npk.NumPack(test_data_dir) as pack:
            a = pack.load('array', lazy=True)
            a -= 5
            
            assert isinstance(a, np.ndarray)
            expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32) - 5
            np.testing.assert_array_equal(a, expected)
    
    def test_itruediv(self, test_data_dir):
        """测试 /= 操作符"""
        with npk.NumPack(test_data_dir) as pack:
            a = pack.load('array', lazy=True)
            a /= 2
            
            assert isinstance(a, np.ndarray)
            expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32) / 2
            np.testing.assert_array_equal(a, expected)
    
    def test_ifloordiv(self, test_data_dir):
        """测试 //= 操作符"""
        with npk.NumPack(test_data_dir) as pack:
            a = pack.load('array', lazy=True)
            a //= 2
            
            assert isinstance(a, np.ndarray)
            expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32) // 2
            np.testing.assert_array_equal(a, expected)
    
    def test_imod(self, test_data_dir):
        """测试 %= 操作符"""
        with npk.NumPack(test_data_dir) as pack:
            a = pack.load('array', lazy=True)
            a %= 3
            
            assert isinstance(a, np.ndarray)
            expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32) % 3
            np.testing.assert_array_equal(a, expected)
    
    def test_ipow(self, test_data_dir):
        """测试 **= 操作符"""
        with npk.NumPack(test_data_dir) as pack:
            a = pack.load('array', lazy=True)
            a **= 2
            
            assert isinstance(a, np.ndarray)
            expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32) ** 2
            np.testing.assert_array_equal(a, expected)
    
    def test_chained_operations(self, test_data_dir):
        """测试连续的原地操作"""
        with npk.NumPack(test_data_dir) as pack:
            a = pack.load('array', lazy=True)
            a *= 2
            a += 10
            a /= 3
            
            assert isinstance(a, np.ndarray)
            
            # 验证结果
            expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
            expected = (expected * 2 + 10) / 3
            np.testing.assert_allclose(a, expected, rtol=1e-5)
    
    def test_with_numpy_array(self, test_data_dir):
        """测试与 numpy 数组的交互"""
        with npk.NumPack(test_data_dir) as pack:
            lazy_array = pack.load('array', lazy=True)
            numpy_array = np.array([1, 2, 3], dtype=np.float32)
            
            # LazyArray 与 numpy 数组相乘
            result = lazy_array
            result *= numpy_array
            
            assert isinstance(result, np.ndarray)
            
            # 验证广播是否正确
            expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32) * numpy_array
            np.testing.assert_array_equal(result, expected)
    
    def test_with_scalar_types(self, test_data_dir):
        """测试不同标量类型"""
        with npk.NumPack(test_data_dir) as pack:
            # 整数
            a = pack.load('array', lazy=True)
            a *= 2
            assert isinstance(a, np.ndarray)
            
            # 浮点数
            b = pack.load('array', lazy=True)
            b *= 2.5
            assert isinstance(b, np.ndarray)
            
            # 复数
            c = pack.load('array', lazy=True)
            c *= (1 + 2j)
            assert isinstance(c, np.ndarray)
    
    def test_original_lazyarray_unchanged(self, test_data_dir):
        """验证原始 LazyArray 不受影响"""
        with npk.NumPack(test_data_dir) as pack:
            # 第一次加载
            a = pack.load('array', lazy=True)
            original_data = np.array(a)
            
            # 应用原地操作
            a *= 2
            
            # 重新加载，验证文件中的数据没有改变
            b = pack.load('array', lazy=True)
            reloaded_data = np.array(b)
            
            np.testing.assert_array_equal(original_data, reloaded_data)

