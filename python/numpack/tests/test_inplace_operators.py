"""测试 LazyArray 的原地操作符支持"""

import pytest
import numpy as np
import numpack as npk
from pathlib import Path
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import conftest
ALL_DTYPES = conftest.ALL_DTYPES
create_test_array = conftest.create_test_array


class TestInPlaceOperators:
    """测试 LazyArray 原地操作符"""
    
    @pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
    def test_imul(self, tmp_path, dtype, test_values):
        """测试 *= 操作符（所有数据类型）"""
        test_dir = tmp_path / f"test_imul_{dtype.__name__}"
        test_dir.mkdir(exist_ok=True)
        
        data = create_test_array(dtype, (3, 3))
        with npk.NumPack(test_dir) as pack:
            pack.save({'array': data})
            a = pack.load('array', lazy=True)
            original = a.copy()
            
            # 根据类型选择标量
            if dtype == np.bool_:
                scalar = True
            elif np.issubdtype(dtype, np.integer):
                scalar = 2
            else:
                scalar = 4.1
            
            a *= scalar
            
            # 结果应该是 numpy 数组
            assert isinstance(a, np.ndarray)
            
            # 验证结果值
            expected = original * scalar
            if dtype == np.bool_:
                np.testing.assert_array_equal(a, expected)
            else:
                np.testing.assert_allclose(a, expected, rtol=1e-5)
    
    @pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
    def test_iadd(self, tmp_path, dtype, test_values):
        """测试 += 操作符（所有数据类型）"""
        test_dir = tmp_path / f"test_iadd_{dtype.__name__}"
        test_dir.mkdir(exist_ok=True)
        
        data = create_test_array(dtype, (3, 3))
        with npk.NumPack(test_dir) as pack:
            pack.save({'array': data})
            a = pack.load('array', lazy=True)
            original = a.copy()
            
            # 根据类型选择标量
            if dtype == np.bool_:
                scalar = True
            elif np.issubdtype(dtype, np.integer):
                scalar = 10
            else:
                scalar = 10.0
            
            a += scalar
            
            assert isinstance(a, np.ndarray)
            expected = original + scalar
            if dtype == np.bool_:
                np.testing.assert_array_equal(a, expected)
            else:
                np.testing.assert_allclose(a, expected)
    
    @pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
    def test_isub(self, tmp_path, dtype, test_values):
        """测试 -= 操作符（所有数据类型）"""
        if dtype == np.bool_:
            pytest.skip("In-place subtraction not supported for boolean types in numpy")

        test_dir = tmp_path / f"test_isub_{dtype.__name__}"
        test_dir.mkdir(exist_ok=True)
        
        data = create_test_array(dtype, (3, 3))
        with npk.NumPack(test_dir) as pack:
            pack.save({'array': data})
            a = pack.load('array', lazy=True)
            original = a.copy()
            
            # 根据类型选择标量
            if dtype == np.bool_:
                scalar = True
            elif np.issubdtype(dtype, np.integer):
                scalar = 5
            else:
                scalar = 5.0
            
            a -= scalar
            
            assert isinstance(a, np.ndarray)
            expected = original - scalar
            if dtype == np.bool_:
                np.testing.assert_array_equal(a, expected)
            else:
                np.testing.assert_allclose(a, expected)
    
    @pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
    def test_itruediv(self, tmp_path, dtype, test_values):
        """测试 /= 操作符（仅浮点和复数类型）"""
        if np.issubdtype(dtype, (np.integer, np.bool_)):
            pytest.skip("True division not applicable for integer/bool types")
        
        test_dir = tmp_path / f"test_itruediv_{dtype.__name__}"
        test_dir.mkdir(exist_ok=True)
        
        data = create_test_array(dtype, (3, 3))
        with npk.NumPack(test_dir) as pack:
            pack.save({'array': data})
            a = pack.load('array', lazy=True)
            original = a.copy()
            
            a /= 2.0
            
            assert isinstance(a, np.ndarray)
            expected = original / 2.0
            np.testing.assert_allclose(a, expected)
    
    @pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
    def test_ifloordiv(self, tmp_path, dtype, test_values):
        """测试 //= 操作符（仅整数类型）"""
        if not np.issubdtype(dtype, np.integer):
            pytest.skip("Floor division only applicable for integer types")
        
        test_dir = tmp_path / f"test_ifloordiv_{dtype.__name__}"
        test_dir.mkdir(exist_ok=True)
        
        data = create_test_array(dtype, (3, 3))
        with npk.NumPack(test_dir) as pack:
            pack.save({'array': data})
            a = pack.load('array', lazy=True)
            original = a.copy()
            
            a //= 2
            
            assert isinstance(a, np.ndarray)
            expected = original // 2
            np.testing.assert_array_equal(a, expected)
    
    @pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
    def test_imod(self, tmp_path, dtype, test_values):
        """测试 %= 操作符（仅整数类型）"""
        if not np.issubdtype(dtype, np.integer):
            pytest.skip("Modulo only applicable for integer types")
        
        test_dir = tmp_path / f"test_imod_{dtype.__name__}"
        test_dir.mkdir(exist_ok=True)
        
        data = create_test_array(dtype, (3, 3))
        with npk.NumPack(test_dir) as pack:
            pack.save({'array': data})
            a = pack.load('array', lazy=True)
            original = a.copy()
            
            a %= 3
            
            assert isinstance(a, np.ndarray)
            expected = original % 3
            np.testing.assert_array_equal(a, expected)
    
    @pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
    def test_ipow(self, tmp_path, dtype, test_values):
        """测试 **= 操作符（数值类型）"""
        if dtype == np.bool_:
            pytest.skip("Power operation not applicable for bool type")
        
        test_dir = tmp_path / f"test_ipow_{dtype.__name__}"
        test_dir.mkdir(exist_ok=True)
        
        data = create_test_array(dtype, (3, 3))
        with npk.NumPack(test_dir) as pack:
            pack.save({'array': data})
            a = pack.load('array', lazy=True)
            original = a.copy()
            
            a **= 2
            
            assert isinstance(a, np.ndarray)
            expected = original ** 2
            if np.issubdtype(dtype, np.complexfloating):
                np.testing.assert_allclose(a, expected)
            else:
                np.testing.assert_allclose(a, expected)
    
    @pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
    def test_chained_operations(self, tmp_path, dtype, test_values):
        """测试连续的原地操作（仅浮点类型）"""
        if np.issubdtype(dtype, (np.integer, np.bool_)):
            pytest.skip("Chained operations with division only applicable for floating point types")
        
        test_dir = tmp_path / f"test_chained_{dtype.__name__}"
        test_dir.mkdir(exist_ok=True)
        
        data = create_test_array(dtype, (3, 3))
        with npk.NumPack(test_dir) as pack:
            pack.save({'array': data})
            a = pack.load('array', lazy=True)
            original = a.copy()
            
            a *= 2
            a += 10.0
            a /= 3.0
            
            assert isinstance(a, np.ndarray)
            expected = (original * 2 + 10.0) / 3.0
            np.testing.assert_allclose(a, expected, rtol=1e-5)
    
    @pytest.mark.parametrize("dtype,test_values", ALL_DTYPES)
    def test_original_lazyarray_unchanged(self, tmp_path, dtype, test_values):
        """验证原始 LazyArray 不受影响（所有数据类型）"""
        test_dir = tmp_path / f"test_unchanged_{dtype.__name__}"
        test_dir.mkdir(exist_ok=True)
        
        data = create_test_array(dtype, (3, 3))
        with npk.NumPack(test_dir) as pack:
            pack.save({'array': data})
            
            # 第一次加载
            a = pack.load('array', lazy=True)
            original_data = np.array(a)
            
            # 应用原地操作
            if dtype == np.bool_:
                a |= True
            elif np.issubdtype(dtype, np.integer):
                a *= 2
            else:
                a *= 2.0
            
            # 重新加载，验证文件中的数据没有改变
            b = pack.load('array', lazy=True)
            reloaded_data = np.array(b)
            
            if dtype == np.bool_:
                np.testing.assert_array_equal(original_data, reloaded_data)
            else:
                np.testing.assert_allclose(original_data, reloaded_data)

