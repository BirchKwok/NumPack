"""
Cross-platform consistency tests

确保NumPack在Windows、macOS和Linux上行为一致
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys

from numpack import NumPack, get_backend_info


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestCrossPlatformConsistency:
    """验证跨平台一致性"""
    
    def test_same_backend_all_platforms(self):
        """所有平台应该使用相同的backend（Rust）"""
        info = get_backend_info()
        # 验证后端类型
        assert info['backend_type'] == 'rust'
    
    def test_file_format_compatibility(self, temp_dir):
        """文件格式应该跨平台兼容"""
        npk_path = temp_dir / "test.npk"
        
        test_arrays = {
            'int32': np.arange(100, dtype=np.int32),
            'float64': np.random.randn(50, 50),
            'bool': np.array([True, False, True, False]),
            'uint8': np.arange(256, dtype=np.uint8),
        }
        
        # 保存
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save(test_arrays)
        
        # 加载并验证
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            for name, expected in test_arrays.items():
                loaded = npk.load(name)
                assert np.array_equal(loaded, expected), \
                    f"数组'{name}'在保存/加载后不相等"
    
    def test_context_manager_behavior(self, temp_dir):
        """Context manager应该在所有平台上行为一致"""
        npk_path = temp_dir / "test.npk"
        
        # 创建并使用context manager
        with NumPack(str(npk_path), strict_context_mode=True, warn_no_context=False) as npk:
            npk.save({'data': np.arange(10)})
            result = npk.load('data')
            assert len(result) == 10
        
        # Context退出后操作应该失败（所有平台）
        with pytest.raises(RuntimeError, match="closed"):
            npk.save({'more': np.arange(5)})
    
    def test_strict_mode_consistent(self, temp_dir):
        """Strict mode在所有平台行为一致"""
        npk_path = temp_dir / "test.npk"
        
        npk = NumPack(str(npk_path), strict_context_mode=True, warn_no_context=False)
        npk.open()  # 需要先打开才能测试strict模式
        
        # 所有平台都应该阻止非context使用
        with pytest.raises(RuntimeError, match="strict context mode"):
            npk.save({'data': np.array([1, 2, 3])})
        
        npk.close()
    
    def test_data_types_consistency(self, temp_dir):
        """所有数据类型在所有平台上一致"""
        npk_path = temp_dir / "types.npk"
        
        # 测试各种数据类型
        test_data = {
            'bool': np.array([True, False, True], dtype=np.bool_),
            'int8': np.array([-128, 0, 127], dtype=np.int8),
            'int16': np.array([-32768, 0, 32767], dtype=np.int16),
            'int32': np.array([-2147483648, 0, 2147483647], dtype=np.int32),
            'int64': np.array([-9223372036854775808, 0, 9223372036854775807], dtype=np.int64),
            'uint8': np.array([0, 128, 255], dtype=np.uint8),
            'uint16': np.array([0, 32768, 65535], dtype=np.uint16),
            'uint32': np.array([0, 2147483648, 4294967295], dtype=np.uint32),
            'float32': np.array([0.0, 3.14, -2.71], dtype=np.float32),
            'float64': np.array([0.0, 3.141592653589793, -2.718281828459045], dtype=np.float64),
        }
        
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save(test_data)
            
            for name, expected in test_data.items():
                loaded = npk.load(name)
                if np.issubdtype(expected.dtype, np.floating):
                    assert np.allclose(loaded, expected), f"{name} data type mismatch"
                else:
                    assert np.array_equal(loaded, expected), f"{name} data type mismatch"
    
    def test_lazy_loading_consistency(self, temp_dir):
        """LazyArray在所有平台上行为一致"""
        npk_path = temp_dir / "lazy.npk"
        data = np.arange(1000).reshape(100, 10)
        
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save({'data': data})
            
            # Lazy加载
            lazy = npk.load('data', lazy=True)
            
            # 各种索引方式
            assert np.array_equal(lazy[0], data[0])
            assert np.array_equal(lazy[0:10], data[0:10])
            assert np.array_equal(lazy[[1, 5, 10]], data[[1, 5, 10]])
    
    def test_metadata_consistency(self, temp_dir):
        """元数据在所有平台上一致"""
        npk_path = temp_dir / "meta.npk"
        
        test_data = {
            'array1': np.random.randn(100, 50),
            'array2': np.random.randn(200, 30),
        }
        
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save(test_data)
            
            # 检查元数据
            members = npk.get_member_list()
            assert set(members) == {'array1', 'array2'}
            
            assert npk.get_shape('array1') == (100, 50)
            assert npk.get_shape('array2') == (200, 30)
            
            assert npk.has_array('array1')
            assert npk.has_array('array2')
            assert not npk.has_array('nonexistent')


class TestPerformanceConsistency:
    """测试性能一致性"""
    
    def test_backend_info(self):
        """验证backend信息正确"""
        info = get_backend_info()
        
        assert 'backend_type' in info
        assert 'platform' in info
        assert 'is_windows' in info
        assert 'version' in info
        

        
        # platform应该是有效值
        assert info['platform'] in ['Windows', 'Darwin', 'Linux']
        
        # is_windows应该与platform一致
        assert info['is_windows'] == (info['platform'] == 'Windows')
    
    def test_batch_operations_work(self, temp_dir):
        """验证批量操作在所有平台工作"""
        npk_path = temp_dir / "batch.npk"
        data = np.arange(10000).reshape(1000, 10)
        
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save({'data': data})
            
            # 批量索引访问
            indices = np.array([0, 10, 50, 100, 500, 999])
            result = npk.load('data', lazy=True)[indices]
            
            expected = data[indices]
            assert np.array_equal(result, expected)


class TestWarningBehavior:
    """测试警告行为"""
    
    def test_warning_on_windows_without_context(self, temp_dir):
        """测试Windows上不使用context manager的警告"""
        npk_path = temp_dir / "warn_test.npk"
        
        # 在Windows上应该警告，其他平台不警告（除非显式设置）
        if sys.platform.startswith('win'):
            with pytest.warns(UserWarning, match="strict context mode"):
                npk = NumPack(str(npk_path))
                npk.close()
        else:
            # 非Windows平台默认不警告
            npk = NumPack(str(npk_path))  # 不应该警告
            npk.close()
    
    def test_warning_suppression(self, temp_dir):
        """测试警告可以被抑制"""
        npk_path = temp_dir / "no_warn.npk"
        
        # 显式设置warn_no_context=False应该不警告
        npk = NumPack(str(npk_path), warn_no_context=False)
        npk.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

