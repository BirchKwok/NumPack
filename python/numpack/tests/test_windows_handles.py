"""
Windows-specific tests for file handle management and context managers

这些测试主要在Windows平台运行，验证句柄管理和context manager的正确性。
在非Windows平台上会跳过某些测试，但仍会运行通用的context manager测试。
"""

import pytest
import numpy as np
import sys
import shutil
import tempfile
import time
from pathlib import Path

# Import NumPack
from numpack import NumPack, get_backend_info


# 只在Windows上运行某些特定测试
windows_only = pytest.mark.skipif(
    not sys.platform.startswith('win'),
    reason="Windows-specific tests"
)


@pytest.fixture
def temp_dir():
    """创建临时目录用于测试"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestContextManagerBasic:
    """测试基本的context manager功能（所有平台）"""
    
    def test_context_manager_basic_usage(self, temp_dir):
        """验证context manager基本使用"""
        npk_path = temp_dir / "test.npk"
        test_data = np.arange(100, dtype=np.int32)
        
        # 使用context manager
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save({'data': test_data})
            loaded = npk.load('data')
            assert np.array_equal(loaded, test_data)
        
        # 验证文件已创建
        assert npk_path.exists()
    
    def test_context_manager_auto_cleanup(self, temp_dir):
        """验证context manager自动清理"""
        npk_path = temp_dir / "test.npk"
        test_data = np.arange(100, dtype=np.int32)
        
        # 使用context manager
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save({'data': test_data})
        
        # Context退出后，应该能够立即删除（在支持的平台上）
        # 注意：Windows可能需要短暂延迟
        if sys.platform.startswith('win'):
            time.sleep(0.1)
        
        try:
            shutil.rmtree(npk_path)
            assert not npk_path.exists()
        except PermissionError:
            # Windows上可能需要更长时间
            time.sleep(0.5)
            shutil.rmtree(npk_path)
            assert not npk_path.exists()
    
    def test_nested_context_managers(self, temp_dir):
        """测试嵌套context manager"""
        npk_path1 = temp_dir / "test1.npk"
        npk_path2 = temp_dir / "test2.npk"
        
        with NumPack(str(npk_path1), warn_no_context=False) as npk1:
            npk1.save({'data1': np.arange(100)})
            
            with NumPack(str(npk_path2), warn_no_context=False) as npk2:
                npk2.save({'data2': np.arange(200)})
                loaded2 = npk2.load('data2')
                assert len(loaded2) == 200
            
            # npk2应该已关闭，npk1仍然打开
            npk1.save({'data3': np.arange(50)})
            loaded1 = npk1.load('data1')
            assert len(loaded1) == 100


class TestWindowsHandleManagement:
    """测试Windows平台的句柄管理（Windows优先）"""
    
    @windows_only
    def test_rapid_create_delete_windows(self, temp_dir):
        """测试快速创建和删除文件（Windows特定）"""
        for i in range(20):  # 减少迭代次数以加快测试
            npk_path = temp_dir / f"test_{i}.npk"
            
            with NumPack(str(npk_path), warn_no_context=False) as npk:
                npk.save({'data': np.arange(100)})
            
            # 在Windows上应该能够立即删除
            time.sleep(0.05)  # 小延迟确保清理完成
            shutil.rmtree(npk_path)
            assert not npk_path.exists()
    
    @windows_only
    def test_manual_close_releases_handles_windows(self, temp_dir):
        """测试手动close()释放句柄（Windows特定）"""
        npk_path = temp_dir / "test.npk"
        
        npk = NumPack(str(npk_path), warn_no_context=False)
        npk.open()  # 需要先打开
        npk.save({'data': np.arange(100)})
        
        # 显式close
        npk.close()
        
        # 应该能够立即删除；如果失败则重试
        try:
            shutil.rmtree(npk_path)
        except PermissionError:
            # Windows上极少数情况可能需要短暂延迟
            time.sleep(0.1)
            shutil.rmtree(npk_path)
        
        assert not npk_path.exists()
    
    def test_lazy_array_context_manager(self, temp_dir):
        """测试LazyArray的context manager（所有平台）"""
        npk_path = temp_dir / "test.npk"
        large_data = np.arange(10000, dtype=np.float64)
        
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save({'large': large_data})
            
            # 直接使用LazyArray（它自己支持context manager）
            lazy_arr = npk.load('large', lazy=True)
            result = lazy_arr[0:100]
            assert len(result) == 100
        
        # 清理应该成功
        if sys.platform.startswith('win'):
            time.sleep(0.1)
        shutil.rmtree(npk_path)
        assert not npk_path.exists()


class TestStrictContextMode:
    """测试严格context模式"""
    
    def test_strict_mode_prevents_non_context_usage(self, temp_dir):
        """测试strict mode阻止非context使用"""
        npk_path = temp_dir / "test.npk"
        
        # 创建strict mode实例
        npk = NumPack(str(npk_path), strict_context_mode=True, warn_no_context=False)
        npk.open()  # 需要先打开才能测试strict模式
        
        # 应该在使用outside context时抛出错误
        with pytest.raises(RuntimeError, match="strict context mode"):
            npk.save({'data': np.array([1, 2, 3])})
        
        # 清理
        npk.close()
    
    def test_strict_mode_works_in_context(self, temp_dir):
        """测试strict mode在context中正常工作"""
        npk_path = temp_dir / "test.npk"
        
        # 在context中应该正常工作
        with NumPack(str(npk_path), strict_context_mode=True, warn_no_context=False) as npk:
            npk.save({'data': np.array([1, 2, 3])})
            result = npk.load('data')
            assert np.array_equal(result, np.array([1, 2, 3]))
    
    def test_operations_after_close_raise_error(self, temp_dir):
        """测试close后操作抛出错误"""
        npk_path = temp_dir / "test.npk"
        
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save({'data': np.array([1, 2, 3])})
        
        # Context退出后应该已关闭
        with pytest.raises(RuntimeError, match="closed"):
            npk.save({'more': np.array([4, 5, 6])})
    
    def test_multiple_close_calls_safe(self, temp_dir):
        """测试多次调用close()是安全的"""
        npk_path = temp_dir / "test.npk"
        
        npk = NumPack(str(npk_path), warn_no_context=False)
        npk.open()  # 手动打开文件
        npk.save({'data': np.arange(10)})
        
        # 多次close应该不抛出错误
        npk.close()
        npk.close()
        npk.close()


class TestExceptionHandling:
    """测试异常处理"""
    
    def test_exception_during_context_still_cleans_up(self, temp_dir):
        """测试异常不阻止cleanup"""
        npk_path = temp_dir / "test.npk"
        
        try:
            with NumPack(str(npk_path), warn_no_context=False) as npk:
                npk.save({'data': np.arange(100)})
                raise ValueError("Test exception")
        except ValueError:
            pass  # 预期的异常
        
        # 应该仍然能够清理
        if sys.platform.startswith('win'):
            time.sleep(0.1)
        shutil.rmtree(npk_path)
        assert not npk_path.exists()
    
    def test_close_is_idempotent(self, temp_dir):
        """测试close是幂等的"""
        npk_path = temp_dir / "test.npk"
        
        npk = NumPack(str(npk_path), warn_no_context=False)
        npk.open()  # 手动打开文件
        npk.save({'data': np.array([1, 2, 3])})
        
        # 第一次close
        npk.close()
        
        # 后续close不应该出错
        for _ in range(5):
            npk.close()  # 应该安全


class TestBackendConsistency:
    """测试后端一致性"""
    
    def test_backend_available(self):
        """验证NumPack后端可用"""
        info = get_backend_info()
        
        # 验证后端信息可获取
        assert 'backend_type' in info
        assert 'platform' in info
        assert 'version' in info
        
        # 验证后端类型一致（现在只有Rust后端）
        assert info['backend_type'] == 'rust'
    
    def test_large_file_operations(self, temp_dir):
        """测试大文件操作可靠性"""
        npk_path = temp_dir / "large.npk"
        
        # 创建较大数据集
        large_array = np.random.randn(1000, 100).astype(np.float64)
        
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save({'large': large_array})
            loaded = npk.load('large')
            
            assert np.allclose(loaded, large_array)
        
        # 验证cleanup
        if sys.platform.startswith('win'):
            time.sleep(0.1)
        shutil.rmtree(npk_path)


class TestResourceManagement:
    """测试资源管理"""
    
    def test_multiple_instances_same_file(self, temp_dir):
        """测试多个实例访问同一文件"""
        npk_path = temp_dir / "shared.npk"
        
        # 第一个实例写入
        with NumPack(str(npk_path), warn_no_context=False) as npk1:
            npk1.save({'data': np.array([1, 2, 3])})
        
        # 第二个实例读取
        with NumPack(str(npk_path), warn_no_context=False) as npk2:
            result = npk2.load('data')
            assert np.array_equal(result, np.array([1, 2, 3]))
    
    def test_lazy_array_multiple_access(self, temp_dir):
        """测试LazyArray多次访问"""
        npk_path = temp_dir / "test.npk"
        data = np.arange(1000, dtype=np.float32)
        
        with NumPack(str(npk_path), warn_no_context=False) as npk:
            npk.save({'data': data})
            
            lazy = npk.load('data', lazy=True)
            
            # 多次访问同一LazyArray
            for _ in range(10):
                result = lazy[0:10]
                assert len(result) == 10
    
    @windows_only
    def test_windows_handle_cleanup_stress(self, temp_dir):
        """Windows句柄清理压力测试"""
        # 快速创建和销毁多个NumPack实例
        for i in range(50):
            npk_path = temp_dir / f"stress_{i}.npk"
            
            with NumPack(str(npk_path), warn_no_context=False) as npk:
                npk.save({'data': np.random.randn(100, 10)})
                loaded = npk.load('data')
                assert loaded.shape == (100, 10)
            
            # 每10次清理一次
            if i % 10 == 0:
                time.sleep(0.05)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

