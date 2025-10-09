"""测试NumPack的open和close方法"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from numpack import NumPack


class TestOpenClose:
    """测试open和close方法的功能"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp = tempfile.mkdtemp()
        yield temp
        if Path(temp).exists():
            shutil.rmtree(temp)
    
    def test_no_auto_open(self, temp_dir):
        """测试默认行为：文件不自动打开"""
        npk_path = Path(temp_dir) / "test_no_auto_open.npk"
        
        # 创建实例时不应该自动打开
        npk = NumPack(npk_path)
        assert not npk.is_opened
        assert npk.is_closed
        
        # 必须手动打开才能使用
        npk.open()
        assert npk.is_opened
        assert not npk.is_closed
        
        # 现在可以保存和加载数据
        test_data = {'array1': np.array([1, 2, 3])}
        npk.save(test_data)
        loaded = npk.load('array1')
        np.testing.assert_array_equal(loaded, test_data['array1'])
        
        npk.close()
        assert npk.is_closed
        assert not npk.is_opened
    
    def test_must_open_before_use(self, temp_dir):
        """测试必须先打开才能使用"""
        npk_path = Path(temp_dir) / "test_must_open.npk"
        
        # 创建实例
        npk = NumPack(npk_path)
        assert not npk.is_opened
        assert npk.is_closed
        
        # 尝试在未打开时使用会失败
        with pytest.raises(RuntimeError, match="not opened or has been closed"):
            npk.save({'array1': np.array([1, 2, 3])})
        
        with pytest.raises(RuntimeError, match="not opened or has been closed"):
            npk.get_member_list()
        
        # 手动打开后可以使用
        npk.open()
        assert npk.is_opened
        assert not npk.is_closed
        
        test_data = {'array1': np.array([1, 2, 3])}
        npk.save(test_data)
        loaded = npk.load('array1')
        np.testing.assert_array_equal(loaded, test_data['array1'])
        
        npk.close()
    
    def test_reopen_after_close(self, temp_dir):
        """测试关闭后重新打开"""
        npk_path = Path(temp_dir) / "test_reopen.npk"
        
        # 第一次使用
        npk = NumPack(npk_path)
        npk.open()
        test_data = {'array1': np.array([1, 2, 3, 4, 5])}
        npk.save(test_data)
        npk.close()
        
        assert npk.is_closed
        assert not npk.is_opened
        
        # 重新打开
        npk.open()
        assert npk.is_opened
        assert not npk.is_closed
        
        # 应该能够加载之前保存的数据
        loaded = npk.load('array1')
        np.testing.assert_array_equal(loaded, test_data['array1'])
        
        # 保存更多数据
        new_data = {'array2': np.array([10, 20, 30])}
        npk.save(new_data)
        
        # 验证两个数组都存在
        assert 'array1' in npk.get_member_list()
        assert 'array2' in npk.get_member_list()
        
        npk.close()
    
    def test_multiple_open_calls(self, temp_dir):
        """测试多次调用open是幂等的"""
        npk_path = Path(temp_dir) / "test_multiple_open.npk"
        
        npk = NumPack(npk_path)
        npk.open()
        npk.open()  # 第二次调用应该不会出错
        npk.open()  # 第三次调用也应该不会出错
        
        assert npk.is_opened
        
        test_data = {'array1': np.array([1, 2, 3])}
        npk.save(test_data)
        
        npk.close()
    
    def test_multiple_close_calls(self, temp_dir):
        """测试多次调用close是幂等的"""
        npk_path = Path(temp_dir) / "test_multiple_close.npk"
        
        npk = NumPack(npk_path)
        npk.open()
        test_data = {'array1': np.array([1, 2, 3])}
        npk.save(test_data)
        
        npk.close()
        npk.close()  # 第二次调用应该不会出错
        npk.close()  # 第三次调用也应该不会出错
        
        assert npk.is_closed
    
    def test_context_manager_auto_open(self, temp_dir):
        """测试context manager会自动打开文件"""
        npk_path = Path(temp_dir) / "test_context_auto_open.npk"
        
        # 创建实例时不自动打开
        npk = NumPack(npk_path)
        assert not npk.is_opened
        
        # 使用 context manager 会自动打开
        with npk as n:
            assert n.is_opened
            test_data = {'array1': np.array([1, 2, 3])}
            n.save(test_data)
        
        # 退出context后应该关闭
        assert npk.is_closed
    
    def test_context_manager_reopen(self, temp_dir):
        """测试context manager可以重新打开已关闭的文件"""
        npk_path = Path(temp_dir) / "test_context_reopen.npk"
        
        # 第一次使用
        with NumPack(npk_path) as npk:
            test_data = {'array1': np.array([1, 2, 3])}
            npk.save(test_data)
        
        # 文件应该已关闭
        assert npk.is_closed
        
        # 重新使用同一个实例
        with npk as n:
            assert n.is_opened
            loaded = n.load('array1')
            np.testing.assert_array_equal(loaded, test_data['array1'])
        
        assert npk.is_closed
    
    def test_open_close_cycle(self, temp_dir):
        """测试多次打开关闭循环"""
        npk_path = Path(temp_dir) / "test_open_close_cycle.npk"
        
        npk = NumPack(npk_path)
        
        for i in range(5):
            npk.open()
            assert npk.is_opened
            
            # 保存数据
            test_data = {f'array{i}': np.array([i, i+1, i+2])}
            npk.save(test_data)
            
            npk.close()
            assert npk.is_closed
        
        # 最后一次打开，验证所有数据都存在
        npk.open()
        members = npk.get_member_list()
        assert len(members) == 5
        for i in range(5):
            assert f'array{i}' in members
        npk.close()
    
    def test_drop_if_exists_with_manual_open(self, temp_dir):
        """测试drop_if_exists与手动打开配合使用"""
        npk_path = Path(temp_dir) / "test_drop.npk"
        
        # 第一次创建文件
        npk1 = NumPack(npk_path)
        npk1.open()
        npk1.save({'array1': np.array([1, 2, 3])})
        npk1.close()
        
        # 第二次打开，drop_if_exists=True
        npk2 = NumPack(npk_path, drop_if_exists=True)
        npk2.open()
        
        # 应该是空的
        assert len(npk2.get_member_list()) == 0
        
        npk2.save({'array2': np.array([4, 5, 6])})
        members = npk2.get_member_list()
        assert len(members) == 1
        assert 'array2' in members
        assert 'array1' not in members
        
        npk2.close()
    
    def test_error_after_close(self, temp_dir):
        """测试关闭后调用方法会抛出错误"""
        npk_path = Path(temp_dir) / "test_error_after_close.npk"
        
        npk = NumPack(npk_path)
        npk.open()
        test_data = {'array1': np.array([1, 2, 3])}
        npk.save(test_data)
        npk.close()
        
        # 所有操作都应该失败
        with pytest.raises(RuntimeError, match="not opened or has been closed"):
            npk.save({'array2': np.array([4, 5, 6])})
        
        with pytest.raises(RuntimeError, match="not opened or has been closed"):
            npk.load('array1')
        
        with pytest.raises(RuntimeError, match="not opened or has been closed"):
            npk.get_member_list()
    
    def test_properties(self, temp_dir):
        """测试is_opened和is_closed属性"""
        npk_path = Path(temp_dir) / "test_properties.npk"
        
        # 未打开状态
        npk = NumPack(npk_path)
        assert not npk.is_opened
        assert npk.is_closed
        
        # 打开状态
        npk.open()
        assert npk.is_opened
        assert not npk.is_closed
        
        # 关闭状态
        npk.close()
        assert not npk.is_opened
        assert npk.is_closed
        
        # 重新打开
        npk.open()
        assert npk.is_opened
        assert not npk.is_closed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

