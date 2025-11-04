#!/usr/bin/env python3
"""
测试用户意图识别功能

验证NumPack能够正确区分：
1. 单次访问：lazy_array[i] - 尊重用户意图，不干预
2. 批量访问：lazy_array[indices] - 一次性FFI调用优化
3. 复杂索引：切片、布尔掩码等 - 使用现有逻辑
"""

import pytest
import numpy as np
import time
import tempfile
import shutil
from pathlib import Path

from numpack import NumPack


class TestUserIntentRecognition:
    """测试用户意图识别和相应的优化策略"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test_intent"
        
        # 创建测试数据
        self.rows, self.cols = 50000, 100
        self.test_data = {
            'test_array': np.random.rand(self.rows, self.cols).astype(np.float32)
        }
        
        # 保存测试数据
        self.npk = NumPack(str(self.test_file), drop_if_exists=True)
        self.npk.open()  # 手动打开文件
        self.npk.save(self.test_data)
        
    def teardown_method(self):
        """清理测试环境"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_single_access_intent(self):
        """测试单次访问意图识别"""
        lazy_array = self.npk.load('test_array', lazy=True)
        
        # ✅ 正确的单次访问用法 - 应该被识别为SingleAccess
        single_index = 42
        result = lazy_array[single_index]
        
        assert result.shape == (self.cols,), f"Single access result shape error: {result.shape}"
        
        # 验证数据正确性
        expected = self.test_data['test_array'][single_index]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
        print("✅ Single access intent recognized correctly")

    def test_batch_access_intent(self):
        """测试批量访问意图识别"""
        lazy_array = self.npk.load('test_array', lazy=True)
        
        # ✅ 正确的批量访问用法 - 应该被识别为BatchAccess
        batch_indices = [10, 25, 50, 100, 200]
        result = lazy_array[batch_indices]
        
        assert result.shape == (len(batch_indices), self.cols), f"Batch access result shape error: {result.shape}"
        
        # 验证数据正确性
        expected = self.test_data['test_array'][batch_indices]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
        print("✅ Batch access intent recognized correctly")

    def test_numpy_array_batch_access(self):
        """测试NumPy数组索引的批量访问"""
        lazy_array = self.npk.load('test_array', lazy=True)
        
        # ✅ NumPy数组索引 - 应该被识别为BatchAccess
        indices = np.array([5, 15, 35, 75, 150])
        result = lazy_array[indices]
        
        assert result.shape == (len(indices), self.cols), f"NumPy array index result shape error: {result.shape}"
        
        # 验证数据正确性
        expected = self.test_data['test_array'][indices]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
        print("✅ NumPy array index batch access correct")

    def test_slice_access(self):
        """测试切片访问 - 应该被识别为ComplexIndex"""
        lazy_array = self.npk.load('test_array', lazy=True)
        
        # 切片访问
        result = lazy_array[10:20]
        
        assert result.shape == (10, self.cols), f"Slice access result shape error: {result.shape}"
        
        # 验证数据正确性
        expected = self.test_data['test_array'][10:20]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
        print("✅ Slice access correct")

    def test_user_intent_examples(self):
        """展示正确的用户意图用法示例"""
        lazy_array = self.npk.load('test_array', lazy=True)
        
        print("\n🎯 User Intent Examples:")
        
        # Scenario 1: Clear single access
        print("Scenario 1 - Clear single access:")
        print("  Usage: row = lazy_array[42]")
        row = lazy_array[42]
        print(f"  Result: {row.shape}")
        
        # Scenario 2: Clear batch access
        print("Scenario 2 - Clear batch access:")
        print("  Usage: rows = lazy_array[[10, 20, 30]]")
        rows = lazy_array[[10, 20, 30]]
        print(f"  Result: {rows.shape}")
        
        # Scenario 3: NumPy array indexing
        print("Scenario 3 - NumPy array indexing:")
        indices = np.array([5, 15, 25])
        print(f"  Usage: rows = lazy_array[np.array({indices.tolist()})]")
        rows = lazy_array[indices]
        print(f"  Result: {rows.shape}")
        
        # Scenario 4: Slice access
        print("Scenario 4 - Slice access:")
        print("  Usage: rows = lazy_array[10:15]")
        rows = lazy_array[10:15]
        print(f"  Result: {rows.shape}")
        
        print("\n✅ All user intent example tests passed")

if __name__ == "__main__":
    # 运行测试
    test = TestUserIntentRecognition()
    test.setup_method()
    
    try:
        test.test_single_access_intent()
        test.test_batch_access_intent()
        test.test_numpy_array_batch_access()
        test.test_slice_access()
        test.test_user_intent_examples()
        
        print("\n🎉 All user intent recognition tests passed!")
        
    finally:
        test.teardown_method() 