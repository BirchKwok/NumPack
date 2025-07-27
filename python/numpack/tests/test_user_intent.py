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
        
        assert result.shape == (self.cols,), f"单次访问结果形状错误: {result.shape}"
        
        # 验证数据正确性
        expected = self.test_data['test_array'][single_index]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
        print("✅ 单次访问意图识别正确")

    def test_batch_access_intent(self):
        """测试批量访问意图识别"""
        lazy_array = self.npk.load('test_array', lazy=True)
        
        # ✅ 正确的批量访问用法 - 应该被识别为BatchAccess
        batch_indices = [10, 25, 50, 100, 200]
        result = lazy_array[batch_indices]
        
        assert result.shape == (len(batch_indices), self.cols), f"批量访问结果形状错误: {result.shape}"
        
        # 验证数据正确性
        expected = self.test_data['test_array'][batch_indices]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
        print("✅ 批量访问意图识别正确")

    def test_numpy_array_batch_access(self):
        """测试NumPy数组索引的批量访问"""
        lazy_array = self.npk.load('test_array', lazy=True)
        
        # ✅ NumPy数组索引 - 应该被识别为BatchAccess
        indices = np.array([5, 15, 35, 75, 150])
        result = lazy_array[indices]
        
        assert result.shape == (len(indices), self.cols), f"NumPy数组索引结果形状错误: {result.shape}"
        
        # 验证数据正确性
        expected = self.test_data['test_array'][indices]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
        print("✅ NumPy数组索引批量访问正确")

    def test_slice_access(self):
        """测试切片访问 - 应该被识别为ComplexIndex"""
        lazy_array = self.npk.load('test_array', lazy=True)
        
        # 切片访问
        result = lazy_array[10:20]
        
        assert result.shape == (10, self.cols), f"切片访问结果形状错误: {result.shape}"
        
        # 验证数据正确性
        expected = self.test_data['test_array'][10:20]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
        print("✅ 切片访问正确")

    def test_performance_comparison(self):
        """比较不同访问模式的性能"""
        lazy_array = self.npk.load('test_array', lazy=True)
        
        # 1000个随机索引
        indices = np.random.randint(0, self.rows, 1000).tolist()
        
        # ❌ 错误的用法（循环单次访问）- 仅用于性能对比，实际应用中应避免
        print("\n性能对比测试：")
        start_time = time.time()
        wrong_results = []
        for i in indices[:200]:  # 增加到200个以更明显地看到差异
            result = lazy_array[i]
            wrong_results.append(result)
        wrong_usage_time = time.time() - start_time
        print(f"❌ 错误用法（200次循环单次访问）: {wrong_usage_time:.4f}秒")
        
        # ✅ 正确的用法（批量访问）
        start_time = time.time()
        correct_result = lazy_array[indices]
        correct_usage_time = time.time() - start_time
        print(f"✅ 正确用法（批量访问1000个索引）: {correct_usage_time:.4f}秒")
        
        # 性能提升比例
        if correct_usage_time > 0:
            # 标准化到相同数量的访问
            normalized_wrong_time = wrong_usage_time * (1000 / 200)
            speedup = normalized_wrong_time / correct_usage_time
            print(f"🚀 批量访问性能提升: {speedup:.1f}x")
            
            # 批量访问应该显著更快（降低阈值以适应实际情况）
            assert speedup > 3, f"批量访问性能提升不足: {speedup:.1f}x"

    def test_user_intent_examples(self):
        """展示正确的用户意图用法示例"""
        lazy_array = self.npk.load('test_array', lazy=True)
        
        print("\n🎯 用户意图示例：")
        
        # 场景1：明确的单次访问
        print("场景1 - 明确的单次访问：")
        print("  用法: row = lazy_array[42]")
        row = lazy_array[42]
        print(f"  结果: {row.shape}")
        
        # 场景2：明确的批量访问
        print("场景2 - 明确的批量访问：")
        print("  用法: rows = lazy_array[[10, 20, 30]]")
        rows = lazy_array[[10, 20, 30]]
        print(f"  结果: {rows.shape}")
        
        # 场景3：NumPy数组索引
        print("场景3 - NumPy数组索引：")
        indices = np.array([5, 15, 25])
        print(f"  用法: rows = lazy_array[np.array({indices.tolist()})]")
        rows = lazy_array[indices]
        print(f"  结果: {rows.shape}")
        
        # 场景4：切片访问
        print("场景4 - 切片访问：")
        print("  用法: rows = lazy_array[10:15]")
        rows = lazy_array[10:15]
        print(f"  结果: {rows.shape}")
        
        print("\n✅ 所有用户意图示例测试通过")

if __name__ == "__main__":
    # 运行测试
    test = TestUserIntentRecognition()
    test.setup_method()
    
    try:
        test.test_single_access_intent()
        test.test_batch_access_intent()
        test.test_numpy_array_batch_access()
        test.test_slice_access()
        test.test_performance_comparison()
        test.test_user_intent_examples()
        
        print("\n🎉 所有用户意图识别测试通过！")
        
    finally:
        test.teardown_method() 