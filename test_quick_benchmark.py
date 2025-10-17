#!/usr/bin/env python3
"""
快速测试修改后的benchmark
"""

import sys
sys.path.insert(0, '/Users/guobingming/projects/NumPack')

from comprehensive_benchmark import BenchmarkRunner
import gc

print("="*90)
print("测试修改后的Benchmark（不包含恢复步骤的时间）")
print("="*90)

# 测试小数据集
runner = BenchmarkRunner("小数据集 (1K rows)", (1000, 10), repeat=3)
try:
    runner.run_all()
finally:
    runner.cleanup()

gc.collect()

print("\n" + "="*90)
print("测试完成!")
print("="*90)


