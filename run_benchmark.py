#!/usr/bin/env python3
"""
NumPack 性能基准测试运行脚本

使用方法:
    python run_benchmark.py [选项]

选项:
    --quick     快速测试 (较小的数据集)
    --full      完整测试 (默认)
    --output    输出文件名 (默认: Benchmark.md)
"""

import sys
import os
import argparse
from pathlib import Path

# 添加 numpack 路径
numpack_path = Path(__file__).parent / "python"
sys.path.insert(0, str(numpack_path))

def main():
    parser = argparse.ArgumentParser(description="NumPack vs NumPy 性能基准测试")
    parser.add_argument("--quick", action="store_true", help="运行快速测试 (较小数据集)")
    parser.add_argument("--full", action="store_true", help="运行完整测试 (默认)")
    parser.add_argument("--output", default="Benchmark.md", help="输出文件名")
    
    args = parser.parse_args()
    
    # 导入基准测试类
    try:
        from numpack.tests.comprehensive_benchmark import ComprehensiveBenchmark
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保在 NumPack 项目根目录下运行此脚本")
        sys.exit(1)
    
    # 创建基准测试实例
    benchmark = ComprehensiveBenchmark(output_file=args.output)
    
    # 根据参数调整测试规模
    if args.quick:
        print("运行快速基准测试...")
        # 重写测试方法使用较小的数据集
        original_methods = {}
        
        # 备份原始方法
        original_methods['test_io_performance'] = benchmark.test_io_performance
        original_methods['test_random_access_performance'] = benchmark.test_random_access_performance
        original_methods['test_streaming_performance'] = benchmark.test_streaming_performance
        original_methods['test_bulk_operations'] = benchmark.test_bulk_operations
        original_methods['test_memory_efficiency'] = benchmark.test_memory_efficiency
        original_methods['test_non_contiguous_access'] = benchmark.test_non_contiguous_access
        original_methods['test_different_dtypes'] = benchmark.test_different_dtypes
        original_methods['test_large_matrix_operations'] = benchmark.test_large_matrix_operations
        original_methods['test_compression_efficiency'] = benchmark.test_compression_efficiency
        
        # 使用较小的参数
        benchmark.test_io_performance = lambda: original_methods['test_io_performance'](rows=100000, cols=50)
        benchmark.test_random_access_performance = lambda: original_methods['test_random_access_performance'](rows=100000, cols=50, access_count=1000)
        benchmark.test_streaming_performance = lambda: original_methods['test_streaming_performance'](rows=100000, cols=50, chunk_size=5000)
        benchmark.test_bulk_operations = lambda: original_methods['test_bulk_operations'](rows=50000, cols=50)
        benchmark.test_memory_efficiency = lambda: original_methods['test_memory_efficiency'](rows=100000, cols=50)
        benchmark.test_non_contiguous_access = lambda: original_methods['test_non_contiguous_access'](rows=100000, cols=50, access_count=500)
        benchmark.test_different_dtypes = lambda: original_methods['test_different_dtypes'](rows=50000, cols=50)
        benchmark.test_large_matrix_operations = lambda: original_methods['test_large_matrix_operations'](rows=50000, cols=64)
        benchmark.test_compression_efficiency = lambda: original_methods['test_compression_efficiency'](rows=50000, cols=50)
    else:
        print("运行完整基准测试...")
    
    # 运行测试
    try:
        benchmark.run_all_tests()
        print(f"\n✅ 基准测试完成！")
        print(f"📊 详细报告已保存到: {args.output}")
        print(f"📁 当前目录: {os.getcwd()}")
        
        # 显示报告摘要
        if os.path.exists(args.output):
            print(f"\n📈 报告摘要:")
            with open(args.output, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # 查找性能总结部分
                in_summary = False
                summary_lines = []
                for line in lines:
                    if line.strip() == "## 性能总结":
                        in_summary = True
                        continue
                    elif in_summary and line.startswith("##"):
                        break
                    elif in_summary and line.strip():
                        summary_lines.append(line.strip())
                
                if summary_lines:
                    for line in summary_lines[:10]:  # 显示前10行摘要
                        print(f"  {line}")
                    if len(summary_lines) > 10:
                        print(f"  ... (查看完整报告获取更多详情)")
    
    except KeyboardInterrupt:
        print(f"\n⚠️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 