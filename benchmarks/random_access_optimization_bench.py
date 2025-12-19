#!/usr/bin/env python3
"""
NumPack 随机访问优化基准测试
=============================

专门针对随机索引访问性能的基准测试，用于：
1. 记录优化前的baseline性能
2. 验证优化后是否有性能提升
3. 检测性能回归

测试场景：
- 小批量随机访问 (100 indices)
- 中批量随机访问 (1K indices)  
- 大批量随机访问 (10K indices)
- 超大批量随机访问 (100K indices)
- 不同数据规模 (100K, 1M rows)
- 不同访问模式 (纯随机、聚簇、稀疏)
"""

import os
import sys
import json
import time
import shutil
import tempfile
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

# 确保可以导入numpack
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from numpack import NumPack
    NUMPACK_AVAILABLE = True
except ImportError as e:
    print(f"NumPack not available: {e}")
    NUMPACK_AVAILABLE = False
    sys.exit(1)


class RandomAccessBenchmark:
    """随机访问性能基准测试"""
    
    def __init__(self, temp_dir: Path, results_dir: Path):
        self.temp_dir = temp_dir
        self.results_dir = results_dir
        self.results: Dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "python_version": sys.version,
                "platform": sys.platform,
            },
            "benchmarks": {}
        }
    
    def _time_operation(self, func, warmup: int = 1, repeat: int = 5) -> Dict[str, float]:
        """精确计时一个操作"""
        # Warmup
        for _ in range(warmup):
            func()
        
        gc.collect()
        
        # 实际测量
        times = []
        for _ in range(repeat):
            gc.collect()
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # 转换为毫秒
        
        return {
            "min_ms": min(times),
            "max_ms": max(times),
            "mean_ms": sum(times) / len(times),
            "median_ms": sorted(times)[len(times) // 2],
            "all_times_ms": times
        }
    
    def _generate_random_indices(self, total_rows: int, count: int, mode: str = "random") -> List[int]:
        """生成不同模式的随机索引"""
        if mode == "random":
            # 纯随机
            return np.random.randint(0, total_rows, count).tolist()
        elif mode == "clustered":
            # 聚簇模式：多个连续块
            num_clusters = max(1, count // 10)
            cluster_size = count // num_clusters
            indices = []
            for _ in range(num_clusters):
                start = np.random.randint(0, max(1, total_rows - cluster_size))
                indices.extend(range(start, min(start + cluster_size, total_rows)))
            return indices[:count]
        elif mode == "sparse":
            # 稀疏模式：均匀分布
            step = total_rows // count
            return [i * step for i in range(count)]
        elif mode == "hot":
            # 热点模式：大部分访问集中在少数区域
            hot_region_size = total_rows // 10
            hot_start = np.random.randint(0, total_rows - hot_region_size)
            hot_indices = np.random.randint(hot_start, hot_start + hot_region_size, int(count * 0.8)).tolist()
            cold_indices = np.random.randint(0, total_rows, int(count * 0.2)).tolist()
            return hot_indices + cold_indices
        else:
            return np.random.randint(0, total_rows, count).tolist()
    
    def benchmark_random_access(
        self, 
        data_rows: int, 
        data_cols: int, 
        access_counts: List[int],
        access_modes: List[str] = ["random"],
        repeat: int = 5
    ) -> Dict[str, Any]:
        """执行随机访问基准测试"""
        
        print(f"\n{'='*60}")
        print(f"Random Access Benchmark: {data_rows:,} rows × {data_cols} cols")
        print(f"{'='*60}")
        
        # 创建测试数据
        print("Creating test data...")
        data = np.random.rand(data_rows, data_cols).astype(np.float32)
        data_size_mb = data.nbytes / (1024 * 1024)
        print(f"Data size: {data_size_mb:.2f} MB")
        
        # 保存到NumPack
        npk_path = self.temp_dir / f"bench_data_{data_rows}x{data_cols}"
        if npk_path.exists():
            shutil.rmtree(npk_path)
        
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({"data": data})
        
        results = {
            "data_shape": [data_rows, data_cols],
            "data_size_mb": data_size_mb,
            "tests": {}
        }
        
        # 打开NumPack进行读取测试
        npk = NumPack(npk_path)
        npk.open()
        
        # 预热：加载lazy array
        lazy_arr = npk.load("data", lazy=True)
        _ = lazy_arr[0]  # 触发mmap
        
        for mode in access_modes:
            print(f"\n  Access Mode: {mode}")
            results["tests"][mode] = {}
            
            for count in access_counts:
                if count > data_rows:
                    continue
                    
                # 生成索引
                indices = self._generate_random_indices(data_rows, count, mode)
                
                # 测试方法1: 通过getitem访问
                def access_via_getitem():
                    _ = npk.getitem("data", indices)
                
                timing = self._time_operation(access_via_getitem, warmup=2, repeat=repeat)
                
                # 计算吞吐量
                rows_per_sec = count / (timing["min_ms"] / 1000)
                
                results["tests"][mode][f"count_{count}"] = {
                    "index_count": count,
                    "timing": timing,
                    "rows_per_second": rows_per_sec,
                    "us_per_row": timing["min_ms"] * 1000 / count
                }
                
                print(f"    {count:>6} indices: {timing['min_ms']:>8.3f}ms (min), "
                      f"{timing['mean_ms']:>8.3f}ms (mean), "
                      f"{rows_per_sec:>10,.0f} rows/s")
        
        npk.close()
        
        # 清理
        if npk_path.exists():
            shutil.rmtree(npk_path)
        
        return results
    
    def benchmark_sorted_vs_unsorted(self, data_rows: int, data_cols: int, index_count: int) -> Dict[str, Any]:
        """对比排序索引 vs 未排序索引的性能"""
        
        print(f"\n{'='*60}")
        print(f"Sorted vs Unsorted Index Comparison")
        print(f"Data: {data_rows:,} rows × {data_cols} cols, {index_count:,} indices")
        print(f"{'='*60}")
        
        # 创建测试数据
        data = np.random.rand(data_rows, data_cols).astype(np.float32)
        
        npk_path = self.temp_dir / "bench_sorted_test"
        if npk_path.exists():
            shutil.rmtree(npk_path)
        
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({"data": data})
        
        # 生成随机索引
        indices = np.random.randint(0, data_rows, index_count).tolist()
        sorted_indices = sorted(indices)
        
        npk = NumPack(npk_path)
        npk.open()
        
        # 预热
        lazy_arr = npk.load("data", lazy=True)
        _ = lazy_arr[0]
        
        # 测试未排序访问
        def access_unsorted():
            _ = npk.getitem("data", indices)
        
        unsorted_timing = self._time_operation(access_unsorted, warmup=2, repeat=5)
        
        # 测试排序后访问
        def access_sorted():
            _ = npk.getitem("data", sorted_indices)
        
        sorted_timing = self._time_operation(access_sorted, warmup=2, repeat=5)
        
        npk.close()
        
        # 计算改进
        improvement = (unsorted_timing["min_ms"] - sorted_timing["min_ms"]) / unsorted_timing["min_ms"] * 100
        
        results = {
            "data_shape": [data_rows, data_cols],
            "index_count": index_count,
            "unsorted": unsorted_timing,
            "sorted": sorted_timing,
            "improvement_percent": improvement
        }
        
        print(f"\n  Unsorted: {unsorted_timing['min_ms']:.3f}ms (min), {unsorted_timing['mean_ms']:.3f}ms (mean)")
        print(f"  Sorted:   {sorted_timing['min_ms']:.3f}ms (min), {sorted_timing['mean_ms']:.3f}ms (mean)")
        print(f"  Improvement: {improvement:.1f}%")
        
        if npk_path.exists():
            shutil.rmtree(npk_path)
        
        return results
    
    def benchmark_batch_sizes(self, data_rows: int, data_cols: int) -> Dict[str, Any]:
        """测试不同批次大小对性能的影响"""
        
        print(f"\n{'='*60}")
        print(f"Batch Size Impact Analysis")
        print(f"Data: {data_rows:,} rows × {data_cols} cols")
        print(f"{'='*60}")
        
        # 创建测试数据
        data = np.random.rand(data_rows, data_cols).astype(np.float32)
        
        npk_path = self.temp_dir / "bench_batch_test"
        if npk_path.exists():
            shutil.rmtree(npk_path)
        
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({"data": data})
        
        # 测试不同的索引数量
        batch_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
        if data_rows >= 100000:
            batch_sizes.extend([50000, 100000])
        
        results = {
            "data_shape": [data_rows, data_cols],
            "batch_tests": {}
        }
        
        npk = NumPack(npk_path)
        npk.open()
        
        # 预热
        lazy_arr = npk.load("data", lazy=True)
        _ = lazy_arr[0]
        
        print(f"\n  {'Batch Size':>12} | {'Min (ms)':>10} | {'Mean (ms)':>10} | {'μs/row':>10} | {'rows/s':>12}")
        print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
        
        for batch_size in batch_sizes:
            if batch_size > data_rows:
                continue
            
            indices = np.random.randint(0, data_rows, batch_size).tolist()
            
            def access_batch():
                _ = npk.getitem("data", indices)
            
            timing = self._time_operation(access_batch, warmup=2, repeat=5)
            
            us_per_row = timing["min_ms"] * 1000 / batch_size
            rows_per_sec = batch_size / (timing["min_ms"] / 1000)
            
            results["batch_tests"][batch_size] = {
                "timing": timing,
                "us_per_row": us_per_row,
                "rows_per_second": rows_per_sec
            }
            
            print(f"  {batch_size:>12,} | {timing['min_ms']:>10.3f} | {timing['mean_ms']:>10.3f} | {us_per_row:>10.2f} | {rows_per_sec:>12,.0f}")
        
        npk.close()
        
        if npk_path.exists():
            shutil.rmtree(npk_path)
        
        return results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """运行完整的基准测试套件"""
        
        print("\n" + "="*70)
        print(" NumPack Random Access Optimization Benchmark")
        print(" " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("="*70)
        
        # 测试配置
        test_configs = [
            # (rows, cols, description)
            (100_000, 10, "Medium dataset (100K × 10)"),
            (1_000_000, 10, "Large dataset (1M × 10)"),
        ]
        
        access_counts = [100, 1000, 10000]
        access_modes = ["random", "clustered", "sparse", "hot"]
        
        all_results = {}
        
        # 1. 基础随机访问测试
        print("\n" + "="*70)
        print(" PART 1: Basic Random Access Performance")
        print("="*70)
        
        for rows, cols, desc in test_configs:
            key = f"random_access_{rows}x{cols}"
            print(f"\n>>> {desc}")
            all_results[key] = self.benchmark_random_access(
                rows, cols, access_counts, access_modes
            )
        
        # 2. 排序 vs 未排序对比
        print("\n" + "="*70)
        print(" PART 2: Sorted vs Unsorted Index Access")
        print("="*70)
        
        all_results["sorted_comparison"] = {}
        for rows, cols, desc in test_configs:
            key = f"{rows}x{cols}"
            print(f"\n>>> {desc}")
            all_results["sorted_comparison"][key] = self.benchmark_sorted_vs_unsorted(
                rows, cols, 10000
            )
        
        # 3. 批次大小影响
        print("\n" + "="*70)
        print(" PART 3: Batch Size Impact")
        print("="*70)
        
        all_results["batch_size_impact"] = {}
        for rows, cols, desc in test_configs:
            key = f"{rows}x{cols}"
            print(f"\n>>> {desc}")
            all_results["batch_size_impact"][key] = self.benchmark_batch_sizes(rows, cols)
        
        self.results["benchmarks"] = all_results
        return self.results
    
    def save_results(self, filename: str = None):
        """保存结果到JSON文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"random_access_baseline_{timestamp}.json"
        
        output_path = self.results_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """打印测试摘要"""
        print("\n" + "="*70)
        print(" BENCHMARK SUMMARY")
        print("="*70)
        
        if "benchmarks" not in self.results:
            print("No benchmark results available.")
            return
        
        benchmarks = self.results["benchmarks"]
        
        # 随机访问摘要
        print("\n## Random Access Performance (min time in ms)")
        print(f"{'Dataset':<20} | {'100 idx':>10} | {'1K idx':>10} | {'10K idx':>10}")
        print("-" * 60)
        
        for key, data in benchmarks.items():
            if key.startswith("random_access_"):
                shape = f"{data['data_shape'][0]//1000}K×{data['data_shape'][1]}"
                if "random" in data.get("tests", {}):
                    tests = data["tests"]["random"]
                    t100 = tests.get("count_100", {}).get("timing", {}).get("min_ms", "N/A")
                    t1k = tests.get("count_1000", {}).get("timing", {}).get("min_ms", "N/A")
                    t10k = tests.get("count_10000", {}).get("timing", {}).get("min_ms", "N/A")
                    
                    t100_str = f"{t100:.3f}" if isinstance(t100, float) else t100
                    t1k_str = f"{t1k:.3f}" if isinstance(t1k, float) else t1k
                    t10k_str = f"{t10k:.3f}" if isinstance(t10k, float) else t10k
                    
                    print(f"{shape:<20} | {t100_str:>10} | {t1k_str:>10} | {t10k_str:>10}")
        
        # 排序对比摘要
        if "sorted_comparison" in benchmarks:
            print("\n## Sorted vs Unsorted Index Access")
            print(f"{'Dataset':<20} | {'Unsorted':>12} | {'Sorted':>12} | {'Improvement':>12}")
            print("-" * 65)
            
            for key, data in benchmarks["sorted_comparison"].items():
                unsorted = data.get("unsorted", {}).get("min_ms", "N/A")
                sorted_t = data.get("sorted", {}).get("min_ms", "N/A")
                improvement = data.get("improvement_percent", 0)
                
                unsorted_str = f"{unsorted:.3f}ms" if isinstance(unsorted, float) else unsorted
                sorted_str = f"{sorted_t:.3f}ms" if isinstance(sorted_t, float) else sorted_t
                
                print(f"{key:<20} | {unsorted_str:>12} | {sorted_str:>12} | {improvement:>10.1f}%")


def main():
    """主函数"""
    # 创建临时目录和结果目录
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        benchmark = RandomAccessBenchmark(temp_path, results_dir)
        
        # 运行完整基准测试
        benchmark.run_full_benchmark()
        
        # 打印摘要
        benchmark.print_summary()
        
        # 保存结果
        result_file = benchmark.save_results()
        
        print(f"\n✓ Baseline benchmark complete!")
        print(f"  Results file: {result_file}")


if __name__ == "__main__":
    main()
