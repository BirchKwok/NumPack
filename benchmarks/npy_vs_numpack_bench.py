#!/usr/bin/env python3
"""
NPY vs NumPack 随机索引性能对比
================================

对比方式：
- NPY: np.load(filename, mmap_mode="r")[idx] 然后复制为 np.ndarray
- NumPack: getitem 然后确保返回 np.ndarray
"""

import os
import sys
import time
import tempfile
import shutil
import gc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from numpack import NumPack


def benchmark_random_access(data_rows: int, data_cols: int, index_counts: list, repeat: int = 5):
    """对比NPY和NumPack的随机索引性能"""
    
    print(f"\n{'='*70}")
    print(f" NPY vs NumPack 随机索引对比")
    print(f" 数据集: {data_rows:,} rows × {data_cols} cols")
    print(f"{'='*70}")
    
    # 创建测试数据
    print("\n创建测试数据...")
    data = np.random.rand(data_rows, data_cols).astype(np.float32)
    data_size_mb = data.nbytes / (1024 * 1024)
    print(f"数据大小: {data_size_mb:.2f} MB")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 保存NPY文件
        npy_path = temp_path / "data.npy"
        np.save(npy_path, data)
        npy_size = npy_path.stat().st_size / (1024 * 1024)
        print(f"NPY文件大小: {npy_size:.2f} MB")
        
        # 保存NumPack文件
        npk_path = temp_path / "data_npk"
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({"data": data})
        npk_size = sum(f.stat().st_size for f in npk_path.rglob("*") if f.is_file()) / (1024 * 1024)
        print(f"NumPack文件大小: {npk_size:.2f} MB")
        
        # 预热mmap
        npy_mmap = np.load(npy_path, mmap_mode='r')
        _ = npy_mmap[0].copy()
        
        npk = NumPack(npk_path)
        npk.open()
        _ = npk.getitem("data", [0])
        
        print(f"\n{'Index Count':>12} | {'NPY (ms)':>12} | {'NumPack (ms)':>12} | {'Winner':>10} | {'Speedup':>10}")
        print(f"{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")
        
        results = []
        
        for count in index_counts:
            if count > data_rows:
                continue
            
            # 生成随机索引
            indices = np.random.randint(0, data_rows, count)
            indices_list = indices.tolist()
            
            # 测试NPY mmap随机访问
            npy_times = []
            for _ in range(repeat):
                gc.collect()
                start = time.perf_counter()
                result_npy = np.array(npy_mmap[indices])  # 复制为ndarray
                end = time.perf_counter()
                npy_times.append((end - start) * 1000)
            npy_min = min(npy_times)
            
            # 测试NumPack随机访问
            npk_times = []
            for _ in range(repeat):
                gc.collect()
                start = time.perf_counter()
                result_npk = np.array(npk.getitem("data", indices_list))  # 确保是ndarray
                end = time.perf_counter()
                npk_times.append((end - start) * 1000)
            npk_min = min(npk_times)
            
            # 验证结果一致性
            assert result_npy.shape == result_npk.shape, f"Shape mismatch: {result_npy.shape} vs {result_npk.shape}"
            
            # 判断winner
            if npk_min < npy_min:
                winner = "NumPack"
                speedup = f"{npy_min / npk_min:.2f}x"
            else:
                winner = "NPY"
                speedup = f"{npk_min / npy_min:.2f}x"
            
            results.append({
                'count': count,
                'npy_ms': npy_min,
                'npk_ms': npk_min,
                'winner': winner,
                'speedup': speedup
            })
            
            print(f"{count:>12,} | {npy_min:>12.3f} | {npk_min:>12.3f} | {winner:>10} | {speedup:>10}")
        
        npk.close()
        
        # 打印总结
        print(f"\n{'='*70}")
        print(" 总结")
        print(f"{'='*70}")
        
        npk_wins = sum(1 for r in results if r['winner'] == 'NumPack')
        npy_wins = len(results) - npk_wins
        
        print(f"NumPack获胜: {npk_wins}/{len(results)}")
        print(f"NPY获胜: {npy_wins}/{len(results)}")
        
        # 计算平均speedup
        avg_speedup = sum(r['npy_ms'] / r['npk_ms'] for r in results) / len(results)
        print(f"\n平均 NumPack vs NPY: {avg_speedup:.2f}x")
        
        if avg_speedup > 1:
            print(f"NumPack 平均快 {(avg_speedup - 1) * 100:.1f}%")
        else:
            print(f"NPY 平均快 {(1 / avg_speedup - 1) * 100:.1f}%")
        
        return results


def main():
    """主函数"""
    print("\n" + "="*70)
    print(" NPY vs NumPack 随机索引性能对比测试")
    print("="*70)
    
    # 测试配置
    test_configs = [
        (100_000, 10, "Medium dataset"),
        (1_000_000, 10, "Large dataset"),
    ]
    
    index_counts = [10, 100, 1000, 10000]
    
    all_results = {}
    
    for rows, cols, desc in test_configs:
        print(f"\n>>> {desc}")
        results = benchmark_random_access(rows, cols, index_counts)
        all_results[f"{rows}x{cols}"] = results
    
    print("\n" + "="*70)
    print(" 测试完成!")
    print("="*70)


if __name__ == "__main__":
    main()
