#!/usr/bin/env python3
"""
NumPack VectorEngine vs NumPy Baseline Benchmark
================================================

测试当前 NumPack VectorEngine 各个算子与 NumPy 实现的性能对比。
用于优化前建立 baseline，确保后续优化不会导致性能回归。

测试算子:
- dot_product (点积)
- cosine_similarity (余弦相似度)
- l2_distance (欧氏距离)
- l2_squared (欧氏距离平方)
- batch_compute (批量计算)
- top_k_search (Top-K 搜索)
"""

import time
import numpy as np
from typing import Callable, Dict, List, Tuple
import json
from datetime import datetime


def timeit(func: Callable, warmup: int = 3, repeat: int = 10) -> Tuple[float, float]:
    """执行函数并返回平均时间和标准差（毫秒）"""
    # Warmup
    for _ in range(warmup):
        func()
    
    # Timed runs
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # 转换为毫秒
    
    return np.mean(times), np.std(times)


def numpy_dot(a: np.ndarray, b: np.ndarray) -> float:
    """NumPy 点积"""
    return np.dot(a, b)


def numpy_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """NumPy 余弦相似度"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def numpy_l2(a: np.ndarray, b: np.ndarray) -> float:
    """NumPy 欧氏距离"""
    return np.linalg.norm(a - b)


def numpy_l2sq(a: np.ndarray, b: np.ndarray) -> float:
    """NumPy 欧氏距离平方"""
    diff = a - b
    return np.dot(diff, diff)


def numpy_batch_dot(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """NumPy 批量点积 (使用矩阵乘法优化)"""
    return candidates @ query


def numpy_batch_cosine(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """NumPy 批量余弦相似度"""
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(len(candidates))
    candidates_norm = np.linalg.norm(candidates, axis=1)
    candidates_norm = np.where(candidates_norm == 0, 1, candidates_norm)  # 避免除零
    return (candidates @ query) / (query_norm * candidates_norm)


def numpy_batch_l2(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """NumPy 批量欧氏距离"""
    return np.linalg.norm(candidates - query, axis=1)


def numpy_batch_l2sq(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """NumPy 批量欧氏距离平方"""
    diff = candidates - query
    return np.sum(diff * diff, axis=1)


def numpy_top_k(query: np.ndarray, candidates: np.ndarray, k: int, metric: str = 'cosine') -> Tuple[np.ndarray, np.ndarray]:
    """NumPy Top-K 搜索"""
    if metric == 'cosine':
        scores = numpy_batch_cosine(query, candidates)
        indices = np.argsort(scores)[-k:][::-1]  # 相似度越大越好
    elif metric == 'dot':
        scores = numpy_batch_dot(query, candidates)
        indices = np.argsort(scores)[-k:][::-1]
    else:  # l2
        scores = numpy_batch_l2(query, candidates)
        indices = np.argsort(scores)[:k]  # 距离越小越好
    return indices, scores[indices]


def run_single_vector_benchmark(dim: int) -> Dict:
    """单向量操作基准测试"""
    print(f"\n{'='*60}")
    print(f"单向量操作基准测试 (维度: {dim})")
    print('='*60)
    
    # 准备数据
    np.random.seed(42)
    a = np.random.randn(dim).astype(np.float32)
    b = np.random.randn(dim).astype(np.float32)
    
    results = {}
    
    try:
        from numpack.vector_engine import VectorEngine
        engine = VectorEngine()
        print(f"SIMD 能力: {engine.capabilities()}")
        
        metrics = ['dot', 'cosine', 'l2', 'l2sq']
        numpy_funcs = {
            'dot': lambda: numpy_dot(a, b),
            'cosine': lambda: numpy_cosine(a, b),
            'l2': lambda: numpy_l2(a, b),
            'l2sq': lambda: numpy_l2sq(a, b),
        }
        
        for metric in metrics:
            print(f"\n--- {metric} ---")
            
            # NumPy
            np_time, np_std = timeit(numpy_funcs[metric])
            np_result = numpy_funcs[metric]()
            print(f"  NumPy:      {np_time:.4f} ± {np_std:.4f} ms  (result: {np_result:.6f})")
            
            # NumPack
            np_func = lambda m=metric: engine.compute_metric(a, b, m)
            npk_time, npk_std = timeit(np_func)
            npk_result = engine.compute_metric(a, b, metric)
            print(f"  NumPack:    {npk_time:.4f} ± {npk_std:.4f} ms  (result: {npk_result:.6f})")
            
            # 比较
            speedup = np_time / npk_time if npk_time > 0 else float('inf')
            print(f"  Speedup:    {speedup:.2f}x {'✓' if speedup >= 1 else '✗'}")
            
            results[f'single_{metric}'] = {
                'numpy_ms': np_time,
                'numpy_std': np_std,
                'numpack_ms': npk_time,
                'numpack_std': npk_std,
                'speedup': speedup,
                'dim': dim,
            }
            
    except ImportError as e:
        print(f"导入错误: {e}")
        results['error'] = str(e)
    
    return results


def run_batch_benchmark(n_candidates: int, dim: int) -> Dict:
    """批量计算基准测试"""
    print(f"\n{'='*60}")
    print(f"批量计算基准测试 (candidates: {n_candidates}, 维度: {dim})")
    print('='*60)
    
    # 准备数据
    np.random.seed(42)
    query = np.random.randn(dim).astype(np.float32)
    candidates = np.random.randn(n_candidates, dim).astype(np.float32)
    
    results = {}
    
    try:
        from numpack.vector_engine import VectorEngine
        engine = VectorEngine()
        
        metrics = ['dot', 'cosine', 'l2', 'l2sq']
        numpy_funcs = {
            'dot': lambda: numpy_batch_dot(query, candidates),
            'cosine': lambda: numpy_batch_cosine(query, candidates),
            'l2': lambda: numpy_batch_l2(query, candidates),
            'l2sq': lambda: numpy_batch_l2sq(query, candidates),
        }
        
        for metric in metrics:
            print(f"\n--- batch_{metric} ---")
            
            # NumPy
            np_time, np_std = timeit(numpy_funcs[metric])
            print(f"  NumPy:      {np_time:.4f} ± {np_std:.4f} ms")
            
            # NumPack
            np_func = lambda m=metric: engine.batch_compute(query, candidates, m)
            npk_time, npk_std = timeit(np_func)
            print(f"  NumPack:    {npk_time:.4f} ± {npk_std:.4f} ms")
            
            # 比较
            speedup = np_time / npk_time if npk_time > 0 else float('inf')
            print(f"  Speedup:    {speedup:.2f}x {'✓' if speedup >= 1 else '✗'}")
            
            # 吞吐量
            throughput_np = n_candidates / np_time * 1000  # ops/sec
            throughput_npk = n_candidates / npk_time * 1000
            print(f"  Throughput: NumPy {throughput_np:.0f} ops/s, NumPack {throughput_npk:.0f} ops/s")
            
            results[f'batch_{metric}'] = {
                'numpy_ms': np_time,
                'numpy_std': np_std,
                'numpack_ms': npk_time,
                'numpack_std': npk_std,
                'speedup': speedup,
                'n_candidates': n_candidates,
                'dim': dim,
                'throughput_numpy': throughput_np,
                'throughput_numpack': throughput_npk,
            }
            
    except ImportError as e:
        print(f"导入错误: {e}")
        results['error'] = str(e)
    
    return results


def run_top_k_benchmark(n_candidates: int, dim: int, k: int) -> Dict:
    """Top-K 搜索基准测试"""
    print(f"\n{'='*60}")
    print(f"Top-K 搜索基准测试 (candidates: {n_candidates}, 维度: {dim}, k: {k})")
    print('='*60)
    
    # 准备数据
    np.random.seed(42)
    query = np.random.randn(dim).astype(np.float32)
    candidates = np.random.randn(n_candidates, dim).astype(np.float32)
    
    results = {}
    
    try:
        from numpack.vector_engine import VectorEngine
        engine = VectorEngine()
        
        for metric in ['cosine', 'dot', 'l2']:
            print(f"\n--- top_k_{metric} ---")
            
            # NumPy
            np_func = lambda m=metric: numpy_top_k(query, candidates, k, m)
            np_time, np_std = timeit(np_func)
            print(f"  NumPy:      {np_time:.4f} ± {np_std:.4f} ms")
            
            # NumPack
            npk_func = lambda m=metric: engine.top_k_search(query, candidates, m, k)
            npk_time, npk_std = timeit(npk_func)
            print(f"  NumPack:    {npk_time:.4f} ± {npk_std:.4f} ms")
            
            # 比较
            speedup = np_time / npk_time if npk_time > 0 else float('inf')
            print(f"  Speedup:    {speedup:.2f}x {'✓' if speedup >= 1 else '✗'}")
            
            results[f'top_k_{metric}'] = {
                'numpy_ms': np_time,
                'numpy_std': np_std,
                'numpack_ms': npk_time,
                'numpack_std': npk_std,
                'speedup': speedup,
                'n_candidates': n_candidates,
                'dim': dim,
                'k': k,
            }
            
    except ImportError as e:
        print(f"导入错误: {e}")
        results['error'] = str(e)
    
    return results


def run_multi_query_benchmark(n_queries: int, n_candidates: int, dim: int) -> Dict:
    """多查询批量计算基准测试"""
    print(f"\n{'='*60}")
    print(f"多查询批量计算基准测试 (queries: {n_queries}, candidates: {n_candidates}, 维度: {dim})")
    print('='*60)
    
    # 准备数据
    np.random.seed(42)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    candidates = np.random.randn(n_candidates, dim).astype(np.float32)
    
    results = {}
    
    try:
        from numpack.vector_engine import VectorEngine
        engine = VectorEngine()
        
        for metric in ['cosine', 'dot']:
            print(f"\n--- multi_query_{metric} ---")
            
            # NumPy: 使用矩阵乘法
            if metric == 'dot':
                np_func = lambda: queries @ candidates.T
            else:  # cosine
                def numpy_multi_cosine():
                    q_norm = np.linalg.norm(queries, axis=1, keepdims=True)
                    c_norm = np.linalg.norm(candidates, axis=1, keepdims=True)
                    q_norm = np.where(q_norm == 0, 1, q_norm)
                    c_norm = np.where(c_norm == 0, 1, c_norm)
                    return (queries @ candidates.T) / (q_norm * c_norm.T)
                np_func = numpy_multi_cosine
            
            np_time, np_std = timeit(np_func)
            print(f"  NumPy:      {np_time:.4f} ± {np_std:.4f} ms")
            
            # NumPack: 使用 batch_compute with 2D query
            npk_func = lambda m=metric: engine.batch_compute(queries, candidates, m)
            npk_time, npk_std = timeit(npk_func)
            print(f"  NumPack:    {npk_time:.4f} ± {npk_std:.4f} ms")
            
            # 比较
            speedup = np_time / npk_time if npk_time > 0 else float('inf')
            print(f"  Speedup:    {speedup:.2f}x {'✓' if speedup >= 1 else '✗'}")
            
            # 总操作数
            total_ops = n_queries * n_candidates
            throughput_np = total_ops / np_time * 1000
            throughput_npk = total_ops / npk_time * 1000
            print(f"  Throughput: NumPy {throughput_np:.0f} ops/s, NumPack {throughput_npk:.0f} ops/s")
            
            results[f'multi_query_{metric}'] = {
                'numpy_ms': np_time,
                'numpy_std': np_std,
                'numpack_ms': npk_time,
                'numpack_std': npk_std,
                'speedup': speedup,
                'n_queries': n_queries,
                'n_candidates': n_candidates,
                'dim': dim,
            }
            
    except ImportError as e:
        print(f"导入错误: {e}")
        results['error'] = str(e)
    
    return results


def run_dtype_benchmark(dim: int, n_candidates: int) -> Dict:
    """不同数据类型基准测试"""
    print(f"\n{'='*60}")
    print(f"不同数据类型基准测试 (维度: {dim}, candidates: {n_candidates})")
    print('='*60)
    
    results = {}
    
    try:
        from numpack.vector_engine import VectorEngine
        engine = VectorEngine()
        
        dtypes = [np.float32, np.float64]
        
        for dtype in dtypes:
            dtype_name = dtype.__name__
            print(f"\n--- {dtype_name} ---")
            
            np.random.seed(42)
            query = np.random.randn(dim).astype(dtype)
            candidates = np.random.randn(n_candidates, dim).astype(dtype)
            
            metric = 'cosine'
            
            # NumPy
            np_func = lambda: numpy_batch_cosine(query, candidates)
            np_time, np_std = timeit(np_func)
            print(f"  NumPy:      {np_time:.4f} ± {np_std:.4f} ms")
            
            # NumPack
            npk_func = lambda: engine.batch_compute(query, candidates, metric)
            npk_time, npk_std = timeit(npk_func)
            print(f"  NumPack:    {npk_time:.4f} ± {npk_std:.4f} ms")
            
            # 比较
            speedup = np_time / npk_time if npk_time > 0 else float('inf')
            print(f"  Speedup:    {speedup:.2f}x {'✓' if speedup >= 1 else '✗'}")
            
            results[f'dtype_{dtype_name}'] = {
                'numpy_ms': np_time,
                'numpy_std': np_std,
                'numpack_ms': npk_time,
                'numpack_std': npk_std,
                'speedup': speedup,
            }
            
    except ImportError as e:
        print(f"导入错误: {e}")
        results['error'] = str(e)
    
    return results


def print_summary(all_results: Dict):
    """打印汇总报告"""
    print("\n" + "="*80)
    print(" BASELINE 性能测试汇总")
    print("="*80)
    
    print("\n{:<40} {:>12} {:>12} {:>10}".format(
        "测试项", "NumPy (ms)", "NumPack (ms)", "Speedup"
    ))
    print("-"*80)
    
    for category, results in all_results.items():
        if isinstance(results, dict) and 'error' not in results:
            for test_name, data in results.items():
                if isinstance(data, dict) and 'numpy_ms' in data:
                    status = "✓" if data['speedup'] >= 1 else "✗"
                    print("{:<40} {:>12.4f} {:>12.4f} {:>8.2f}x {}".format(
                        test_name,
                        data['numpy_ms'],
                        data['numpack_ms'],
                        data['speedup'],
                        status
                    ))


def main():
    print("="*80)
    print(" NumPack VectorEngine vs NumPy Baseline Benchmark")
    print(f" 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    all_results = {}
    
    # 1. 单向量操作测试 (不同维度)
    for dim in [128, 512, 1024]:
        all_results[f'single_dim{dim}'] = run_single_vector_benchmark(dim)
    
    # 2. 批量计算测试 (不同规模)
    for n_candidates, dim in [(10000, 128), (100000, 128), (10000, 512)]:
        all_results[f'batch_{n_candidates}x{dim}'] = run_batch_benchmark(n_candidates, dim)
    
    # 3. Top-K 搜索测试
    for n_candidates, dim, k in [(10000, 128, 10), (100000, 128, 100)]:
        all_results[f'topk_{n_candidates}x{dim}_k{k}'] = run_top_k_benchmark(n_candidates, dim, k)
    
    # 4. 多查询测试
    all_results['multi_query'] = run_multi_query_benchmark(
        n_queries=100, n_candidates=10000, dim=128
    )
    
    # 5. 数据类型测试
    all_results['dtype'] = run_dtype_benchmark(dim=128, n_candidates=10000)
    
    # 打印汇总
    print_summary(all_results)
    
    # 保存结果
    output_file = 'vector_engine_baseline_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n结果已保存到: {output_file}")
    
    return all_results


if __name__ == '__main__':
    main()
