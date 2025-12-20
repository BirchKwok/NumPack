#!/usr/bin/env python3
"""
VectorEngine 数据传输性能基准测试

测试目标：
1. 确认数据传输 vs 计算的时间占比
2. 比较常规加载 vs LazyArray 的性能差异
3. 分析不同数据规模下的瓶颈

测试场景：
- 场景A: query和candidates都是普通numpy数组
- 场景B: query是numpy数组，candidates是LazyArray
- 场景C: query和candidates都是LazyArray
- 场景D: 从同一个NumPack文件加载的数组
"""

import numpy as np
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import gc

# 导入numpack
import numpack
from numpack import NumPack
from numpack.vector_engine import VectorEngine


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def format_throughput(ops: int, seconds: float) -> str:
    """格式化吞吐量"""
    if seconds == 0:
        return "inf ops/s"
    ops_per_sec = ops / seconds
    if ops_per_sec > 1e6:
        return f"{ops_per_sec / 1e6:.2f} M ops/s"
    elif ops_per_sec > 1e3:
        return f"{ops_per_sec / 1e3:.2f} K ops/s"
    else:
        return f"{ops_per_sec:.2f} ops/s"


class BenchmarkContext:
    """基准测试上下文管理器"""
    
    def __init__(self, name: str):
        self.name = name
        self.temp_dir = None
        self.npk_path = None
        
    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="numpack_bench_")
        self.npk_path = Path(self.temp_dir) / "bench_data.npk"
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
        return False


def measure_time(func, warmup: int = 2, iterations: int = 10) -> Tuple[float, float, float]:
    """测量函数执行时间，返回(平均时间, 最小时间, 最大时间)"""
    # 预热
    for _ in range(warmup):
        func()
    
    # GC清理
    gc.collect()
    
    # 正式测量
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times), np.min(times), np.max(times)


def benchmark_pure_computation(engine: VectorEngine, dim: int, n_candidates: int) -> Dict:
    """基准测试1: 纯计算性能（数据已在内存中）"""
    print(f"\n[基准测试1] 纯计算性能 (dim={dim}, n_candidates={n_candidates})")
    print("-" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    query = np.random.randn(dim).astype(np.float32)
    candidates = np.random.randn(n_candidates, dim).astype(np.float32)
    
    # 确保数据连续
    query = np.ascontiguousarray(query)
    candidates = np.ascontiguousarray(candidates)
    
    results = {}
    
    # 测试batch_compute
    def run_batch():
        return engine.batch_compute(query, candidates, "cosine")
    
    avg, min_t, max_t = measure_time(run_batch)
    results['batch_compute'] = {
        'avg': avg, 'min': min_t, 'max': max_t,
        'throughput': n_candidates / avg
    }
    print(f"  batch_compute: {format_time(avg)} (min={format_time(min_t)}, max={format_time(max_t)})")
    print(f"    吞吐量: {format_throughput(n_candidates, avg)}")
    
    # 测试top_k_search
    def run_topk():
        return engine.top_k_search(query, candidates, "cosine", k=10)
    
    avg, min_t, max_t = measure_time(run_topk)
    results['top_k_search'] = {
        'avg': avg, 'min': min_t, 'max': max_t,
        'throughput': n_candidates / avg
    }
    print(f"  top_k_search:  {format_time(avg)} (min={format_time(min_t)}, max={format_time(max_t)})")
    
    return results


def benchmark_data_loading(ctx: BenchmarkContext, dim: int, n_candidates: int) -> Dict:
    """基准测试2: 数据加载性能"""
    print(f"\n[基准测试2] 数据加载性能 (dim={dim}, n_candidates={n_candidates})")
    print("-" * 60)
    
    # 创建测试数据并保存
    np.random.seed(42)
    query_data = np.random.randn(1, dim).astype(np.float32)
    candidates_data = np.random.randn(n_candidates, dim).astype(np.float32)
    
    with NumPack(ctx.npk_path, drop_if_exists=True) as npk:
        npk.save({'query': query_data, 'candidates': candidates_data})
    
    results = {}
    
    # 测试常规加载
    with NumPack(ctx.npk_path) as npk:
        def load_regular():
            q = npk.load('query', lazy=False)
            c = npk.load('candidates', lazy=False)
            return q, c
        
        avg, min_t, max_t = measure_time(load_regular)
        results['regular_load'] = {'avg': avg, 'min': min_t, 'max': max_t}
        print(f"  常规加载: {format_time(avg)} (min={format_time(min_t)}, max={format_time(max_t)})")
        
        # 测试懒加载
        def load_lazy():
            q = npk.load('query', lazy=True)
            c = npk.load('candidates', lazy=True)
            return q, c
        
        avg, min_t, max_t = measure_time(load_lazy)
        results['lazy_load'] = {'avg': avg, 'min': min_t, 'max': max_t}
        print(f"  懒加载:   {format_time(avg)} (min={format_time(min_t)}, max={format_time(max_t)})")
    
    return results


def benchmark_end_to_end_numpy(engine: VectorEngine, ctx: BenchmarkContext, 
                               dim: int, n_candidates: int) -> Dict:
    """基准测试3: 端到端性能 - NumPy数组"""
    print(f"\n[基准测试3] 端到端性能 - NumPy数组 (dim={dim}, n_candidates={n_candidates})")
    print("-" * 60)
    
    # 准备数据
    np.random.seed(42)
    query_data = np.random.randn(1, dim).astype(np.float32)
    candidates_data = np.random.randn(n_candidates, dim).astype(np.float32)
    
    with NumPack(ctx.npk_path, drop_if_exists=True) as npk:
        npk.save({'query': query_data, 'candidates': candidates_data})
    
    results = {}
    
    with NumPack(ctx.npk_path) as npk:
        # 场景A: 预加载数据到内存，测量纯计算
        query = npk.load('query', lazy=False)[0]  # 取第一行作为1D向量
        candidates = npk.load('candidates', lazy=False)
        
        def compute_preloaded():
            return engine.batch_compute(query, candidates, "cosine")
        
        avg, min_t, max_t = measure_time(compute_preloaded)
        results['preloaded'] = {
            'avg': avg, 'min': min_t, 'max': max_t,
            'throughput': n_candidates / avg
        }
        print(f"  预加载后计算: {format_time(avg)} ({format_throughput(n_candidates, avg)})")
        
        # 场景B: 每次都重新加载（模拟真实场景）
        def load_and_compute():
            q = npk.load('query', lazy=False)[0]
            c = npk.load('candidates', lazy=False)
            return engine.batch_compute(q, c, "cosine")
        
        avg, min_t, max_t = measure_time(load_and_compute, iterations=5)
        results['load_each_time'] = {
            'avg': avg, 'min': min_t, 'max': max_t,
            'throughput': n_candidates / avg
        }
        print(f"  每次加载+计算: {format_time(avg)} ({format_throughput(n_candidates, avg)})")
    
    return results


def benchmark_end_to_end_lazy(engine: VectorEngine, ctx: BenchmarkContext,
                              dim: int, n_candidates: int) -> Dict:
    """基准测试4: 端到端性能 - LazyArray"""
    print(f"\n[基准测试4] 端到端性能 - LazyArray (dim={dim}, n_candidates={n_candidates})")
    print("-" * 60)
    
    # 准备数据
    np.random.seed(42)
    query_data = np.random.randn(1, dim).astype(np.float32)
    candidates_data = np.random.randn(n_candidates, dim).astype(np.float32)
    
    with NumPack(ctx.npk_path, drop_if_exists=True) as npk:
        npk.save({'query': query_data, 'candidates': candidates_data})
    
    results = {}
    
    with NumPack(ctx.npk_path) as npk:
        # 获取LazyArray
        query_lazy = npk.load('query', lazy=True)
        candidates_lazy = npk.load('candidates', lazy=True)
        
        # 方式1: LazyArray直接传递（需要转换为numpy）
        def compute_lazy_direct():
            q = np.asarray(query_lazy)[0]  # LazyArray -> numpy
            c = np.asarray(candidates_lazy)
            return engine.batch_compute(q, c, "cosine")
        
        avg, min_t, max_t = measure_time(compute_lazy_direct)
        results['lazy_direct'] = {
            'avg': avg, 'min': min_t, 'max': max_t,
            'throughput': n_candidates / avg
        }
        print(f"  LazyArray直接使用: {format_time(avg)} ({format_throughput(n_candidates, avg)})")
        
        # 方式2: 预转换为numpy后使用
        query_np = np.asarray(query_lazy)[0]
        candidates_np = np.asarray(candidates_lazy)
        
        def compute_lazy_preconvert():
            return engine.batch_compute(query_np, candidates_np, "cosine")
        
        avg, min_t, max_t = measure_time(compute_lazy_preconvert)
        results['lazy_preconvert'] = {
            'avg': avg, 'min': min_t, 'max': max_t,
            'throughput': n_candidates / avg
        }
        print(f"  预转换后计算:     {format_time(avg)} ({format_throughput(n_candidates, avg)})")
        
        # 方式3: 每次加载LazyArray
        def load_lazy_and_compute():
            q_lazy = npk.load('query', lazy=True)
            c_lazy = npk.load('candidates', lazy=True)
            q = np.asarray(q_lazy)[0]
            c = np.asarray(c_lazy)
            return engine.batch_compute(q, c, "cosine")
        
        avg, min_t, max_t = measure_time(load_lazy_and_compute, iterations=5)
        results['load_lazy_each_time'] = {
            'avg': avg, 'min': min_t, 'max': max_t,
            'throughput': n_candidates / avg
        }
        print(f"  每次懒加载+计算:  {format_time(avg)} ({format_throughput(n_candidates, avg)})")
    
    return results


def benchmark_breakdown(engine: VectorEngine, ctx: BenchmarkContext,
                        dim: int, n_candidates: int) -> Dict:
    """基准测试5: 时间分解分析"""
    print(f"\n[基准测试5] 时间分解分析 (dim={dim}, n_candidates={n_candidates})")
    print("-" * 60)
    
    # 准备数据
    np.random.seed(42)
    query_data = np.random.randn(1, dim).astype(np.float32)
    candidates_data = np.random.randn(n_candidates, dim).astype(np.float32)
    
    with NumPack(ctx.npk_path, drop_if_exists=True) as npk:
        npk.save({'query': query_data, 'candidates': candidates_data})
    
    results = {}
    iterations = 10
    
    with NumPack(ctx.npk_path) as npk:
        # 分步测量
        load_times = []
        convert_times = []
        compute_times = []
        
        for _ in range(iterations):
            # 1. 加载时间
            gc.collect()
            t0 = time.perf_counter()
            q_lazy = npk.load('query', lazy=True)
            c_lazy = npk.load('candidates', lazy=True)
            t1 = time.perf_counter()
            load_times.append(t1 - t0)
            
            # 2. 转换时间
            t2 = time.perf_counter()
            q = np.asarray(q_lazy)[0]
            c = np.asarray(c_lazy)
            t3 = time.perf_counter()
            convert_times.append(t3 - t2)
            
            # 3. 计算时间
            t4 = time.perf_counter()
            _ = engine.batch_compute(q, c, "cosine")
            t5 = time.perf_counter()
            compute_times.append(t5 - t4)
        
        avg_load = np.mean(load_times)
        avg_convert = np.mean(convert_times)
        avg_compute = np.mean(compute_times)
        total = avg_load + avg_convert + avg_compute
        
        results = {
            'load': avg_load,
            'convert': avg_convert,
            'compute': avg_compute,
            'total': total
        }
        
        print(f"  1. 懒加载时间:   {format_time(avg_load)} ({avg_load/total*100:.1f}%)")
        print(f"  2. 数据转换时间: {format_time(avg_convert)} ({avg_convert/total*100:.1f}%)")
        print(f"  3. 计算时间:     {format_time(avg_compute)} ({avg_compute/total*100:.1f}%)")
        print(f"  --------------")
        print(f"  总时间:          {format_time(total)}")
    
    return results


def benchmark_scale_analysis(engine: VectorEngine, ctx: BenchmarkContext) -> Dict:
    """基准测试6: 规模分析"""
    print(f"\n[基准测试6] 不同规模下的性能分析")
    print("=" * 60)
    
    configs = [
        (128, 1000),      # 小规模
        (128, 10000),     # 中规模
        (128, 100000),    # 大规模
        (512, 10000),     # 高维中规模
        (1024, 10000),    # 超高维中规模
    ]
    
    results = {}
    
    for dim, n_candidates in configs:
        print(f"\n>>> 配置: dim={dim}, n_candidates={n_candidates}")
        
        # 准备数据
        np.random.seed(42)
        query_data = np.random.randn(1, dim).astype(np.float32)
        candidates_data = np.random.randn(n_candidates, dim).astype(np.float32)
        
        with NumPack(ctx.npk_path, drop_if_exists=True) as npk:
            npk.save({'query': query_data, 'candidates': candidates_data})
        
        with NumPack(ctx.npk_path) as npk:
            # 预加载
            query = npk.load('query', lazy=False)[0]
            candidates = npk.load('candidates', lazy=False)
            
            def compute():
                return engine.batch_compute(query, candidates, "cosine")
            
            avg, min_t, max_t = measure_time(compute, iterations=5)
            
            config_key = f"dim{dim}_n{n_candidates}"
            results[config_key] = {
                'dim': dim,
                'n_candidates': n_candidates,
                'avg': avg,
                'throughput': n_candidates / avg,
                'ops_per_dim': n_candidates * dim / avg
            }
            
            print(f"    平均时间: {format_time(avg)}")
            print(f"    吞吐量:   {format_throughput(n_candidates, avg)}")
            print(f"    带宽:     {n_candidates * dim * 4 / avg / 1e9:.2f} GB/s (float32)")
    
    return results


def benchmark_dtype_comparison(engine: VectorEngine, dim: int, n_candidates: int) -> Dict:
    """基准测试7: 不同数据类型的性能比较"""
    print(f"\n[基准测试7] 数据类型性能比较 (dim={dim}, n_candidates={n_candidates})")
    print("-" * 60)
    
    np.random.seed(42)
    results = {}
    
    # Float64
    query_f64 = np.random.randn(dim).astype(np.float64)
    candidates_f64 = np.random.randn(n_candidates, dim).astype(np.float64)
    
    def compute_f64():
        return engine.batch_compute(query_f64, candidates_f64, "cosine")
    
    avg, _, _ = measure_time(compute_f64)
    results['float64'] = {'avg': avg, 'throughput': n_candidates / avg}
    print(f"  float64: {format_time(avg)} ({format_throughput(n_candidates, avg)})")
    
    # Float32
    query_f32 = np.random.randn(dim).astype(np.float32)
    candidates_f32 = np.random.randn(n_candidates, dim).astype(np.float32)
    
    def compute_f32():
        return engine.batch_compute(query_f32, candidates_f32, "cosine")
    
    avg, _, _ = measure_time(compute_f32)
    results['float32'] = {'avg': avg, 'throughput': n_candidates / avg}
    print(f"  float32: {format_time(avg)} ({format_throughput(n_candidates, avg)})")
    
    # Int8
    query_i8 = (np.random.randn(dim) * 100).astype(np.int8)
    candidates_i8 = (np.random.randn(n_candidates, dim) * 100).astype(np.int8)
    
    def compute_i8():
        return engine.batch_compute(query_i8, candidates_i8, "cosine")
    
    avg, _, _ = measure_time(compute_i8)
    results['int8'] = {'avg': avg, 'throughput': n_candidates / avg}
    print(f"  int8:    {format_time(avg)} ({format_throughput(n_candidates, avg)})")
    
    return results


def run_all_benchmarks():
    """运行所有基准测试"""
    print("=" * 70)
    print("VectorEngine 数据传输性能基准测试")
    print("=" * 70)
    
    # 显示环境信息
    engine = VectorEngine()
    print(f"\nSIMD能力: {engine.capabilities()}")
    print(f"NumPack版本: {numpack.__version__}")
    
    # 默认测试参数
    dim = 128
    n_candidates = 10000
    
    with BenchmarkContext("vector_engine_bench") as ctx:
        # 1. 纯计算性能
        benchmark_pure_computation(engine, dim, n_candidates)
        
        # 2. 数据加载性能
        benchmark_data_loading(ctx, dim, n_candidates)
        
        # 3. 端到端 - NumPy
        benchmark_end_to_end_numpy(engine, ctx, dim, n_candidates)
        
        # 4. 端到端 - LazyArray
        benchmark_end_to_end_lazy(engine, ctx, dim, n_candidates)
        
        # 5. 时间分解
        breakdown = benchmark_breakdown(engine, ctx, dim, n_candidates)
        
        # 6. 规模分析
        benchmark_scale_analysis(engine, ctx)
        
        # 7. 数据类型比较
        benchmark_dtype_comparison(engine, dim, n_candidates)
    
    # 总结
    print("\n" + "=" * 70)
    print("基准测试总结")
    print("=" * 70)
    print(f"""
分析结论:
1. 时间分解 (dim={dim}, n_candidates={n_candidates}):
   - 懒加载时间:   {breakdown['load']/breakdown['total']*100:.1f}%
   - 数据转换时间: {breakdown['convert']/breakdown['total']*100:.1f}%
   - 计算时间:     {breakdown['compute']/breakdown['total']*100:.1f}%

2. 优化建议:
   - 如果"懒加载时间"占比高: 考虑缓存机制或批量加载
   - 如果"数据转换时间"占比高: 可优化buffer protocol或直接访问mmap
   - 如果"计算时间"占比高: 当前实现已接近最优

3. 瓶颈判断:
   - 计算密集型: 计算时间 > 80%
   - IO密集型: 加载+转换时间 > 50%
   - 混合型: 各部分时间接近
""")


if __name__ == "__main__":
    run_all_benchmarks()
