#!/usr/bin/env python3
"""
StreamingVectorEngine 性能基准测试

测试目标：
1. 比较流式计算 vs 直接计算的性能
2. 测试不同batch_size的影响
3. 测试大规模数据集的内存效率
4. 测试流式Top-K搜索的性能
"""

import numpy as np
import time
import tempfile
import shutil
import gc
import psutil
from pathlib import Path
from typing import Dict

import numpack
from numpack import NumPack
from numpack.vector_engine import VectorEngine, StreamingVectorEngine


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


def format_memory(bytes_val: int) -> str:
    """格式化内存显示"""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.2f} KB"
    elif bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.2f} MB"
    else:
        return f"{bytes_val / (1024 * 1024 * 1024):.2f} GB"


def get_memory_usage() -> int:
    """获取当前进程内存使用"""
    process = psutil.Process()
    return process.memory_info().rss


def measure_time(func, warmup: int = 1, iterations: int = 5):
    """测量函数执行时间"""
    for _ in range(warmup):
        func()
    
    gc.collect()
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times), np.min(times), np.max(times), result


class StreamingBenchmark:
    """流式计算基准测试"""
    
    def __init__(self):
        self.engine = VectorEngine()
        self.temp_dir = None
        
    def setup(self):
        self.temp_dir = tempfile.mkdtemp(prefix="streaming_bench_")
        return self
    
    def cleanup(self):
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def __enter__(self):
        return self.setup()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
    
    def benchmark_batch_compute_comparison(self, dim: int, n_candidates: int):
        """比较流式 vs 直接批量计算"""
        print(f"\n[基准测试] 流式 vs 直接批量计算 (dim={dim}, n={n_candidates})")
        print("-" * 60)
        
        npk_path = Path(self.temp_dir) / "bench.npk"
        
        # 准备数据
        np.random.seed(42)
        query = np.random.randn(dim).astype(np.float32)
        candidates = np.random.randn(n_candidates, dim).astype(np.float32)
        
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({'candidates': candidates})
        
        results = {}
        
        with NumPack(npk_path) as npk:
            streaming_searcher = StreamingVectorEngine()
            
            # 直接计算（全部加载到内存）
            loaded_candidates = npk.load('candidates', lazy=False)
            
            def direct_compute():
                return self.engine.batch_compute(query, loaded_candidates, 'cosine')
            
            avg, min_t, max_t, _ = measure_time(direct_compute)
            results['direct'] = {'avg': avg, 'min': min_t, 'max': max_t}
            print(f"  直接计算:       {format_time(avg)} (min={format_time(min_t)})")
            
            # 流式计算（不同batch_size）
            batch_sizes = [1000, 5000, 10000, 50000]
            
            for bs in batch_sizes:
                if bs > n_candidates:
                    continue
                
                def stream_compute():
                    return streaming_searcher.streaming_batch_compute(
                        query, str(npk_path), 'candidates', 'cosine', batch_size=bs
                    )
                
                avg, min_t, max_t, _ = measure_time(stream_compute)
                results[f'stream_bs{bs}'] = {'avg': avg, 'min': min_t, 'max': max_t}
                overhead = (avg / results['direct']['avg'] - 1) * 100
                print(f"  流式 (bs={bs:5d}): {format_time(avg)} (开销: {overhead:+.1f}%)")
        
        return results
    
    def benchmark_top_k_comparison(self, dim: int, n_candidates: int, k: int = 10):
        """比较流式 vs 直接Top-K搜索"""
        print(f"\n[基准测试] 流式 vs 直接Top-K (dim={dim}, n={n_candidates}, k={k})")
        print("-" * 60)
        
        npk_path = Path(self.temp_dir) / "bench.npk"
        
        np.random.seed(42)
        query = np.random.randn(dim).astype(np.float32)
        candidates = np.random.randn(n_candidates, dim).astype(np.float32)
        
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({'candidates': candidates})
        
        results = {}
        
        with NumPack(npk_path) as npk:
            streaming_searcher = StreamingVectorEngine()
            loaded_candidates = npk.load('candidates', lazy=False)
            
            # 直接Top-K
            def direct_topk():
                return self.engine.top_k_search(query, loaded_candidates, 'cosine', k=k)
            
            avg, min_t, max_t, (direct_idx, direct_scores) = measure_time(direct_topk)
            results['direct'] = {'avg': avg}
            print(f"  直接Top-K:       {format_time(avg)}")
            
            # 流式Top-K
            batch_sizes = [1000, 5000, 10000]
            
            for bs in batch_sizes:
                if bs > n_candidates:
                    continue
                
                def stream_topk():
                    return streaming_searcher.streaming_top_k_from_file(
                        query, str(npk_path), 'candidates', 'cosine', k=k, batch_size=bs
                    )
                
                avg, min_t, max_t, (stream_idx, stream_scores) = measure_time(stream_topk)
                results[f'stream_bs{bs}'] = {'avg': avg}
                overhead = (avg / results['direct']['avg'] - 1) * 100
                
                # 验证结果一致性
                match = np.array_equal(stream_idx, direct_idx)
                status = "OK" if match else "MISMATCH"
                
                print(f"  流式 (bs={bs:5d}):  {format_time(avg)} (开销: {overhead:+.1f}%) [{status}]")
        
        return results
    
    def benchmark_memory_efficiency(self, dim: int, n_candidates: int):
        """测试内存效率"""
        print(f"\n[基准测试] 内存效率 (dim={dim}, n={n_candidates})")
        print("-" * 60)
        
        data_size = n_candidates * dim * 4  # float32
        print(f"  数据大小: {format_memory(data_size)}")
        
        npk_path = Path(self.temp_dir) / "bench.npk"
        
        np.random.seed(42)
        query = np.random.randn(dim).astype(np.float32)
        candidates = np.random.randn(n_candidates, dim).astype(np.float32)
        
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({'candidates': candidates})
        
        # 清理内存
        del candidates
        gc.collect()
        
        with NumPack(npk_path) as npk:
            streaming_searcher = StreamingVectorEngine()
            
            # 测试1：全部加载
            gc.collect()
            mem_before = get_memory_usage()
            
            loaded = npk.load('candidates', lazy=False)
            _ = self.engine.batch_compute(query, loaded, 'cosine')
            
            mem_after = get_memory_usage()
            mem_direct = mem_after - mem_before
            print(f"  直接加载内存增量: {format_memory(mem_direct)}")
            
            del loaded
            gc.collect()
            
            # 测试2：流式计算
            gc.collect()
            mem_before = get_memory_usage()
            
            batch_size = 10000
            
            _ = streaming_searcher.streaming_batch_compute(
                query, str(npk_path), 'candidates', 'cosine', batch_size=batch_size
            )
            
            mem_after = get_memory_usage()
            mem_stream = mem_after - mem_before
            print(f"  流式计算内存增量: {format_memory(mem_stream)} (batch_size={batch_size})")
            
            if mem_direct > 0:
                ratio = mem_stream / mem_direct
                print(f"  内存节省: {(1 - ratio) * 100:.1f}%")
    
    def benchmark_large_scale(self):
        """大规模数据测试"""
        print(f"\n[基准测试] 大规模数据性能")
        print("=" * 60)
        
        configs = [
            (128, 100000),    # 100K vectors
            (128, 500000),    # 500K vectors
            (256, 100000),    # 高维 100K
        ]
        
        for dim, n in configs:
            data_size_mb = n * dim * 4 / (1024 * 1024)
            print(f"\n>>> 配置: dim={dim}, n={n} (数据: {data_size_mb:.1f} MB)")
            
            npk_path = Path(self.temp_dir) / "large_bench.npk"
            
            np.random.seed(42)
            query = np.random.randn(dim).astype(np.float32)
            
            # 分批生成和保存数据以节省内存
            with NumPack(npk_path, drop_if_exists=True) as npk:
                chunk_size = 50000
                first = True
                for i in range(0, n, chunk_size):
                    size = min(chunk_size, n - i)
                    chunk = np.random.randn(size, dim).astype(np.float32)
                    if first:
                        npk.save({'candidates': chunk})
                        first = False
                    else:
                        npk.append({'candidates': chunk})
            
            with NumPack(npk_path) as npk:
                streaming_searcher = StreamingVectorEngine()
                
                # 流式Top-K
                start = time.perf_counter()
                indices, scores = streaming_searcher.streaming_top_k_from_file(
                    query, str(npk_path), 'candidates', 'cosine', k=10, batch_size=10000
                )
                elapsed = time.perf_counter() - start
                
                throughput = n / elapsed
                print(f"    流式Top-K: {format_time(elapsed)} ({throughput/1e6:.2f} M vecs/s)")
    
    def benchmark_multi_query(self, dim: int, n_candidates: int, n_queries: int):
        """多查询测试"""
        print(f"\n[基准测试] 多查询性能 (queries={n_queries}, candidates={n_candidates})")
        print("-" * 60)
        
        npk_path = Path(self.temp_dir) / "bench.npk"
        
        np.random.seed(42)
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        candidates = np.random.randn(n_candidates, dim).astype(np.float32)
        
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({'candidates': candidates})
        
        with NumPack(npk_path) as npk:
            streaming_searcher = StreamingVectorEngine()
            loaded = npk.load('candidates', lazy=False)
            
            # 直接计算（每个query单独计算）
            start = time.perf_counter()
            for q in queries:
                self.engine.top_k_search(q, loaded, 'cosine', k=10)
            direct_time = time.perf_counter() - start
            print(f"  直接（逐个）:    {format_time(direct_time)}")
            
            # 直接计算（批量多查询优化）
            start = time.perf_counter()
            direct_batch_indices, direct_batch_scores = self.engine.multi_query_top_k(
                queries, loaded, 'cosine', k=10
            )
            direct_batch_time = time.perf_counter() - start
            
            speedup_direct = direct_time / direct_batch_time
            print(f"  直接（批量）:    {format_time(direct_batch_time)} (vs逐个提升: {speedup_direct:.2f}x)")
            
            # 流式多查询（逐个查询）
            start = time.perf_counter()
            for q in queries:
                streaming_searcher.streaming_top_k_from_file(
                    q, str(npk_path), 'candidates', 'cosine', k=10, batch_size=10000
                )
            stream_time = time.perf_counter() - start
            print(f"  流式（逐个）:    {format_time(stream_time)}")
            
            # 流式多查询（批量优化）
            start = time.perf_counter()
            batch_indices, batch_scores = streaming_searcher.streaming_multi_query_top_k(
                queries, str(npk_path), 'candidates', 'cosine', k=10, batch_size=10000
            )
            stream_batch_time = time.perf_counter() - start
            
            speedup_stream = stream_time / stream_batch_time
            print(f"  流式（批量）:    {format_time(stream_batch_time)} (vs逐个提升: {speedup_stream:.2f}x)")
            
            # 总结
            print(f"\n  [最快方法] {'直接批量' if direct_batch_time < stream_batch_time else '流式批量'}: {format_time(min(direct_batch_time, stream_batch_time))}")


def run_all_benchmarks():
    """运行所有基准测试"""
    print("=" * 70)
    print("StreamingVectorEngine 性能基准测试")
    print("=" * 70)
    
    engine = VectorEngine()
    print(f"\nSIMD能力: {engine.capabilities()}")
    print(f"NumPack版本: {numpack.__version__}")
    
    with StreamingBenchmark() as bench:
        # 1. 批量计算比较
        bench.benchmark_batch_compute_comparison(dim=128, n_candidates=100000)
        
        # 2. Top-K比较
        bench.benchmark_top_k_comparison(dim=128, n_candidates=100000, k=10)
        
        # 3. 内存效率
        bench.benchmark_memory_efficiency(dim=128, n_candidates=100000)
        
        # 4. 大规模测试
        bench.benchmark_large_scale()
        
        # 5. 多查询测试
        bench.benchmark_multi_query(dim=128, n_candidates=50000, n_queries=100)
    
    print("\n" + "=" * 70)
    print("基准测试总结")
    print("=" * 70)
    print("""
流式计算特点：
1. 内存效率：只需要batch_size大小的内存，适合超大数据集
2. 性能开销：相比直接计算有少量开销（通常<20%）
3. Top-K搜索：使用堆维护全局Top-K，正确性有保证
4. 多查询优化：可以逐个查询使用流式搜索

使用建议：
- 数据能完全加载到内存：使用直接计算
- 数据超过内存容量：使用流式计算
- 多查询场景：可以循环调用streaming_top_k_from_file
- batch_size选择：通常10000-50000是较好的平衡点
""")


if __name__ == "__main__":
    run_all_benchmarks()
