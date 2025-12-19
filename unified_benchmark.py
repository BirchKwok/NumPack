#!/usr/bin/env python3
"""
NumPack 综合格式性能基准测试
=====================================

对比格式:
- NumPack (Rust backend)
- NPY (NumPy binary format)
- NPY (with mmap)
- NPZ (NumPy compressed format)
- Zarr (chunked array storage)
- HDF5 (Hierarchical Data Format)
- Parquet (Apache Parquet)
- Arrow/Feather (Apache Arrow)

测试操作:
1. Save/Write (写入)
2. Full Load (完整加载)
3. Lazy Load (懒加载/内存映射)
4. GetItem (索引访问)
5. Replace (替换操作)
6. Append (追加操作)
7. Random Access (随机访问)
8. File Size (文件大小)

测试数据集:
- Large: 1M rows × 10 columns (Float32, ~38.1MB)
- Medium: 100K rows × 10 columns (Float32, ~3.8MB)
"""

import os
import sys
import shutil
import time
import timeit
import tempfile
import gc
from pathlib import Path
from typing import Dict, Any, List, Callable, Tuple
import warnings

import numpy as np
import pandas as pd

# 导入 NumPack
try:
    from numpack import NumPack
    NUMPACK_AVAILABLE = True
except ImportError:
    NUMPACK_AVAILABLE = False
    warnings.warn("NumPack not available, skipping NumPack tests")

# 可选依赖
try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    warnings.warn("Zarr not available, install with: pip install zarr")

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    warnings.warn("h5py not available, install with: pip install h5py")

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.feather as feather
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False
    warnings.warn("PyArrow not available, install with: pip install pyarrow")


class BenchmarkResult:
    """基准测试结果容器"""
    
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def add(self, format_name: str, operation: str, time_ms: float, 
            file_size_mb: float = None, note: str = ""):
        """添加测试结果"""
        if format_name not in self.results:
            self.results[format_name] = {}
        
        self.results[format_name][operation] = {
            'time_ms': time_ms,
            'file_size_mb': file_size_mb,
            'note': note
        }
    
    def add_error(self, format_name: str, operation: str, error: str):
        """添加错误信息"""
        if format_name not in self.results:
            self.results[format_name] = {}
        
        self.results[format_name][operation] = {
            'time_ms': None,
            'error': error
        }
    
    def get_summary(self, operation: str) -> pd.DataFrame:
        """获取指定操作的汇总结果"""
        data = []
        for format_name, ops in self.results.items():
            if operation in ops:
                result = ops[operation]
                if 'error' not in result:
                    data.append({
                        'Format': format_name,
                        'Time (ms)': result['time_ms'],
                        'Note': result.get('note', '')
                    })
        
        df = pd.DataFrame(data)
        if not df.empty and len(df) > 0:
            # 计算相对于最快的倍数
            min_time = df['Time (ms)'].min()
            df['vs Best'] = df['Time (ms)'] / min_time
            df = df.sort_values('Time (ms)')
        
        return df


class FormatBenchmark:
    """格式性能基准测试框架"""
    
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.results = BenchmarkResult()
    
    def get_dir_size_mb(self, path: Path) -> float:
        """获取目录或文件大小(MB)"""
        if path.is_file():
            return path.stat().st_size / (1024 * 1024)
        
        total = 0
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
        return total / (1024 * 1024)
    
    def benchmark_operation(
        self, 
        format_name: str, 
        operation_name: str, 
        operation_func: Callable,
        number: int = 1,
        repeat: int = 3
    ) -> float:
        """
        使用timeit精确测量操作时间
        
        Args:
            format_name: 格式名称
            operation_name: 操作名称
            operation_func: 操作函数
            number: 每次重复执行的次数
            repeat: 重复测试的次数
        
        Returns:
            平均时间（毫秒）
        """
        try:
            # 清理垃圾
            gc.collect()
            
            # 使用timeit测量
            times = timeit.repeat(
                operation_func,
                number=number,
                repeat=repeat,
                globals=globals()
            )
            
            # 取最小值（最佳性能）
            best_time = min(times) / number
            time_ms = best_time * 1000
            
            return time_ms
            
        except Exception as e:
            print(f"  {format_name} {operation_name} failed: {e}")
            self.results.add_error(format_name, operation_name, str(e))
            return None
    
    # ==================== NumPack 测试 ====================
    
    def test_numpack(self, data: np.ndarray, array_name: str = "data"):
        """测试 NumPack 格式（改进版：排除上下文管理器开销）"""
        if not NUMPACK_AVAILABLE:
            return
        
        format_name = "NumPack"
        print(f"\nTesting {format_name}...")
        
        path = self.temp_dir / "numpack_data"
        if path.exists():
            shutil.rmtree(path)
        
        # Save/Write - 测量纯保存时间
        # 先打开文件，然后只测量 save 操作
        npk_save = NumPack(path, drop_if_exists=True)
        npk_save.open()
        
        def save_op():
            npk_save.save({array_name: data})
        
        time_ms = self.benchmark_operation(format_name, "Save", save_op, number=1, repeat=3)
        npk_save.close()
        
        if time_ms:
            file_size = self.get_dir_size_mb(path)
            self.results.add(format_name, "Save", time_ms, file_size)
            print(f"  ✓ Save: {time_ms:.3f}ms ({file_size:.2f}MB)")
        
        # Full Load - 测量纯加载时间
        npk_load = NumPack(path)
        npk_load.open()
        
        def load_op():
            _ = npk_load.load(array_name, lazy=False)
        
        time_ms = self.benchmark_operation(format_name, "Full Load", load_op, number=5, repeat=3)
        npk_load.close()
        
        if time_ms:
            self.results.add(format_name, "Full Load", time_ms)
            print(f"  ✓ Full Load: {time_ms:.3f}ms")
        
        # Lazy Load - 测量纯懒加载时间
        npk_lazy = NumPack(path)
        npk_lazy.open()
        
        def lazy_load_op():
            _ = npk_lazy.load(array_name, lazy=True)
        
        time_ms = self.benchmark_operation(format_name, "Lazy Load", lazy_load_op, number=10, repeat=3)
        npk_lazy.close()
        
        if time_ms:
            self.results.add(format_name, "Lazy Load", time_ms)
            print(f"  ✓ Lazy Load: {time_ms:.3f}ms")
        
        # GetItem[0] - 单行访问
        npk_get = NumPack(path)
        npk_get.open()
        
        def getitem_single_op():
            _ = npk_get.getitem(array_name, [0])
        
        time_ms = self.benchmark_operation(format_name, "GetItem[0]", getitem_single_op, number=100, repeat=3)
        npk_get.close()
        
        if time_ms:
            self.results.add(format_name, "GetItem[0]", time_ms)
            print(f"  ✓ GetItem[0]: {time_ms:.3f}ms")
        
        # GetItem[:100] - 范围访问
        npk_range = NumPack(path)
        npk_range.open()
        
        def getitem_range_op():
            _ = npk_range.getitem(array_name, list(range(100)))
        
        time_ms = self.benchmark_operation(format_name, "GetItem[:100]", getitem_range_op, number=10, repeat=3)
        npk_range.close()
        
        if time_ms:
            self.results.add(format_name, "GetItem[:100]", time_ms)
            print(f"  ✓ GetItem[:100]: {time_ms:.3f}ms")
        
        # ==================== Random Access Tests ====================
        # 注意：必须在 replace/append 之前测试，因为它们会修改文件状态
        # 重要：为确保测试的准确性，在随机访问测试前重新创建干净的文件
        with NumPack(path, drop_if_exists=True) as npk:
            npk.save({array_name: data})
        
        print(f"\n  Testing Random Access Performance...")
        
        # Random Access - Small Batch (100 indices)
        random_indices_small = np.random.randint(0, min(len(data), 10000), 100).tolist()
        npk_random_small = NumPack(path)
        npk_random_small.open()
        
        def random_access_small_op():
            _ = npk_random_small.getitem(array_name, random_indices_small)
        
        time_ms = self.benchmark_operation(format_name, "Random Access (100)", random_access_small_op, number=10, repeat=3)
        npk_random_small.close()
        
        if time_ms:
            self.results.add(format_name, "Random Access (100)", time_ms)
            print(f"  ✓ Random Access (100): {time_ms:.3f}ms")
        
        # Random Access - Medium Batch (1000 indices)
        random_indices_medium = np.random.randint(0, min(len(data), 10000), 1000).tolist()
        npk_random_medium = NumPack(path)
        npk_random_medium.open()
        
        def random_access_medium_op():
            _ = npk_random_medium.getitem(array_name, random_indices_medium)
        
        time_ms = self.benchmark_operation(format_name, "Random Access (1K)", random_access_medium_op, number=5, repeat=3)
        npk_random_medium.close()
        
        if time_ms:
            self.results.add(format_name, "Random Access (1K)", time_ms)
            print(f"  ✓ Random Access (1K): {time_ms:.3f}ms")
        
        # Random Access - Large Batch (10000 indices)
        random_indices_large = np.random.randint(0, len(data), 10000).tolist()
        npk_random_large = NumPack(path)
        npk_random_large.open()
        
        def random_access_large_op():
            _ = npk_random_large.getitem(array_name, random_indices_large)
        
        time_ms = self.benchmark_operation(format_name, "Random Access (10K)", random_access_large_op, number=3, repeat=3)
        npk_random_large.close()
        
        if time_ms:
            self.results.add(format_name, "Random Access (10K)", time_ms)
            print(f"  ✓ Random Access (10K): {time_ms:.3f}ms")
        
        # ==================== Sequential Access Tests ====================
        print(f"\n  Testing Sequential Access Performance...")
        
        # Sequential Access - Small Batch (100 consecutive rows)
        # 使用slice以利用零复制优化
        npk_seq_small = NumPack(path)
        npk_seq_small.open()
        
        def seq_access_small_op():
            _ = npk_seq_small.getitem(array_name, slice(0, 100))
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (100)", seq_access_small_op, number=10, repeat=3)
        npk_seq_small.close()
        
        if time_ms:
            self.results.add(format_name, "Sequential Access (100)", time_ms)
            print(f"  ✓ Sequential Access (100): {time_ms:.3f}ms")
        
        # Sequential Access - Medium Batch (1000 consecutive rows)
        npk_seq_medium = NumPack(path)
        npk_seq_medium.open()
        
        def seq_access_medium_op():
            _ = npk_seq_medium.getitem(array_name, slice(0, 1000))
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (1K)", seq_access_medium_op, number=5, repeat=3)
        npk_seq_medium.close()
        
        if time_ms:
            self.results.add(format_name, "Sequential Access (1K)", time_ms)
            print(f"  ✓ Sequential Access (1K): {time_ms:.3f}ms")
        
        # Sequential Access - Large Batch (10000 consecutive rows)
        npk_seq_large = NumPack(path)
        npk_seq_large.open()
        
        def seq_access_large_op():
            _ = npk_seq_large.getitem(array_name, slice(0, 10000))
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (10K)", seq_access_large_op, number=3, repeat=3)
        npk_seq_large.close()
        
        if time_ms:
            self.results.add(format_name, "Sequential Access (10K)", time_ms)
            print(f"  ✓ Sequential Access (10K): {time_ms:.3f}ms")
        
        # ==================== Batch Mode 测试 ====================
        print(f"\n  Testing Batch Mode Performance...")
        
        # 在 Batch Mode 测试之前，先执行 Replace 和 Append 测试
        # （这些会修改文件，所以放在最后）
        
        # Replace 100 rows - 测量纯替换时间
        replace_data = np.random.rand(100, data.shape[1]).astype(data.dtype)
        replace_indices = list(range(100))
        
        npk_replace = NumPack(path)
        npk_replace.open()
        
        def replace_op():
            npk_replace.replace({array_name: replace_data}, replace_indices)
        
        time_ms = self.benchmark_operation(format_name, "Replace 100", replace_op, number=5, repeat=3)
        npk_replace.close()
        
        if time_ms:
            self.results.add(format_name, "Replace 100", time_ms)
            print(f"  ✓ Replace 100: {time_ms:.3f}ms (moved before batch mode)")
        
        # Append 100 rows - 测量纯追加时间
        append_data = np.random.rand(100, data.shape[1]).astype(data.dtype)
        
        # 先重置数据
        with NumPack(path, drop_if_exists=True) as npk:
            npk.save({array_name: data})
        
        npk_append = NumPack(path)
        npk_append.open()
        
        def append_op():
            npk_append.append({array_name: append_data})
        
        time_ms = self.benchmark_operation(format_name, "Append 100", append_op, number=1, repeat=3)
        npk_append.close()
        
        if time_ms:
            self.results.add(format_name, "Append 100", time_ms)
            print(f"  ✓ Append 100: {time_ms:.3f}ms (moved before batch mode)")
        
        # Batch Mode: 连续修改 100 次
        num_iterations = 100
        modify_data = np.random.rand(10, data.shape[1]).astype(data.dtype)
        modify_indices = list(range(10))
        
        # 1. 普通模式 - 100次独立操作
        with NumPack(path, drop_if_exists=True) as npk:
            npk.save({array_name: data})
        
        npk_normal = NumPack(path)
        npk_normal.open()
        
        def normal_mode_op():
            for _ in range(num_iterations):
                arr = npk_normal.load(array_name, lazy=False)
                arr[:10] *= 2.0
                npk_normal.replace({array_name: arr[:10]}, modify_indices)
        
        time_normal = self.benchmark_operation(format_name, "Normal Mode (100x modify)", normal_mode_op, number=1, repeat=3)
        npk_normal.close()
        
        if time_normal:
            self.results.add(format_name, "Normal Mode (100x)", time_normal)
            print(f"  ✓ Normal Mode (100x modify): {time_normal:.3f}ms")
        
        # 2. Batch Mode - 100次操作（内存缓存）
        with NumPack(path, drop_if_exists=True) as npk:
            npk.save({array_name: data})
        
        npk_batch = NumPack(path)
        npk_batch.open()
        
        def batch_mode_op():
            with npk_batch.batch_mode():
                for _ in range(num_iterations):
                    arr = npk_batch.load(array_name)
                    arr[:10] *= 2.0
                    npk_batch.save({array_name: arr})
        
        time_batch = self.benchmark_operation(format_name, "Batch Mode (100x modify)", batch_mode_op, number=1, repeat=3)
        npk_batch.close()
        
        if time_batch:
            self.results.add(format_name, "Batch Mode (100x)", time_batch)
            speedup = time_normal / time_batch if time_normal and time_batch else 0
            print(f"  ✓ Batch Mode (100x modify): {time_batch:.3f}ms (speedup: {speedup:.1f}x)")
        
        # 3. Writable Batch Mode - 100次操作（零拷贝）
        # 注意：writable_batch_mode 不支持改变数组形状，只能就地修改
        with NumPack(path, drop_if_exists=True) as npk:
            npk.save({array_name: data})
        
        try:
            from numpack.writable_array import WritableBatchMode
            
            npk_writable = NumPack(path)
            npk_writable.open()
            
            def writable_batch_mode_op():
                with npk_writable.writable_batch_mode() as wb:
                    for _ in range(num_iterations):
                        arr = wb.load(array_name)
                        arr[:10] *= 2.0
                        # writable_batch_mode 中 save 是无操作
            
            time_writable = self.benchmark_operation(format_name, "Writable Batch Mode (100x modify)", writable_batch_mode_op, number=1, repeat=3)
            npk_writable.close()
            
            if time_writable:
                self.results.add(format_name, "Writable Batch Mode (100x)", time_writable)
                speedup = time_normal / time_writable if time_normal and time_writable else 0
                print(f"  ✓ Writable Batch Mode (100x modify): {time_writable:.3f}ms (speedup: {speedup:.1f}x)")
        except Exception as e:
            print(f"  Writable Batch Mode test skipped: {e}")
    
    # ==================== NPY 测试 ====================
    
    def test_npy(self, data: np.ndarray):
        """测试 NPY 格式"""
        format_name = "NPY"
        print(f"\nTesting {format_name}...")
        
        path = self.temp_dir / "data.npy"
        
        # Save
        def save_op():
            np.save(path, data)
        
        time_ms = self.benchmark_operation(format_name, "Save", save_op, number=1, repeat=3)
        if time_ms:
            file_size = self.get_dir_size_mb(path)
            self.results.add(format_name, "Save", time_ms, file_size)
            print(f"  ✓ Save: {time_ms:.3f}ms ({file_size:.2f}MB)")
        
        # Full Load
        def load_op():
            _ = np.load(path)
        
        time_ms = self.benchmark_operation(format_name, "Full Load", load_op, number=5, repeat=3)
        if time_ms:
            self.results.add(format_name, "Full Load", time_ms)
            print(f"  ✓ Full Load: {time_ms:.3f}ms")
        
        # Lazy Load (mmap)
        def lazy_load_op():
            _ = np.load(path, mmap_mode='r')
        
        time_ms = self.benchmark_operation(format_name, "Lazy Load", lazy_load_op, number=10, repeat=3)
        if time_ms:
            self.results.add(format_name, "Lazy Load", time_ms, note="mmap")
            print(f"  ✓ Lazy Load: {time_ms:.3f}ms (mmap)")
        
        # GetItem operations
        arr_mmap = np.load(path, mmap_mode='r')
        
        def getitem_single_op():
            _ = arr_mmap[0]
        
        time_ms = self.benchmark_operation(format_name, "GetItem[0]", getitem_single_op, number=100, repeat=3)
        if time_ms:
            self.results.add(format_name, "GetItem[0]", time_ms)
            print(f"  ✓ GetItem[0]: {time_ms:.3f}ms")
        
        def getitem_range_op():
            _ = arr_mmap[:100]
        
        time_ms = self.benchmark_operation(format_name, "GetItem[:100]", getitem_range_op, number=10, repeat=3)
        if time_ms:
            self.results.add(format_name, "GetItem[:100]", time_ms)
            print(f"  ✓ GetItem[:100]: {time_ms:.3f}ms")
        
        # Replace (需要重写整个文件)
        replace_data = np.random.rand(100, data.shape[1]).astype(data.dtype)
        
        def replace_op():
            arr = np.load(path)
            arr[:100] = replace_data
            np.save(path, arr)
        
        time_ms = self.benchmark_operation(format_name, "Replace 100", replace_op, number=1, repeat=3)
        if time_ms:
            self.results.add(format_name, "Replace 100", time_ms, note="full rewrite")
            print(f"  ✓ Replace 100: {time_ms:.3f}ms (full rewrite)")
        
        # Append
        append_data = np.random.rand(100, data.shape[1]).astype(data.dtype)
        
        def append_op():
            arr = np.load(path)
            arr_new = np.vstack([arr, append_data])
            np.save(path, arr_new)
        
        time_ms = self.benchmark_operation(format_name, "Append 100", append_op, number=1, repeat=3)
        if time_ms:
            self.results.add(format_name, "Append 100", time_ms, note="full rewrite")
            print(f"  ✓ Append 100: {time_ms:.3f}ms (full rewrite)")
        
        # Random Access Tests
        # Note: Force actual data access by converting to array (not just creating view)
        # Small Batch (100)
        random_indices_small = np.random.randint(0, min(len(data), 10000), 100)
        
        def random_access_small_op():
            result = arr_mmap[random_indices_small]
            _ = np.array(result)  # Force actual data read from mmap
        
        time_ms = self.benchmark_operation(format_name, "Random Access (100)", random_access_small_op, number=10, repeat=3)
        if time_ms:
            self.results.add(format_name, "Random Access (100)", time_ms)
            print(f"  ✓ Random Access (100): {time_ms:.3f}ms")
        
        # Medium Batch (1K)
        random_indices_medium = np.random.randint(0, min(len(data), 10000), 1000)
        
        def random_access_medium_op():
            result = arr_mmap[random_indices_medium]
            _ = np.array(result)  # Force actual data read from mmap
        
        time_ms = self.benchmark_operation(format_name, "Random Access (1K)", random_access_medium_op, number=5, repeat=3)
        if time_ms:
            self.results.add(format_name, "Random Access (1K)", time_ms)
            print(f"  ✓ Random Access (1K): {time_ms:.3f}ms")
        
        # Large Batch (10K)
        random_indices_large = np.random.randint(0, len(data), 10000)
        
        def random_access_large_op():
            result = arr_mmap[random_indices_large]
            _ = np.array(result)  # Force actual data read from mmap
        
        time_ms = self.benchmark_operation(format_name, "Random Access (10K)", random_access_large_op, number=3, repeat=3)
        if time_ms:
            self.results.add(format_name, "Random Access (10K)", time_ms)
            print(f"  ✓ Random Access (10K): {time_ms:.3f}ms")
        
        # Sequential Access Tests
        # Note: Force actual data access by converting to array (not just creating view)
        # Small Batch (100)
        def seq_access_small_op():
            result = arr_mmap[:100]
            _ = np.array(result)  # Force actual data read from mmap
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (100)", seq_access_small_op, number=10, repeat=3)
        if time_ms:
            self.results.add(format_name, "Sequential Access (100)", time_ms)
            print(f"  ✓ Sequential Access (100): {time_ms:.3f}ms")
        
        # Medium Batch (1K)
        def seq_access_medium_op():
            result = arr_mmap[:1000]
            _ = np.array(result)  # Force actual data read from mmap
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (1K)", seq_access_medium_op, number=5, repeat=3)
        if time_ms:
            self.results.add(format_name, "Sequential Access (1K)", time_ms)
            print(f"  ✓ Sequential Access (1K): {time_ms:.3f}ms")
        
        # Large Batch (10K)
        def seq_access_large_op():
            result = arr_mmap[:10000]
            _ = np.array(result)  # Force actual data read from mmap
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (10K)", seq_access_large_op, number=3, repeat=3)
        if time_ms:
            self.results.add(format_name, "Sequential Access (10K)", time_ms)
            print(f"  ✓ Sequential Access (10K): {time_ms:.3f}ms")
    
    # ==================== NPZ 测试 ====================
    
    def test_npz(self, data: np.ndarray, array_name: str = "data"):
        """测试 NPZ 格式"""
        format_name = "NPZ"
        print(f"\nTesting {format_name}...")
        
        path = self.temp_dir / "data.npz"
        
        # Save
        def save_op():
            np.savez_compressed(path, **{array_name: data})
        
        time_ms = self.benchmark_operation(format_name, "Save", save_op, number=1, repeat=3)
        if time_ms:
            file_size = self.get_dir_size_mb(path)
            self.results.add(format_name, "Save", time_ms, file_size)
            print(f"  ✓ Save: {time_ms:.3f}ms ({file_size:.2f}MB)")
        
        # Full Load
        def load_op():
            with np.load(path) as npz:
                _ = npz[array_name]
        
        time_ms = self.benchmark_operation(format_name, "Full Load", load_op, number=5, repeat=3)
        if time_ms:
            self.results.add(format_name, "Full Load", time_ms)
            print(f"  ✓ Full Load: {time_ms:.3f}ms")
        
        # NPZ不支持真正的lazy load（压缩格式必须解压）
        self.results.add(format_name, "Lazy Load", None, note="N/A (compressed)")
        print(f"  Lazy Load: N/A (compressed format)")
        
        # Replace
        replace_data = np.random.rand(100, data.shape[1]).astype(data.dtype)
        
        def replace_op():
            with np.load(path) as npz:
                arr = npz[array_name].copy()
            arr[:100] = replace_data
            np.savez_compressed(path, **{array_name: arr})
        
        time_ms = self.benchmark_operation(format_name, "Replace 100", replace_op, number=1, repeat=3)
        if time_ms:
            self.results.add(format_name, "Replace 100", time_ms, note="full rewrite")
            print(f"  ✓ Replace 100: {time_ms:.3f}ms (full rewrite)")
        
        # Append
        append_data = np.random.rand(100, data.shape[1]).astype(data.dtype)
        
        def append_op():
            with np.load(path) as npz:
                arr = npz[array_name].copy()
            arr_new = np.vstack([arr, append_data])
            np.savez_compressed(path, **{array_name: arr_new})
        
        time_ms = self.benchmark_operation(format_name, "Append 100", append_op, number=1, repeat=3)
        if time_ms:
            self.results.add(format_name, "Append 100", time_ms, note="full rewrite")
            print(f"  ✓ Append 100: {time_ms:.3f}ms (full rewrite)")
        
        # Random Access Tests
        # Small Batch (100)
        random_indices_small = np.random.randint(0, min(len(data), 10000), 100)
        
        def random_access_small_op():
            with np.load(path) as npz:
                arr = npz[array_name]
                _ = arr[random_indices_small]
        
        time_ms = self.benchmark_operation(format_name, "Random Access (100)", random_access_small_op, number=10, repeat=3)
        if time_ms:
            self.results.add(format_name, "Random Access (100)", time_ms)
            print(f"  ✓ Random Access (100): {time_ms:.3f}ms")
        
        # Medium Batch (1K)
        random_indices_medium = np.random.randint(0, min(len(data), 10000), 1000)
        
        def random_access_medium_op():
            with np.load(path) as npz:
                arr = npz[array_name]
                _ = arr[random_indices_medium]
        
        time_ms = self.benchmark_operation(format_name, "Random Access (1K)", random_access_medium_op, number=5, repeat=3)
        if time_ms:
            self.results.add(format_name, "Random Access (1K)", time_ms)
            print(f"  ✓ Random Access (1K): {time_ms:.3f}ms")
        
        # Large Batch (10K)
        random_indices_large = np.random.randint(0, len(data), 10000)
        
        def random_access_large_op():
            with np.load(path) as npz:
                arr = npz[array_name]
                _ = arr[random_indices_large]
        
        time_ms = self.benchmark_operation(format_name, "Random Access (10K)", random_access_large_op, number=3, repeat=3)
        if time_ms:
            self.results.add(format_name, "Random Access (10K)", time_ms)
            print(f"  ✓ Random Access (10K): {time_ms:.3f}ms")
        
        # Sequential Access Tests
        # Small Batch (100)
        def seq_access_small_op():
            with np.load(path) as npz:
                arr = npz[array_name]
                _ = arr[:100]
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (100)", seq_access_small_op, number=10, repeat=3)
        if time_ms:
            self.results.add(format_name, "Sequential Access (100)", time_ms)
            print(f"  ✓ Sequential Access (100): {time_ms:.3f}ms")
        
        # Medium Batch (1K)
        def seq_access_medium_op():
            with np.load(path) as npz:
                arr = npz[array_name]
                _ = arr[:1000]
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (1K)", seq_access_medium_op, number=5, repeat=3)
        if time_ms:
            self.results.add(format_name, "Sequential Access (1K)", time_ms)
            print(f"  ✓ Sequential Access (1K): {time_ms:.3f}ms")
        
        # Large Batch (10K)
        def seq_access_large_op():
            with np.load(path) as npz:
                arr = npz[array_name]
                _ = arr[:10000]
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (10K)", seq_access_large_op, number=3, repeat=3)
        if time_ms:
            self.results.add(format_name, "Sequential Access (10K)", time_ms)
            print(f"  ✓ Sequential Access (10K): {time_ms:.3f}ms")
    
    # ==================== Zarr 测试 ====================
    
    def test_zarr(self, data: np.ndarray, array_name: str = "data"):
        """测试 Zarr 格式"""
        if not ZARR_AVAILABLE:
            print("\nZarr not available, skipping...")
            return
        
        format_name = "Zarr"
        print(f"\nTesting {format_name}...")
        
        path = self.temp_dir / "data.zarr"
        if path.exists():
            shutil.rmtree(path)
        
        # Save
        def save_op():
            if path.exists():
                shutil.rmtree(path)
            zarr.save(str(path), data)
        
        time_ms = self.benchmark_operation(format_name, "Save", save_op, number=1, repeat=3)
        if time_ms:
            file_size = self.get_dir_size_mb(path)
            self.results.add(format_name, "Save", time_ms, file_size)
            print(f"  ✓ Save: {time_ms:.3f}ms ({file_size:.2f}MB)")
        
        # Full Load
        def load_op():
            _ = zarr.load(str(path))
        
        time_ms = self.benchmark_operation(format_name, "Full Load", load_op, number=5, repeat=3)
        if time_ms:
            self.results.add(format_name, "Full Load", time_ms)
            print(f"  ✓ Full Load: {time_ms:.3f}ms")
        
        # Lazy Load
        def lazy_load_op():
            _ = zarr.open(str(path), mode='r')
        
        time_ms = self.benchmark_operation(format_name, "Lazy Load", lazy_load_op, number=10, repeat=3)
        if time_ms:
            self.results.add(format_name, "Lazy Load", time_ms)
            print(f"  ✓ Lazy Load: {time_ms:.3f}ms")
        
        # GetItem operations
        z = zarr.open(str(path), mode='r')
        
        def getitem_single_op():
            _ = z[0]
        
        time_ms = self.benchmark_operation(format_name, "GetItem[0]", getitem_single_op, number=100, repeat=3)
        if time_ms:
            self.results.add(format_name, "GetItem[0]", time_ms)
            print(f"  ✓ GetItem[0]: {time_ms:.3f}ms")
        
        def getitem_range_op():
            _ = z[:100]
        
        time_ms = self.benchmark_operation(format_name, "GetItem[:100]", getitem_range_op, number=10, repeat=3)
        if time_ms:
            self.results.add(format_name, "GetItem[:100]", time_ms)
            print(f"  ✓ GetItem[:100]: {time_ms:.3f}ms")
        
        # Replace (Zarr 支持部分写入)
        replace_data = np.random.rand(100, data.shape[1]).astype(data.dtype)
        
        def replace_op():
            z_w = zarr.open(str(path), mode='r+')
            z_w[:100] = replace_data
        
        time_ms = self.benchmark_operation(format_name, "Replace 100", replace_op, number=5, repeat=3)
        if time_ms:
            self.results.add(format_name, "Replace 100", time_ms, note="in-place")
            print(f"  ✓ Replace 100: {time_ms:.3f}ms (in-place)")
        
        # Append (Zarr 支持 append，但需要调整大小)
        append_data = np.random.rand(100, data.shape[1]).astype(data.dtype)
        
        def append_op():
            z_w = zarr.open(str(path), mode='r+')
            old_len = z_w.shape[0]
            new_shape = (old_len + 100,) + z_w.shape[1:]
            z_w.resize(new_shape)
            z_w[old_len:] = append_data
            # 恢复原始大小
            z_w.resize(data.shape)
        
        time_ms = self.benchmark_operation(format_name, "Append 100", append_op, number=1, repeat=3)
        if time_ms:
            self.results.add(format_name, "Append 100", time_ms)
            print(f"  ✓ Append 100: {time_ms:.3f}ms")
        
        # Random Access Tests
        # Small Batch (100)
        random_indices_small = np.random.randint(0, min(len(data), 10000), 100)
        
        def random_access_small_op():
            _ = z[random_indices_small]
        
        time_ms = self.benchmark_operation(format_name, "Random Access (100)", random_access_small_op, number=10, repeat=3)
        if time_ms:
            self.results.add(format_name, "Random Access (100)", time_ms)
            print(f"  ✓ Random Access (100): {time_ms:.3f}ms")
        
        # Medium Batch (1K)
        random_indices_medium = np.random.randint(0, min(len(data), 10000), 1000)
        
        def random_access_medium_op():
            _ = z[random_indices_medium]
        
        time_ms = self.benchmark_operation(format_name, "Random Access (1K)", random_access_medium_op, number=5, repeat=3)
        if time_ms:
            self.results.add(format_name, "Random Access (1K)", time_ms)
            print(f"  ✓ Random Access (1K): {time_ms:.3f}ms")
        
        # Large Batch (10K)
        random_indices_large = np.random.randint(0, len(data), 10000)
        
        def random_access_large_op():
            _ = z[random_indices_large]
        
        time_ms = self.benchmark_operation(format_name, "Random Access (10K)", random_access_large_op, number=3, repeat=3)
        if time_ms:
            self.results.add(format_name, "Random Access (10K)", time_ms)
            print(f"  ✓ Random Access (10K): {time_ms:.3f}ms")
        
        # Sequential Access Tests
        # Small Batch (100)
        def seq_access_small_op():
            _ = z[:100]
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (100)", seq_access_small_op, number=10, repeat=3)
        if time_ms:
            self.results.add(format_name, "Sequential Access (100)", time_ms)
            print(f"  ✓ Sequential Access (100): {time_ms:.3f}ms")
        
        # Medium Batch (1K)
        def seq_access_medium_op():
            _ = z[:1000]
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (1K)", seq_access_medium_op, number=5, repeat=3)
        if time_ms:
            self.results.add(format_name, "Sequential Access (1K)", time_ms)
            print(f"  ✓ Sequential Access (1K): {time_ms:.3f}ms")
        
        # Large Batch (10K)
        def seq_access_large_op():
            _ = z[:10000]
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (10K)", seq_access_large_op, number=3, repeat=3)
        if time_ms:
            self.results.add(format_name, "Sequential Access (10K)", time_ms)
            print(f"  ✓ Sequential Access (10K): {time_ms:.3f}ms")
    
    # ==================== HDF5 测试 ====================
    
    def test_hdf5(self, data: np.ndarray, dataset_name: str = "data"):
        """测试 HDF5 格式"""
        if not HDF5_AVAILABLE:
            print("\nHDF5 not available, skipping...")
            return
        
        format_name = "HDF5"
        print(f"\nTesting {format_name}...")
        
        path = self.temp_dir / "data.h5"
        if path.exists():
            path.unlink()
        
        # Save
        def save_op():
            with h5py.File(path, 'w') as f:
                # 创建可调整大小的数据集
                f.create_dataset(dataset_name, data=data, maxshape=(None,) + data.shape[1:])
        
        time_ms = self.benchmark_operation(format_name, "Save", save_op, number=1, repeat=3)
        if time_ms:
            file_size = self.get_dir_size_mb(path)
            self.results.add(format_name, "Save", time_ms, file_size)
            print(f"  ✓ Save: {time_ms:.3f}ms ({file_size:.2f}MB)")
        
        # Full Load
        def load_op():
            with h5py.File(path, 'r') as f:
                _ = f[dataset_name][:]
        
        time_ms = self.benchmark_operation(format_name, "Full Load", load_op, number=5, repeat=3)
        if time_ms:
            self.results.add(format_name, "Full Load", time_ms)
            print(f"  ✓ Full Load: {time_ms:.3f}ms")
        
        # Lazy Load
        def lazy_load_op():
            f = h5py.File(path, 'r')
            _ = f[dataset_name]
            f.close()
        
        time_ms = self.benchmark_operation(format_name, "Lazy Load", lazy_load_op, number=10, repeat=3)
        if time_ms:
            self.results.add(format_name, "Lazy Load", time_ms)
            print(f"  ✓ Lazy Load: {time_ms:.3f}ms")
        
        # GetItem operations
        with h5py.File(path, 'r') as f:
            dset = f[dataset_name]
            
            def getitem_single_op():
                _ = dset[0]
            
            time_ms = self.benchmark_operation(format_name, "GetItem[0]", getitem_single_op, number=100, repeat=3)
            if time_ms:
                self.results.add(format_name, "GetItem[0]", time_ms)
                print(f"  ✓ GetItem[0]: {time_ms:.3f}ms")
            
            def getitem_range_op():
                _ = dset[:100]
            
            time_ms = self.benchmark_operation(format_name, "GetItem[:100]", getitem_range_op, number=10, repeat=3)
            if time_ms:
                self.results.add(format_name, "GetItem[:100]", time_ms)
                print(f"  ✓ GetItem[:100]: {time_ms:.3f}ms")
        
        # Replace (HDF5 支持部分写入)
        replace_data = np.random.rand(100, data.shape[1]).astype(data.dtype)
        
        def replace_op():
            with h5py.File(path, 'r+') as f:
                f[dataset_name][:100] = replace_data
        
        time_ms = self.benchmark_operation(format_name, "Replace 100", replace_op, number=5, repeat=3)
        if time_ms:
            self.results.add(format_name, "Replace 100", time_ms, note="in-place")
            print(f"  ✓ Replace 100: {time_ms:.3f}ms (in-place)")
        
        # Append (HDF5 支持 resize)
        append_data = np.random.rand(100, data.shape[1]).astype(data.dtype)
        
        def append_op():
            with h5py.File(path, 'r+') as f:
                dset = f[dataset_name]
                old_len = dset.shape[0]
                dset.resize(old_len + 100, axis=0)
                dset[old_len:] = append_data
                # 恢复原始大小
                dset.resize(data.shape[0], axis=0)
        
        time_ms = self.benchmark_operation(format_name, "Append 100", append_op, number=1, repeat=3)
        if time_ms:
            self.results.add(format_name, "Append 100", time_ms)
            print(f"  ✓ Append 100: {time_ms:.3f}ms")
        
        # Random Access Tests (HDF5 需要逐个读取或使用mask)
        # Small Batch (100)
        random_indices_small = np.random.randint(0, min(len(data), 10000), 100)
        
        def random_access_small_op():
            with h5py.File(path, 'r') as f:
                dset = f[dataset_name]
                _ = np.array([dset[i] for i in random_indices_small])
        
        time_ms = self.benchmark_operation(format_name, "Random Access (100)", random_access_small_op, number=10, repeat=3)
        if time_ms:
            self.results.add(format_name, "Random Access (100)", time_ms, note="sequential fetch")
            print(f"  ✓ Random Access (100): {time_ms:.3f}ms (sequential fetch)")
        
        # Medium Batch (1K)
        random_indices_medium = np.random.randint(0, min(len(data), 10000), 1000)
        
        def random_access_medium_op():
            with h5py.File(path, 'r') as f:
                dset = f[dataset_name]
                _ = np.array([dset[i] for i in random_indices_medium])
        
        time_ms = self.benchmark_operation(format_name, "Random Access (1K)", random_access_medium_op, number=5, repeat=3)
        if time_ms:
            self.results.add(format_name, "Random Access (1K)", time_ms, note="sequential fetch")
            print(f"  ✓ Random Access (1K): {time_ms:.3f}ms (sequential fetch)")
        
        # Large Batch (10K)
        random_indices_large = np.random.randint(0, len(data), 10000)
        
        def random_access_large_op():
            with h5py.File(path, 'r') as f:
                dset = f[dataset_name]
                _ = np.array([dset[i] for i in random_indices_large])
        
        time_ms = self.benchmark_operation(format_name, "Random Access (10K)", random_access_large_op, number=3, repeat=3)
        if time_ms:
            self.results.add(format_name, "Random Access (10K)", time_ms, note="sequential fetch")
            print(f"  ✓ Random Access (10K): {time_ms:.3f}ms (sequential fetch)")
        
        # Sequential Access Tests
        # Small Batch (100)
        def seq_access_small_op():
            with h5py.File(path, 'r') as f:
                dset = f[dataset_name]
                _ = dset[:100]
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (100)", seq_access_small_op, number=10, repeat=3)
        if time_ms:
            self.results.add(format_name, "Sequential Access (100)", time_ms)
            print(f"  ✓ Sequential Access (100): {time_ms:.3f}ms")
        
        # Medium Batch (1K)
        def seq_access_medium_op():
            with h5py.File(path, 'r') as f:
                dset = f[dataset_name]
                _ = dset[:1000]
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (1K)", seq_access_medium_op, number=5, repeat=3)
        if time_ms:
            self.results.add(format_name, "Sequential Access (1K)", time_ms)
            print(f"  ✓ Sequential Access (1K): {time_ms:.3f}ms")
        
        # Large Batch (10K)
        def seq_access_large_op():
            with h5py.File(path, 'r') as f:
                dset = f[dataset_name]
                _ = dset[:10000]
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (10K)", seq_access_large_op, number=3, repeat=3)
        if time_ms:
            self.results.add(format_name, "Sequential Access (10K)", time_ms)
            print(f"  ✓ Sequential Access (10K): {time_ms:.3f}ms")
    
    # ==================== Parquet 测试 ====================
    
    def test_parquet(self, data: np.ndarray):
        """测试 Parquet 格式"""
        if not ARROW_AVAILABLE:
            print("\nParquet not available, skipping...")
            return
        
        format_name = "Parquet"
        print(f"\nTesting {format_name}...")
        
        path = self.temp_dir / "data.parquet"
        
        # 转换为 DataFrame
        df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(data.shape[1])])
        
        # Save
        def save_op():
            df.to_parquet(path, engine='pyarrow', compression='snappy')
        
        time_ms = self.benchmark_operation(format_name, "Save", save_op, number=1, repeat=3)
        if time_ms:
            file_size = self.get_dir_size_mb(path)
            self.results.add(format_name, "Save", time_ms, file_size)
            print(f"  ✓ Save: {time_ms:.3f}ms ({file_size:.2f}MB)")
        
        # Full Load
        def load_op():
            _ = pd.read_parquet(path, engine='pyarrow').values
        
        time_ms = self.benchmark_operation(format_name, "Full Load", load_op, number=5, repeat=3)
        if time_ms:
            self.results.add(format_name, "Full Load", time_ms)
            print(f"  ✓ Full Load: {time_ms:.3f}ms")
        
        # Parquet 不支持真正的 lazy load
        self.results.add(format_name, "Lazy Load", None, note="N/A (columnar)")
        print(f"  Lazy Load: N/A (columnar format)")
        
        # Replace (需要重写)
        replace_data = np.random.rand(100, data.shape[1]).astype(data.dtype)
        
        def replace_op():
            df_load = pd.read_parquet(path)
            df_load.iloc[:100] = replace_data
            df_load.to_parquet(path, engine='pyarrow', compression='snappy')
        
        time_ms = self.benchmark_operation(format_name, "Replace 100", replace_op, number=1, repeat=3)
        if time_ms:
            self.results.add(format_name, "Replace 100", time_ms, note="full rewrite")
            print(f"  ✓ Replace 100: {time_ms:.3f}ms (full rewrite)")
        
        # Append
        append_data = np.random.rand(100, data.shape[1]).astype(data.dtype)
        df_append = pd.DataFrame(append_data, columns=df.columns)
        
        def append_op():
            df_load = pd.read_parquet(path)
            df_new = pd.concat([df_load, df_append], ignore_index=True)
            df_new.to_parquet(path, engine='pyarrow', compression='snappy')
        
        time_ms = self.benchmark_operation(format_name, "Append 100", append_op, number=1, repeat=3)
        if time_ms:
            self.results.add(format_name, "Append 100", time_ms, note="full rewrite")
            print(f"  ✓ Append 100: {time_ms:.3f}ms (full rewrite)")
        
        # Random Access Tests (需要加载全部数据)
        # Small Batch (100)
        random_indices_small = np.random.randint(0, min(len(data), 10000), 100)
        
        def random_access_small_op():
            df_load = pd.read_parquet(path)
            _ = df_load.iloc[random_indices_small].values
        
        time_ms = self.benchmark_operation(format_name, "Random Access (100)", random_access_small_op, number=10, repeat=3)
        if time_ms:
            self.results.add(format_name, "Random Access (100)", time_ms, note="full load")
            print(f"  ✓ Random Access (100): {time_ms:.3f}ms (full load)")
        
        # Medium Batch (1K)
        random_indices_medium = np.random.randint(0, min(len(data), 10000), 1000)
        
        def random_access_medium_op():
            df_load = pd.read_parquet(path)
            _ = df_load.iloc[random_indices_medium].values
        
        time_ms = self.benchmark_operation(format_name, "Random Access (1K)", random_access_medium_op, number=5, repeat=3)
        if time_ms:
            self.results.add(format_name, "Random Access (1K)", time_ms, note="full load")
            print(f"  ✓ Random Access (1K): {time_ms:.3f}ms (full load)")
        
        # Large Batch (10K)
        random_indices_large = np.random.randint(0, len(data), 10000)
        
        def random_access_large_op():
            df_load = pd.read_parquet(path)
            _ = df_load.iloc[random_indices_large].values
        
        time_ms = self.benchmark_operation(format_name, "Random Access (10K)", random_access_large_op, number=3, repeat=3)
        if time_ms:
            self.results.add(format_name, "Random Access (10K)", time_ms, note="full load")
            print(f"  ✓ Random Access (10K): {time_ms:.3f}ms (full load)")
        
        # Sequential Access Tests
        # Small Batch (100)
        def seq_access_small_op():
            df_load = pd.read_parquet(path)
            _ = df_load.iloc[:100].values
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (100)", seq_access_small_op, number=10, repeat=3)
        if time_ms:
            self.results.add(format_name, "Sequential Access (100)", time_ms, note="full load")
            print(f"  ✓ Sequential Access (100): {time_ms:.3f}ms (full load)")
        
        # Medium Batch (1K)
        def seq_access_medium_op():
            df_load = pd.read_parquet(path)
            _ = df_load.iloc[:1000].values
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (1K)", seq_access_medium_op, number=5, repeat=3)
        if time_ms:
            self.results.add(format_name, "Sequential Access (1K)", time_ms, note="full load")
            print(f"  ✓ Sequential Access (1K): {time_ms:.3f}ms (full load)")
        
        # Large Batch (10K)
        def seq_access_large_op():
            df_load = pd.read_parquet(path)
            _ = df_load.iloc[:10000].values
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (10K)", seq_access_large_op, number=3, repeat=3)
        if time_ms:
            self.results.add(format_name, "Sequential Access (10K)", time_ms, note="full load")
            print(f"  ✓ Sequential Access (10K): {time_ms:.3f}ms (full load)")
    
    # ==================== Arrow/Feather 测试 ====================
    
    def test_arrow(self, data: np.ndarray):
        """测试 Arrow/Feather 格式"""
        if not ARROW_AVAILABLE:
            print("\nArrow not available, skipping...")
            return
        
        format_name = "Arrow/Feather"
        print(f"\nTesting {format_name}...")
        
        path = self.temp_dir / "data.feather"
        
        # 转换为 DataFrame
        df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(data.shape[1])])
        
        # Save
        def save_op():
            feather.write_feather(df, path, compression='uncompressed')
        
        time_ms = self.benchmark_operation(format_name, "Save", save_op, number=1, repeat=3)
        if time_ms:
            file_size = self.get_dir_size_mb(path)
            self.results.add(format_name, "Save", time_ms, file_size)
            print(f"  ✓ Save: {time_ms:.3f}ms ({file_size:.2f}MB)")
        
        # Full Load
        def load_op():
            _ = feather.read_feather(path).values
        
        time_ms = self.benchmark_operation(format_name, "Full Load", load_op, number=5, repeat=3)
        if time_ms:
            self.results.add(format_name, "Full Load", time_ms)
            print(f"  ✓ Full Load: {time_ms:.3f}ms")
        
        # Arrow/Feather 支持部分列读取，但不支持行级lazy load
        self.results.add(format_name, "Lazy Load", None, note="N/A (columnar)")
        print(f"  Lazy Load: N/A (columnar format)")
        
        # Replace (需要重写)
        replace_data = np.random.rand(100, data.shape[1]).astype(data.dtype)
        
        def replace_op():
            df_load = feather.read_feather(path)
            df_load.iloc[:100] = replace_data
            feather.write_feather(df_load, path, compression='uncompressed')
        
        time_ms = self.benchmark_operation(format_name, "Replace 100", replace_op, number=1, repeat=3)
        if time_ms:
            self.results.add(format_name, "Replace 100", time_ms, note="full rewrite")
            print(f"  ✓ Replace 100: {time_ms:.3f}ms (full rewrite)")
        
        # Append
        append_data = np.random.rand(100, data.shape[1]).astype(data.dtype)
        df_append = pd.DataFrame(append_data, columns=df.columns)
        
        def append_op():
            df_load = feather.read_feather(path)
            df_new = pd.concat([df_load, df_append], ignore_index=True)
            feather.write_feather(df_new, path, compression='uncompressed')
        
        time_ms = self.benchmark_operation(format_name, "Append 100", append_op, number=1, repeat=3)
        if time_ms:
            self.results.add(format_name, "Append 100", time_ms, note="full rewrite")
            print(f"  ✓ Append 100: {time_ms:.3f}ms (full rewrite)")
        
        # Random Access Tests
        # Small Batch (100)
        random_indices_small = np.random.randint(0, min(len(data), 10000), 100)
        
        def random_access_small_op():
            df_load = feather.read_feather(path)
            _ = df_load.iloc[random_indices_small].values
        
        time_ms = self.benchmark_operation(format_name, "Random Access (100)", random_access_small_op, number=10, repeat=3)
        if time_ms:
            self.results.add(format_name, "Random Access (100)", time_ms, note="full load")
            print(f"  ✓ Random Access (100): {time_ms:.3f}ms (full load)")
        
        # Medium Batch (1K)
        random_indices_medium = np.random.randint(0, min(len(data), 10000), 1000)
        
        def random_access_medium_op():
            df_load = feather.read_feather(path)
            _ = df_load.iloc[random_indices_medium].values
        
        time_ms = self.benchmark_operation(format_name, "Random Access (1K)", random_access_medium_op, number=5, repeat=3)
        if time_ms:
            self.results.add(format_name, "Random Access (1K)", time_ms, note="full load")
            print(f"  ✓ Random Access (1K): {time_ms:.3f}ms (full load)")
        
        # Large Batch (10K)
        random_indices_large = np.random.randint(0, len(data), 10000)
        
        def random_access_large_op():
            df_load = feather.read_feather(path)
            _ = df_load.iloc[random_indices_large].values
        
        time_ms = self.benchmark_operation(format_name, "Random Access (10K)", random_access_large_op, number=3, repeat=3)
        if time_ms:
            self.results.add(format_name, "Random Access (10K)", time_ms, note="full load")
            print(f"  ✓ Random Access (10K): {time_ms:.3f}ms (full load)")
        
        # Sequential Access Tests
        # Small Batch (100)
        def seq_access_small_op():
            df_load = feather.read_feather(path)
            _ = df_load.iloc[:100].values
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (100)", seq_access_small_op, number=10, repeat=3)
        if time_ms:
            self.results.add(format_name, "Sequential Access (100)", time_ms, note="full load")
            print(f"  ✓ Sequential Access (100): {time_ms:.3f}ms (full load)")
        
        # Medium Batch (1K)
        def seq_access_medium_op():
            df_load = feather.read_feather(path)
            _ = df_load.iloc[:1000].values
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (1K)", seq_access_medium_op, number=5, repeat=3)
        if time_ms:
            self.results.add(format_name, "Sequential Access (1K)", time_ms, note="full load")
            print(f"  ✓ Sequential Access (1K): {time_ms:.3f}ms (full load)")
        
        # Large Batch (10K)
        def seq_access_large_op():
            df_load = feather.read_feather(path)
            _ = df_load.iloc[:10000].values
        
        time_ms = self.benchmark_operation(format_name, "Sequential Access (10K)", seq_access_large_op, number=3, repeat=3)
        if time_ms:
            self.results.add(format_name, "Sequential Access (10K)", time_ms, note="full load")
            print(f"  ✓ Sequential Access (10K): {time_ms:.3f}ms (full load)")


def print_summary_table(results: BenchmarkResult, operations: List[str], title: str):
    """打印汇总表格"""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")
    
    for op in operations:
        df = results.get_summary(op)
        if not df.empty:
            print(f"\n{op}:")
            print(df.to_string(index=False))


def main():
    """主测试函数"""
    print("="*80)
    print("NumPack Comprehensive Format Benchmark".center(80))
    print("="*80)
    
    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp(prefix="numpack_benchmark_"))
    print(f"\nTemp directory: {temp_dir}")
    
    try:
        # 测试数据集
        datasets = [
            {
                'name': 'Large Dataset (1M × 10, Float32, ~38.1MB)',
                'data': np.random.rand(1_000_000, 10).astype(np.float32),
                'array_name': 'data_large'
            },
            {
                'name': 'Medium Dataset (100K × 10, Float32, ~3.8MB)',
                'data': np.random.rand(100_000, 10).astype(np.float32),
                'array_name': 'data_medium'
            }
        ]
        
        all_results = {}
        
        for dataset_info in datasets:
            print(f"\n{'='*80}")
            print(f"Testing: {dataset_info['name']}")
            print(f"{'='*80}")
            
            data = dataset_info['data']
            array_name = dataset_info['array_name']
            
            benchmark = FormatBenchmark(temp_dir)
            
            # 运行所有格式测试
            benchmark.test_numpack(data, array_name)
            benchmark.test_npy(data)
            benchmark.test_npz(data, array_name)
            benchmark.test_zarr(data, array_name)
            benchmark.test_hdf5(data, array_name)
            benchmark.test_parquet(data)
            benchmark.test_arrow(data)
            
            all_results[dataset_info['name']] = benchmark.results
        
        # 打印汇总结果
        print("\n\n" + "="*80)
        print("BENCHMARK SUMMARY".center(80))
        print("="*80)
        
        operations = [
            "Save",
            "Full Load",
            "Lazy Load",
            "GetItem[0]",
            "GetItem[:100]",
            "Replace 100",
            "Append 100",
            "Random Access (100)",
            "Random Access (1K)",
            "Random Access (10K)",
            "Sequential Access (100)",
            "Sequential Access (1K)",
            "Sequential Access (10K)",
            "Normal Mode (100x)",
            "Batch Mode (100x)",
            "Writable Batch Mode (100x)"
        ]
        
        for dataset_name, results in all_results.items():
            print_summary_table(results, operations, dataset_name)
        
        # 文件大小对比
        print(f"\n{'='*80}")
        print("FILE SIZE COMPARISON".center(80))
        print(f"{'='*80}\n")
        
        for dataset_name, results in all_results.items():
            print(f"\n{dataset_name}:")
            size_data = []
            for format_name, ops in results.results.items():
                if 'Save' in ops and ops['Save'].get('file_size_mb'):
                    size_data.append({
                        'Format': format_name,
                        'Size (MB)': f"{ops['Save']['file_size_mb']:.2f}"
                    })
            
            if size_data:
                df = pd.DataFrame(size_data)
                print(df.to_string(index=False))
        
    finally:
        # 清理临时目录
        print(f"\n\nCleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n" + "="*80)
    print("Benchmark Complete!".center(80))
    print("="*80)


if __name__ == "__main__":
    main()

