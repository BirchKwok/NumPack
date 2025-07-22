#!/usr/bin/env python3
"""
NumPack vs NumPy 全面性能基准测试

此脚本对比 NumPack 与 NumPy 在以下方面的性能：
1. IO 性能测试 (.npy, .npz)
2. 懒加载性能测试 (mmap_mode, memmap)
3. 随机读取性能测试
4. 批量读取性能测试
5. 流读取性能测试
6. 内存使用对比
7. 文件大小对比

支持分别测试 Python 后端和 Rust 后端
"""

import os
import sys
import time
import tempfile
import shutil
import gc
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging

# 添加 numpack 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 强制使用特定后端进行测试
def set_backend(backend_type: str):
    """设置要使用的后端类型"""
    if backend_type == 'python':
        os.environ['NUMPACK_FORCE_PYTHON_BACKEND'] = '1'
    else:
        os.environ.pop('NUMPACK_FORCE_PYTHON_BACKEND', None)

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    operation: str
    backend: str
    time_seconds: float
    memory_mb: float = 0.0
    file_size_mb: float = 0.0
    throughput_mb_s: float = 0.0
    extra_info: Dict[str, Any] = field(default_factory=dict)

class ComprehensiveBenchmark:
    """全面性能基准测试类"""
    
    def __init__(self, output_file: str = "Benchmark.md"):
        self.results: List[BenchmarkResult] = []
        self.output_file = output_file
        self.temp_dir = None
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def setup(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp(prefix="numpack_benchmark_")
        self.logger.info(f"创建临时目录: {self.temp_dir}")
        
    def cleanup(self):
        """清理测试环境"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.logger.info(f"清理临时目录: {self.temp_dir}")
    
    @contextmanager
    def timer_and_memory(self, operation_name: str, backend: str):
        """计时和内存监控上下文管理器"""
        gc.collect()  # 强制垃圾回收
        
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            time_taken = end_time - start_time
            memory_used = max(0, memory_after - memory_before)
            
            result = BenchmarkResult(
                operation=operation_name,
                backend=backend,
                time_seconds=time_taken,
                memory_mb=memory_used
            )
            self.results.append(result)
            
            self.logger.info(f"{operation_name} ({backend}): {time_taken:.3f}s, 内存: {memory_used:.1f}MB")
    
    def generate_test_data(self, rows: int, cols: int, arrays_count: int = 1) -> Dict[str, np.ndarray]:
        """生成测试数据"""
        data = {}
        for i in range(arrays_count):
            data[f'array_{i}'] = np.random.randn(rows, cols).astype(np.float32)
        return data
    
    def get_file_size_mb(self, filepath: str) -> float:
        """获取文件大小（MB）"""
        if os.path.isdir(filepath):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(filepath):
                for filename in filenames:
                    total_size += os.path.getsize(os.path.join(dirpath, filename))
            return total_size / 1024 / 1024
        else:
            return os.path.getsize(filepath) / 1024 / 1024
    
    def test_io_performance(self, rows: int = 1000000, cols: int = 100):
        """IO 性能测试"""
        self.logger.info(f"开始 IO 性能测试 (数据大小: {rows}x{cols})")
        
        test_data = self.generate_test_data(rows, cols, 3)
        data_size_mb = sum(arr.nbytes for arr in test_data.values()) / 1024 / 1024
        
        for backend_name in ['python', 'rust']:
            try:
                set_backend(backend_name)
                from numpack import NumPack
                
                # NumPack 保存测试
                numpack_file = os.path.join(self.temp_dir, f'test_numpack_{backend_name}')
                with self.timer_and_memory(f"NumPack Save", backend_name):
                    npk = NumPack(numpack_file, drop_if_exists=True)
                    npk.save(test_data)
                
                # 记录文件大小
                numpack_size = self.get_file_size_mb(numpack_file)
                self.results[-1].file_size_mb = numpack_size
                self.results[-1].throughput_mb_s = data_size_mb / self.results[-1].time_seconds
                
                # NumPack 加载测试
                with self.timer_and_memory(f"NumPack Load", backend_name):
                    for key in test_data.keys():
                        loaded = npk.load(key)
                        
                self.results[-1].throughput_mb_s = data_size_mb / self.results[-1].time_seconds
                
                # NumPack 懒加载测试
                with self.timer_and_memory(f"NumPack Lazy Load", backend_name):
                    for key in test_data.keys():
                        lazy_loaded = npk.load(key, lazy=True)
                        
                self.results[-1].throughput_mb_s = data_size_mb / self.results[-1].time_seconds
                
            except Exception as e:
                self.logger.error(f"NumPack {backend_name} 后端测试失败: {e}")
                continue
        
        # NumPy .npy 测试
        npy_files = []
        with self.timer_and_memory("NumPy .npy Save", "numpy"):
            for key, array in test_data.items():
                npy_file = os.path.join(self.temp_dir, f'{key}.npy')
                np.save(npy_file, array)
                npy_files.append(npy_file)
        
        npy_total_size = sum(self.get_file_size_mb(f) for f in npy_files)
        self.results[-1].file_size_mb = npy_total_size
        self.results[-1].throughput_mb_s = data_size_mb / self.results[-1].time_seconds
        
        with self.timer_and_memory("NumPy .npy Load", "numpy"):
            for npy_file in npy_files:
                loaded = np.load(npy_file)
                
        self.results[-1].throughput_mb_s = data_size_mb / self.results[-1].time_seconds
        
        with self.timer_and_memory("NumPy .npy mmap Load", "numpy"):
            for npy_file in npy_files:
                mmap_loaded = np.load(npy_file, mmap_mode='r')
                
        self.results[-1].throughput_mb_s = data_size_mb / self.results[-1].time_seconds
        
        # NumPy .npz 测试
        npz_file = os.path.join(self.temp_dir, 'test_data.npz')
        with self.timer_and_memory("NumPy .npz Save", "numpy"):
            np.savez(npz_file, **test_data)
            
        npz_size = self.get_file_size_mb(npz_file)
        self.results[-1].file_size_mb = npz_size
        self.results[-1].throughput_mb_s = data_size_mb / self.results[-1].time_seconds
        
        with self.timer_and_memory("NumPy .npz Load", "numpy"):
            npz_data = np.load(npz_file)
            for key in test_data.keys():
                loaded = npz_data[key]
                
        self.results[-1].throughput_mb_s = data_size_mb / self.results[-1].time_seconds
        
        with self.timer_and_memory("NumPy .npz mmap Load", "numpy"):
            npz_mmap = np.load(npz_file, mmap_mode='r')
            for key in test_data.keys():
                mmap_loaded = npz_mmap[key]
                
        self.results[-1].throughput_mb_s = data_size_mb / self.results[-1].time_seconds
    
    def test_random_access_performance(self, rows: int = 1000000, cols: int = 100, access_count: int = 10000):
        """随机访问性能测试"""
        self.logger.info(f"开始随机访问性能测试 (访问 {access_count} 次)")
        
        test_data = self.generate_test_data(rows, cols, 1)
        test_array = test_data['array_0']
        
        # 生成随机索引
        random_indices = np.random.randint(0, rows, access_count)
        single_indices = random_indices[:1000]  # 单个索引访问
        batch_indices = random_indices  # 批量索引访问
        
        for backend_name in ['python', 'rust']:
            try:
                set_backend(backend_name)
                from numpack import NumPack
                
                # 准备数据
                numpack_file = os.path.join(self.temp_dir, f'random_access_{backend_name}')
                npk = NumPack(numpack_file, drop_if_exists=True)
                npk.save({'test_array': test_array})
                
                # 单个随机访问测试
                with self.timer_and_memory(f"NumPack Single Random Access", backend_name):
                    for idx in single_indices:
                        result = npk.getitem('test_array', [int(idx)])
                        
                # 批量随机访问测试
                with self.timer_and_memory(f"NumPack Batch Random Access", backend_name):
                    result = npk.getitem('test_array', batch_indices.tolist())
                    
            except Exception as e:
                self.logger.error(f"NumPack {backend_name} 随机访问测试失败: {e}")
                continue
        
        # NumPy 比较测试
        npy_file = os.path.join(self.temp_dir, 'random_access.npy')
        np.save(npy_file, test_array)
        
        # NumPy 内存中随机访问
        with self.timer_and_memory("NumPy In-Memory Random Access", "numpy"):
            for idx in single_indices:
                result = test_array[idx:idx+1]
                
        with self.timer_and_memory("NumPy In-Memory Batch Random Access", "numpy"):
            result = test_array[batch_indices]
            
        # NumPy mmap 随机访问
        mmap_array = np.load(npy_file, mmap_mode='r')
        with self.timer_and_memory("NumPy mmap Single Random Access", "numpy"):
            for idx in single_indices:
                result = mmap_array[idx:idx+1]
                
        with self.timer_and_memory("NumPy mmap Batch Random Access", "numpy"):
            result = mmap_array[batch_indices]
        
        # NumPy memmap 随机访问
        memmap_file = os.path.join(self.temp_dir, 'memmap_test.dat')
        memmap_array = np.memmap(memmap_file, dtype='float32', mode='w+', shape=(rows, cols))
        memmap_array[:] = test_array[:]
        del memmap_array  # 确保写入磁盘
        
        memmap_array = np.memmap(memmap_file, dtype='float32', mode='r', shape=(rows, cols))
        with self.timer_and_memory("NumPy memmap Random Access", "numpy"):
            for idx in single_indices:
                result = memmap_array[idx:idx+1]
                
        with self.timer_and_memory("NumPy memmap Batch Random Access", "numpy"):
            result = memmap_array[batch_indices]
    
    def test_streaming_performance(self, rows: int = 1000000, cols: int = 100, chunk_size: int = 10000):
        """流读取性能测试"""
        self.logger.info(f"开始流读取性能测试 (块大小: {chunk_size})")
        
        test_data = self.generate_test_data(rows, cols, 1)
        test_array = test_data['array_0']
        
        for backend_name in ['python', 'rust']:
            try:
                set_backend(backend_name)
                from numpack import NumPack
                
                # 准备数据
                numpack_file = os.path.join(self.temp_dir, f'streaming_{backend_name}')
                npk = NumPack(numpack_file, drop_if_exists=True)
                npk.save({'test_array': test_array})
                
                # NumPack 流读取测试
                with self.timer_and_memory(f"NumPack Stream Read", backend_name):
                    total_processed = 0
                    for chunk in npk.stream_load('test_array', chunk_size):
                        total_processed += len(chunk)
                        if total_processed >= rows:
                            break
                            
            except Exception as e:
                self.logger.error(f"NumPack {backend_name} 流读取测试失败: {e}")
                continue
        
        # NumPy 分块读取对比
        npy_file = os.path.join(self.temp_dir, 'streaming.npy')
        np.save(npy_file, test_array)
        
        mmap_array = np.load(npy_file, mmap_mode='r')
        with self.timer_and_memory("NumPy mmap Chunk Read", "numpy"):
            for i in range(0, rows, chunk_size):
                chunk = mmap_array[i:i+chunk_size]
    
    def test_bulk_operations(self, rows: int = 500000, cols: int = 100):
        """批量操作性能测试"""
        self.logger.info(f"开始批量操作性能测试")
        
        test_data = self.generate_test_data(rows, cols, 1)
        test_array = test_data['array_0']
        
        # 准备追加数据
        append_data = np.random.randn(rows // 4, cols).astype(np.float32)
        replace_data = np.random.randn(rows // 4, cols).astype(np.float32)
        replace_indices = list(range(rows // 4))
        
        for backend_name in ['python', 'rust']:
            try:
                set_backend(backend_name)
                from numpack import NumPack
                
                # 准备数据
                numpack_file = os.path.join(self.temp_dir, f'bulk_{backend_name}')
                npk = NumPack(numpack_file, drop_if_exists=True)
                npk.save({'test_array': test_array})
                
                # 追加操作测试
                with self.timer_and_memory(f"NumPack Append", backend_name):
                    npk.append({'test_array': append_data})
                
                # 替换操作测试
                with self.timer_and_memory(f"NumPack Replace", backend_name):
                    npk.replace({'test_array': replace_data}, replace_indices)
                
                # 删除操作测试
                delete_indices = list(range(1000))
                with self.timer_and_memory(f"NumPack Delete", backend_name):
                    npk.drop('test_array', delete_indices)
                    
            except Exception as e:
                self.logger.error(f"NumPack {backend_name} 批量操作测试失败: {e}")
                continue
        
        # NumPy 对比操作
        numpy_array = test_array.copy()
        
        with self.timer_and_memory("NumPy Append (vstack)", "numpy"):
            numpy_array = np.vstack([numpy_array, append_data])
            
        with self.timer_and_memory("NumPy Replace", "numpy"):
            numpy_array[:len(replace_indices)] = replace_data
            
        with self.timer_and_memory("NumPy Delete", "numpy"):
            mask = np.ones(len(numpy_array), dtype=bool)
            mask[:1000] = False
            numpy_array = numpy_array[mask]
    
    def test_memory_efficiency(self, rows: int = 1000000, cols: int = 100):
        """内存效率测试"""
        self.logger.info(f"开始内存效率测试")
        
        test_data = self.generate_test_data(rows, cols, 1)
        test_array = test_data['array_0']
        data_size_mb = test_array.nbytes / 1024 / 1024
        
        # 测试懒加载内存效率
        for backend_name in ['python', 'rust']:
            try:
                set_backend(backend_name)
                from numpack import NumPack
                
                numpack_file = os.path.join(self.temp_dir, f'memory_{backend_name}')
                npk = NumPack(numpack_file, drop_if_exists=True)
                npk.save({'test_array': test_array})
                
                # 懒加载内存使用
                gc.collect()
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024
                lazy_array = npk.load('test_array', lazy=True)
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                
                result = BenchmarkResult(
                    operation=f"NumPack Lazy Load Memory",
                    backend=backend_name,
                    time_seconds=0,
                    memory_mb=memory_after - memory_before,
                    extra_info={"data_size_mb": data_size_mb, "memory_ratio": (memory_after - memory_before) / data_size_mb}
                )
                self.results.append(result)
                
            except Exception as e:
                self.logger.error(f"NumPack {backend_name} 内存效率测试失败: {e}")
                continue
        
        # NumPy 内存使用对比
        npy_file = os.path.join(self.temp_dir, 'memory_test.npy')
        np.save(npy_file, test_array)
        
        # 完整加载内存使用
        gc.collect()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        loaded_array = np.load(npy_file)
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        
        result = BenchmarkResult(
            operation="NumPy Full Load Memory",
            backend="numpy",
            time_seconds=0,
            memory_mb=memory_after - memory_before,
            extra_info={"data_size_mb": data_size_mb, "memory_ratio": (memory_after - memory_before) / data_size_mb}
        )
        self.results.append(result)
        
        # mmap 内存使用
        del loaded_array
        gc.collect()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        mmap_array = np.load(npy_file, mmap_mode='r')
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        
        result = BenchmarkResult(
            operation="NumPy mmap Load Memory",
            backend="numpy",
            time_seconds=0,
            memory_mb=memory_after - memory_before,
            extra_info={"data_size_mb": data_size_mb, "memory_ratio": (memory_after - memory_before) / data_size_mb}
        )
        self.results.append(result)
    
    def test_non_contiguous_access(self, rows: int = 1000000, cols: int = 100, access_count: int = 5000):
        """非连续访问性能测试"""
        self.logger.info(f"开始非连续访问性能测试")
        
        test_data = self.generate_test_data(rows, cols, 1)
        test_array = test_data['array_0']
        
        # 生成不规则的非连续索引模式
        stride_indices = list(range(0, rows, 200))[:access_count]  # 每隔200行
        reverse_indices = list(range(rows-1, 0, -100))[:access_count]  # 倒序
        fibonacci_indices = []
        a, b = 0, 1
        while len(fibonacci_indices) < access_count and b < rows:
            fibonacci_indices.append(b)
            a, b = b, a + b
        
        for backend_name in ['python', 'rust']:
            try:
                set_backend(backend_name)
                from numpack import NumPack
                
                numpack_file = os.path.join(self.temp_dir, f'non_contiguous_{backend_name}')
                npk = NumPack(numpack_file, drop_if_exists=True)
                npk.save({'test_array': test_array})
                
                # 步长访问测试
                with self.timer_and_memory(f"NumPack Stride Access", backend_name):
                    result = npk.getitem('test_array', stride_indices)
                
                # 倒序访问测试
                with self.timer_and_memory(f"NumPack Reverse Access", backend_name):
                    result = npk.getitem('test_array', reverse_indices)
                
                # 斐波那契索引访问测试
                with self.timer_and_memory(f"NumPack Fibonacci Access", backend_name):
                    result = npk.getitem('test_array', fibonacci_indices)
                    
            except Exception as e:
                self.logger.error(f"NumPack {backend_name} 非连续访问测试失败: {e}")
                continue
        
        # NumPy 对比测试
        npy_file = os.path.join(self.temp_dir, 'non_contiguous.npy')
        np.save(npy_file, test_array)
        mmap_array = np.load(npy_file, mmap_mode='r')
        
        with self.timer_and_memory("NumPy mmap Stride Access", "numpy"):
            result = mmap_array[stride_indices]
            
        with self.timer_and_memory("NumPy mmap Reverse Access", "numpy"):
            result = mmap_array[reverse_indices]
            
        with self.timer_and_memory("NumPy mmap Fibonacci Access", "numpy"):
            result = mmap_array[fibonacci_indices]
    
    def test_different_dtypes(self, rows: int = 500000, cols: int = 100):
        """不同数据类型性能测试"""
        self.logger.info(f"开始不同数据类型性能测试")
        
        dtypes = [
            ('float32', np.float32),
            ('float64', np.float64),
            ('int32', np.int32),
            ('int64', np.int64),
            ('uint8', np.uint8)
        ]
        
        for dtype_name, dtype in dtypes:
            # 生成指定类型的测试数据
            if dtype == np.uint8:
                test_array = np.random.randint(0, 256, (rows, cols), dtype=dtype)
            elif 'int' in dtype_name:
                test_array = np.random.randint(-1000, 1000, (rows, cols), dtype=dtype)
            else:
                test_array = np.random.randn(rows, cols).astype(dtype)
            
            data_size_mb = test_array.nbytes / 1024 / 1024
            
            for backend_name in ['python', 'rust']:
                try:
                    set_backend(backend_name)
                    from numpack import NumPack
                    
                    numpack_file = os.path.join(self.temp_dir, f'dtype_{dtype_name}_{backend_name}')
                    npk = NumPack(numpack_file, drop_if_exists=True)
                    
                    # 保存测试
                    with self.timer_and_memory(f"NumPack Save {dtype_name}", backend_name):
                        npk.save({'test_array': test_array})
                    
                    file_size = self.get_file_size_mb(numpack_file)
                    self.results[-1].file_size_mb = file_size
                    self.results[-1].throughput_mb_s = data_size_mb / self.results[-1].time_seconds
                    
                    # 加载测试
                    with self.timer_and_memory(f"NumPack Load {dtype_name}", backend_name):
                        loaded = npk.load('test_array')
                    
                    self.results[-1].throughput_mb_s = data_size_mb / self.results[-1].time_seconds
                    
                except Exception as e:
                    self.logger.error(f"NumPack {backend_name} {dtype_name} 测试失败: {e}")
                    continue
            
            # NumPy 对比
            npy_file = os.path.join(self.temp_dir, f'dtype_{dtype_name}.npy')
            
            with self.timer_and_memory(f"NumPy Save {dtype_name}", "numpy"):
                np.save(npy_file, test_array)
            
            file_size = self.get_file_size_mb(npy_file)
            self.results[-1].file_size_mb = file_size
            self.results[-1].throughput_mb_s = data_size_mb / self.results[-1].time_seconds
            
            with self.timer_and_memory(f"NumPy Load {dtype_name}", "numpy"):
                loaded = np.load(npy_file)
            
            self.results[-1].throughput_mb_s = data_size_mb / self.results[-1].time_seconds
    
    def test_large_matrix_operations(self, rows: int = 500000, cols: int = 128):
        """大型矩阵运算性能测试"""
        self.logger.info(f"开始大型矩阵运算性能测试")
        
        test_data = self.generate_test_data(rows, cols, 1)
        test_array = test_data['array_0']
        query_vector = np.random.randn(1, cols).astype(np.float32)
        
        for backend_name in ['python', 'rust']:
            try:
                set_backend(backend_name)
                from numpack import NumPack
                
                numpack_file = os.path.join(self.temp_dir, f'matrix_{backend_name}')
                npk = NumPack(numpack_file, drop_if_exists=True)
                npk.save({'test_array': test_array})
                
                # 懒加载矩阵运算
                lazy_array = npk.load('test_array', lazy=True)
                with self.timer_and_memory(f"NumPack Lazy Matrix Dot", backend_name):
                    result = np.dot(query_vector, lazy_array.T)
                
                # 点积运算
                with self.timer_and_memory(f"NumPack Lazy Matrix Inner", backend_name):
                    result = np.inner(query_vector, lazy_array)
                    
            except Exception as e:
                self.logger.error(f"NumPack {backend_name} 矩阵运算测试失败: {e}")
                continue
        
        # NumPy 对比
        npy_file = os.path.join(self.temp_dir, 'matrix_ops.npy')
        np.save(npy_file, test_array)
        
        # 内存中运算
        with self.timer_and_memory("NumPy In-Memory Matrix Dot", "numpy"):
            result = np.dot(query_vector, test_array.T)
            
        with self.timer_and_memory("NumPy In-Memory Matrix Inner", "numpy"):
            result = np.inner(query_vector, test_array)
        
        # mmap 运算
        mmap_array = np.load(npy_file, mmap_mode='r')
        with self.timer_and_memory("NumPy mmap Matrix Dot", "numpy"):
            result = np.dot(query_vector, mmap_array.T)
            
        with self.timer_and_memory("NumPy mmap Matrix Inner", "numpy"):
            result = np.inner(query_vector, mmap_array)
    
    def test_compression_efficiency(self, rows: int = 500000, cols: int = 100):
        """压缩效率测试"""
        self.logger.info(f"开始压缩效率测试")
        
        # 测试不同压缩特性的数据
        test_cases = [
            ("Random Data", np.random.randn(rows, cols).astype(np.float32)),
            ("Sparse Data", np.zeros((rows, cols), dtype=np.float32)),
            ("Repeated Pattern", np.tile(np.arange(cols, dtype=np.float32), (rows, 1))),
        ]
        
        # 稀疏数据：只有5%的非零值
        sparse_data = test_cases[1][1].copy()
        non_zero_indices = np.random.choice(rows * cols, size=int(0.05 * rows * cols), replace=False)
        sparse_data.flat[non_zero_indices] = np.random.randn(len(non_zero_indices))
        test_cases[1] = ("Sparse Data", sparse_data)
        
        for test_name, test_array in test_cases:
            original_size_mb = test_array.nbytes / 1024 / 1024
            
            for backend_name in ['python', 'rust']:
                try:
                    set_backend(backend_name)
                    from numpack import NumPack
                    
                    numpack_file = os.path.join(self.temp_dir, f'compression_{test_name.replace(" ", "_")}_{backend_name}')
                    npk = NumPack(numpack_file, drop_if_exists=True)
                    npk.save({'test_array': test_array})
                    
                    compressed_size_mb = self.get_file_size_mb(numpack_file)
                    compression_ratio = original_size_mb / compressed_size_mb if compressed_size_mb > 0 else 0
                    
                    result = BenchmarkResult(
                        operation=f"NumPack Compression {test_name}",
                        backend=backend_name,
                        time_seconds=0,
                        file_size_mb=compressed_size_mb,
                        extra_info={
                            "original_size_mb": original_size_mb,
                            "compression_ratio": compression_ratio
                        }
                    )
                    self.results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"NumPack {backend_name} 压缩测试失败: {e}")
                    continue
            
            # NumPy 对比
            npy_file = os.path.join(self.temp_dir, f'compression_{test_name.replace(" ", "_")}.npy')
            np.save(npy_file, test_array)
            npy_size_mb = self.get_file_size_mb(npy_file)
            npy_ratio = original_size_mb / npy_size_mb if npy_size_mb > 0 else 0
            
            npz_file = os.path.join(self.temp_dir, f'compression_{test_name.replace(" ", "_")}.npz')
            np.savez_compressed(npz_file, test_array=test_array)
            npz_size_mb = self.get_file_size_mb(npz_file)
            npz_ratio = original_size_mb / npz_size_mb if npz_size_mb > 0 else 0
            
            result = BenchmarkResult(
                operation=f"NumPy .npy Compression {test_name}",
                backend="numpy",
                time_seconds=0,
                file_size_mb=npy_size_mb,
                extra_info={
                    "original_size_mb": original_size_mb,
                    "compression_ratio": npy_ratio
                }
            )
            self.results.append(result)
            
            result = BenchmarkResult(
                operation=f"NumPy .npz Compression {test_name}",
                backend="numpy",
                time_seconds=0,
                file_size_mb=npz_size_mb,
                extra_info={
                    "original_size_mb": original_size_mb,
                    "compression_ratio": npz_ratio
                }
            )
            self.results.append(result)
    
    def run_all_tests(self):
        """运行所有测试"""
        self.logger.info("开始全面性能基准测试")
        
        try:
            self.setup()
            
            # 运行各项测试
            self.test_io_performance()
            self.test_random_access_performance()
            self.test_streaming_performance()
            self.test_bulk_operations()
            self.test_memory_efficiency()
            self.test_non_contiguous_access()
            self.test_different_dtypes()
            self.test_large_matrix_operations()
            self.test_compression_efficiency()
            
            # 生成报告
            self.generate_report()
            
        finally:
            self.cleanup()
    
    def generate_report(self):
        """生成 Markdown 格式的性能报告"""
        self.logger.info(f"生成性能报告: {self.output_file}")
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("# NumPack vs NumPy 性能基准测试报告\n\n")
            f.write(f"测试环境信息:\n")
            f.write(f"- Python 版本: {sys.version}\n")
            f.write(f"- NumPy 版本: {np.__version__}\n")
            f.write(f"- 系统信息: {os.uname()}\n")
            f.write(f"- CPU 核心数: {psutil.cpu_count()}\n")
            f.write(f"- 总内存: {psutil.virtual_memory().total / 1024**3:.1f} GB\n\n")
            
            # 按测试类型分组结果
            test_groups = {}
            for result in self.results:
                category = result.operation.split()[0] + " " + result.operation.split()[1] if len(result.operation.split()) > 1 else result.operation
                if category not in test_groups:
                    test_groups[category] = []
                test_groups[category].append(result)
            
            for category, results in test_groups.items():
                f.write(f"## {category}\n\n")
                
                # 创建表格
                f.write("| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |\n")
                f.write("|------|------|----------|--------------|--------------|-------------|----------|\n")
                
                for result in sorted(results, key=lambda x: (x.operation, x.backend)):
                    extra_info = ", ".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" 
                                          for k, v in result.extra_info.items()])
                    
                    f.write(f"| {result.operation} | {result.backend} | "
                           f"{result.time_seconds:.3f} | {result.memory_mb:.1f} | "
                           f"{result.file_size_mb:.1f} | {result.throughput_mb_s:.1f} | "
                           f"{extra_info} |\n")
                f.write("\n")
            
            # 生成总结
            f.write("## 性能总结\n\n")
            
            # IO 性能总结
            f.write("### IO 性能总结\n")
            io_results = [r for r in self.results if ('Save' in r.operation or 'Load' in r.operation) and 'Compression' not in r.operation]
            if io_results:
                save_results = [r for r in io_results if 'Save' in r.operation]
                load_results = [r for r in io_results if 'Load' in r.operation]
                
                if save_results:
                    fastest_save = min(save_results, key=lambda x: x.time_seconds)
                    f.write(f"- 最快保存: {fastest_save.operation} ({fastest_save.backend}) - {fastest_save.time_seconds:.3f}秒\n")
                
                if load_results:
                    fastest_load = min(load_results, key=lambda x: x.time_seconds)
                    f.write(f"- 最快加载: {fastest_load.operation} ({fastest_load.backend}) - {fastest_load.time_seconds:.3f}秒\n")
                
                # 吞吐量总结
                if save_results:
                    highest_save_throughput = max(save_results, key=lambda x: x.throughput_mb_s)
                    f.write(f"- 最高保存吞吐量: {highest_save_throughput.operation} ({highest_save_throughput.backend}) - {highest_save_throughput.throughput_mb_s:.1f}MB/s\n")
                
                if load_results:
                    highest_load_throughput = max(load_results, key=lambda x: x.throughput_mb_s)
                    f.write(f"- 最高加载吞吐量: {highest_load_throughput.operation} ({highest_load_throughput.backend}) - {highest_load_throughput.throughput_mb_s:.1f}MB/s\n\n")
            
            # 随机访问性能总结
            f.write("### 随机访问性能总结\n")
            random_results = [r for r in self.results if 'Random Access' in r.operation or 'Access' in r.operation]
            if random_results:
                fastest_random = min(random_results, key=lambda x: x.time_seconds)
                f.write(f"- 最快随机访问: {fastest_random.operation} ({fastest_random.backend}) - {fastest_random.time_seconds:.3f}秒\n")
                
                # 非连续访问性能
                access_results = [r for r in random_results if any(pattern in r.operation for pattern in ['Stride', 'Reverse', 'Fibonacci'])]
                if access_results:
                    fastest_access = min(access_results, key=lambda x: x.time_seconds)
                    f.write(f"- 最快非连续访问: {fastest_access.operation} ({fastest_access.backend}) - {fastest_access.time_seconds:.3f}秒\n\n")
            
            # 内存效率总结
            f.write("### 内存效率总结\n")
            memory_results = [r for r in self.results if 'Memory' in r.operation]
            if memory_results:
                most_efficient = min(memory_results, key=lambda x: x.memory_mb)
                f.write(f"- 最节省内存: {most_efficient.operation} ({most_efficient.backend}) - {most_efficient.memory_mb:.1f}MB\n")
                
                # 内存比率总结
                for result in memory_results:
                    if 'memory_ratio' in result.extra_info:
                        f.write(f"  - {result.operation} ({result.backend}): 内存占用比 {result.extra_info['memory_ratio']:.2f}\n")
                f.write("\n")
            
            # 压缩效率总结
            f.write("### 压缩效率总结\n")
            compression_results = [r for r in self.results if 'Compression' in r.operation]
            if compression_results:
                best_compression = max(compression_results, key=lambda x: x.extra_info.get('compression_ratio', 0))
                f.write(f"- 最佳压缩比: {best_compression.operation} ({best_compression.backend}) - {best_compression.extra_info.get('compression_ratio', 0):.2f}x\n")
                
                # 按数据类型分组压缩结果
                data_types = set()
                for result in compression_results:
                    for pattern in ['Random Data', 'Sparse Data', 'Repeated Pattern']:
                        if pattern in result.operation:
                            data_types.add(pattern)
                
                for data_type in sorted(data_types):
                    type_results = [r for r in compression_results if data_type in r.operation]
                    if type_results:
                        best_for_type = max(type_results, key=lambda x: x.extra_info.get('compression_ratio', 0))
                        f.write(f"  - {data_type} 最佳压缩: {best_for_type.operation.replace(f'Compression {data_type}', '').strip()} ({best_for_type.backend}) - {best_for_type.extra_info.get('compression_ratio', 0):.2f}x\n")
                f.write("\n")
            
            # 矩阵运算性能总结
            f.write("### 矩阵运算性能总结\n")
            matrix_results = [r for r in self.results if 'Matrix' in r.operation]
            if matrix_results:
                fastest_matrix = min(matrix_results, key=lambda x: x.time_seconds)
                f.write(f"- 最快矩阵运算: {fastest_matrix.operation} ({fastest_matrix.backend}) - {fastest_matrix.time_seconds:.3f}秒\n\n")
            
            # 数据类型性能总结
            f.write("### 数据类型性能总结\n")
            dtype_results = [r for r in self.results if any(dtype in r.operation for dtype in ['float32', 'float64', 'int32', 'int64', 'uint8'])]
            if dtype_results:
                # 按操作类型分组
                save_dtype_results = [r for r in dtype_results if 'Save' in r.operation]
                load_dtype_results = [r for r in dtype_results if 'Load' in r.operation]
                
                if save_dtype_results:
                    fastest_save_dtype = min(save_dtype_results, key=lambda x: x.time_seconds)
                    f.write(f"- 最快数据类型保存: {fastest_save_dtype.operation} ({fastest_save_dtype.backend}) - {fastest_save_dtype.time_seconds:.3f}秒\n")
                    
                if load_dtype_results:
                    fastest_load_dtype = min(load_dtype_results, key=lambda x: x.time_seconds)
                    f.write(f"- 最快数据类型加载: {fastest_load_dtype.operation} ({fastest_load_dtype.backend}) - {fastest_load_dtype.time_seconds:.3f}秒\n\n")
            
            f.write("---\n")
            f.write(f"报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

def main():
    """主函数"""
    benchmark = ComprehensiveBenchmark()
    benchmark.run_all_tests()
    print(f"性能测试完成！报告已保存到 {benchmark.output_file}")

if __name__ == "__main__":
    main() 