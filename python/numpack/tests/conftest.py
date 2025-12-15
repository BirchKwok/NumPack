"""
pytest configuration file for Windows platform resource cleanup
"""
import pytest
import os
import gc
import time
import numpy as np

# 统一的NumPack支持的所有数据类型定义
ALL_DTYPES = [
    (np.bool_, [[True, False], [False, True]]),
    (np.uint8, [[0, 255], [128, 64]]),
    (np.uint16, [[0, 65535], [32768, 16384]]),
    (np.uint32, [[0, 4294967295], [2147483648, 1073741824]]),
    (np.uint64, [[0, 18446744073709551615], [9223372036854775808, 4611686018427387904]]),
    (np.int8, [[-128, 127], [0, -64]]),
    (np.int16, [[-32768, 32767], [0, -16384]]),
    (np.int32, [[-2147483648, 2147483647], [0, -1073741824]]),
    (np.int64, [[-9223372036854775808, 9223372036854775807], [0, -4611686018427387904]]),
    (np.float16, [[-1.0, 1.0], [0.0, 0.5]]),
    (np.float32, [[-1.0, 1.0], [0.0, 0.5]]),
    (np.float64, [[-1.0, 1.0], [0.0, 0.5]]),
    (np.complex64, [[1+2j, 3+4j], [0+0j, -1-2j]]),
    (np.complex128, [[1+2j, 3+4j], [0+0j, -1-2j]])
]

# 统一的数组维度定义
ARRAY_DIMS = [
    (1, (100,)),                           # 1 dimension
    (2, (50, 40)),                         # 2 dimension
    (3, (30, 20, 10)),                     # 3 dimension
    (4, (20, 15, 10, 5)),                  # 4 dimension
    (5, (10, 8, 6, 4, 2))                  # 5 dimension
]

# 辅助函数：创建测试数组
def create_test_array(dtype, shape):
    """创建测试数组的辅助函数"""
    if dtype == np.bool_:
        return np.random.choice([True, False], size=shape).astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.random.randint(info.min // 2, info.max // 2, size=shape, dtype=dtype)
    elif np.issubdtype(dtype, np.complexfloating):
        # Generate complex numbers with random real and imaginary parts
        real_part = np.random.rand(*shape) * 10 - 5  # random values between -5 and 5
        imag_part = np.random.rand(*shape) * 10 - 5  # random values between -5 and 5
        return (real_part + 1j * imag_part).astype(dtype)
    else:  # floating point
        return np.random.rand(*shape).astype(dtype)


def pytest_runtest_teardown(item, nextitem):
    """Cleanup after each test"""
    if os.name == 'nt':
        # 优化的Windows平台清理 - 减少延迟但保持功能
        try:
            from numpack import force_cleanup_windows_handles
            # 只执行一次清理以减少延迟
            force_cleanup_windows_handles()
        except ImportError:
            pass
        
        # 减少垃圾回收次数和等待时间
        for _ in range(2):  # 从5次减少到2次
            gc.collect()
            time.sleep(0.002)  # 从10ms减少到2ms
        
        # 大幅减少额外等待时间
        time.sleep(0.005)  # 从100ms减少到5ms
    else:
        # Basic cleanup for non-Windows platforms
        gc.collect()


def pytest_sessionfinish(session, exitstatus):
    """Final cleanup after entire test session"""
    if os.name == 'nt':
        # 优化的最终清理 - 保持功能但减少延迟
        try:
            from numpack import force_cleanup_windows_handles
            # 减少清理次数
            for _ in range(2):  # 从3次减少到2次
                force_cleanup_windows_handles()
                time.sleep(0.01)  # 从50ms减少到10ms
        except ImportError:
            pass
        
        # 减少最终垃圾回收次数
        for _ in range(3):  # 从10次减少到3次
            gc.collect()
            time.sleep(0.002)  # 从10ms减少到2ms
        
        # 减少最终等待时间
        time.sleep(0.05)  # 从200ms减少到50ms
    else:
        # Basic cleanup for non-Windows platforms
        gc.collect() 