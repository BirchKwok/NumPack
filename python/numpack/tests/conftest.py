"""
pytest configuration file for Windows platform resource cleanup
"""
import pytest
import os
import gc
import time
import numpy as np

# Unified definition of all dtypes supported by NumPack
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

# Unified definition of array dimensions
ARRAY_DIMS = [
    (1, (100,)),                           # 1 dimension
    (2, (50, 40)),                         # 2 dimension
    (3, (30, 20, 10)),                     # 3 dimension
    (4, (20, 15, 10, 5)),                  # 4 dimension
    (5, (10, 8, 6, 4, 2))                  # 5 dimension
]

# Helper: create a test array
def create_test_array(dtype, shape):
    """Helper function to create a test array."""
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
        # Optimized Windows cleanup: reduce latency while keeping functionality
        try:
            from numpack import force_cleanup_windows_handles
            # Run cleanup only once to reduce latency
            force_cleanup_windows_handles()
        except ImportError:
            pass
        
        # Reduce garbage collection cycles and sleep time
        for _ in range(2):  # reduced from 5 to 2
            gc.collect()
            time.sleep(0.002)  # reduced from 10ms to 2ms
        
        # Reduce extra wait time
        time.sleep(0.005)  # reduced from 100ms to 5ms
    else:
        # Basic cleanup for non-Windows platforms
        gc.collect()


def pytest_sessionfinish(session, exitstatus):
    """Final cleanup after entire test session"""
    if os.name == 'nt':
        # Optimized final cleanup: keep functionality while reducing latency
        try:
            from numpack import force_cleanup_windows_handles
            # Reduce cleanup iterations
            for _ in range(2):  # reduced from 3 to 2
                force_cleanup_windows_handles()
                time.sleep(0.01)  # reduced from 50ms to 10ms
        except ImportError:
            pass
        
        # Reduce final garbage collection cycles
        for _ in range(3):  # reduced from 10 to 3
            gc.collect()
            time.sleep(0.002)  # reduced from 10ms to 2ms
        
        # Reduce final wait time
        time.sleep(0.05)  # reduced from 200ms to 50ms
    else:
        # Basic cleanup for non-Windows platforms
        gc.collect()