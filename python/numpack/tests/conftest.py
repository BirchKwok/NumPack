"""
pytest configuration file for Windows platform resource cleanup
"""
import pytest
import os
import gc
import time


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