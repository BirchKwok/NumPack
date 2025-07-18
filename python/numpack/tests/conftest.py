"""
pytest配置文件，为Windows平台提供特殊的资源清理
"""
import pytest
import os
import gc
import time


def pytest_runtest_teardown(item, nextitem):
    """在每个测试后执行清理"""
    if os.name == 'nt':
        # Windows平台的强化清理 - 针对文件句柄问题
        try:
            from numpack import force_cleanup_windows_handles
            # 执行两次清理，确保彻底释放
            force_cleanup_windows_handles()
            time.sleep(0.05)
            force_cleanup_windows_handles()
        except ImportError:
            pass
        
        # 强化垃圾回收
        for _ in range(5):  # 增加垃圾回收次数
            gc.collect()
            time.sleep(0.01)
        
        # 额外等待时间确保文件句柄释放
        time.sleep(0.1)
    else:
        # 非Windows平台的基本清理
        gc.collect()


def pytest_sessionfinish(session, exitstatus):
    """在整个测试会话结束后执行清理"""
    if os.name == 'nt':
        # Windows平台的最终清理 - 确保所有资源释放
        try:
            from numpack import force_cleanup_windows_handles
            # 多次执行最终清理
            for _ in range(3):
                force_cleanup_windows_handles()
                time.sleep(0.05)
        except ImportError:
            pass
        
        # 最终强制垃圾回收
        for _ in range(10):
            gc.collect()
            time.sleep(0.01)
        
        # 最终等待时间
        time.sleep(0.2)
    else:
        # 非Windows平台的基本清理
        gc.collect() 