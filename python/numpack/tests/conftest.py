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
        # Windows平台的特殊清理
        try:
            from numpack import force_cleanup_windows_handles
            force_cleanup_windows_handles()
        except ImportError:
            pass
        
        # 多次垃圾回收
        for _ in range(3):
            gc.collect()
            time.sleep(0.01)
        
        # 额外等待时间
        time.sleep(0.05)
    else:
        # 非Windows平台的基本清理
        gc.collect()


def pytest_sessionfinish(session, exitstatus):
    """在整个测试会话结束后执行清理"""
    if os.name == 'nt':
        # Windows平台的最终清理
        try:
            from numpack import force_cleanup_windows_handles
            force_cleanup_windows_handles()
        except ImportError:
            pass
        
        # 强制垃圾回收
        for _ in range(5):
            gc.collect()
            time.sleep(0.02)
        
        # 最终等待
        time.sleep(0.2)
    else:
        # 非Windows平台的基本清理
        gc.collect() 