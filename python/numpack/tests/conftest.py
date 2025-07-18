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
        # Windows平台的特殊清理 - 简化以避免卡住
        try:
            from numpack import force_cleanup_windows_handles
            force_cleanup_windows_handles()
        except ImportError:
            pass
        
        # 减少垃圾回收次数，避免过度等待
        for _ in range(2):  # 从3减少到2
            gc.collect()
            time.sleep(0.005)  # 从0.01减少到0.005
        
        # 减少额外等待时间
        time.sleep(0.02)  # 从0.05减少到0.02
    else:
        # 非Windows平台的基本清理
        gc.collect()


def pytest_sessionfinish(session, exitstatus):
    """在整个测试会话结束后执行清理"""
    if os.name == 'nt':
        # Windows平台的最终清理 - 简化以确保快速退出
        try:
            from numpack import force_cleanup_windows_handles
            force_cleanup_windows_handles()
        except ImportError:
            pass
        
        # 减少垃圾回收次数和等待时间
        for _ in range(3):  # 从5减少到3
            gc.collect()
            time.sleep(0.01)  # 从0.02减少到0.01
        
        # 减少最终等待时间
        time.sleep(0.05)  # 从0.2减少到0.05
    else:
        # 非Windows平台的基本清理
        gc.collect() 