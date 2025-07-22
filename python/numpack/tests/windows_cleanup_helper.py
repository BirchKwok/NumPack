"""
Windows 平台专用的文件句柄清理工具
"""
import os
import gc
import time
import tempfile
import shutil
from pathlib import Path


def force_release_file_handles(file_path_or_dir):
    """强制释放指定文件或目录的文件句柄"""
    if os.name != 'nt':
        return  # 仅Windows平台
    
    try:
        from numpack import force_cleanup_windows_handles
        force_cleanup_windows_handles()
    except ImportError:
        pass
    
    # 优化的垃圾回收
    for _ in range(2):  # 从5次减少到2次
        gc.collect()
        time.sleep(0.002)  # 从10ms减少到2ms
    
    # 减少额外等待
    time.sleep(0.01)  # 从100ms减少到10ms


def safe_rmtree(path):
    """Windows平台安全删除目录"""
    if os.name != 'nt':
        # 非Windows平台直接删除
        if os.path.exists(path):
            shutil.rmtree(path)
        return
    
    # Windows平台强化清理
    if not os.path.exists(path):
        return
    
    # 多次尝试删除
    for attempt in range(3):
        try:
            # 先释放文件句柄
            force_release_file_handles(path)
            
            # 尝试删除
            shutil.rmtree(path)
            break
        except PermissionError:
            if attempt < 2:  # 不是最后一次尝试
                # 更激进的清理
                try:
                    from numpack import force_cleanup_windows_handles
                    force_cleanup_windows_handles()
                except ImportError:
                    pass
                
                # 优化的强制垃圾回收
                for _ in range(3):  # 从10次减少到3次
                    gc.collect()
                    time.sleep(0.002)  # 从5ms减少到2ms
                
                # 减少等待时间
                time.sleep(0.02)  # 从200ms减少到20ms
            else:
                # 最后一次尝试失败，记录错误但不抛出异常
                print(f"Warning: Failed to delete {path} after 3 attempts")


def cleanup_lazy_arrays(*lazy_arrays):
    """清理LazyArray对象"""
    for lazy_arr in lazy_arrays:
        if lazy_arr is not None:
            try:
                del lazy_arr
            except:
                pass
    
    if os.name == 'nt':
        force_release_file_handles(None)


def cleanup_test_environment():
    """清理整个测试环境"""
    if os.name != 'nt':
        gc.collect()
        return
    
    # Windows平台强化清理
    try:
        from numpack import force_cleanup_windows_handles
        force_cleanup_windows_handles()
    except ImportError:
        pass
    
    # 优化的强制垃圾回收
    for _ in range(3):  # 从8次减少到3次
        gc.collect()
        time.sleep(0.002)  # 从5ms减少到2ms
    
    # 减少最终等待
    time.sleep(0.01)  # 从100ms减少到10ms 