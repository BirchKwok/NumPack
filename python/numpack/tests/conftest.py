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
        # Enhanced cleanup for Windows platform - addressing file handle issues
        try:
            from numpack import force_cleanup_windows_handles
            # Execute cleanup twice to ensure thorough release
            force_cleanup_windows_handles()
            time.sleep(0.05)
            force_cleanup_windows_handles()
        except ImportError:
            pass
        
        # Enhanced garbage collection
        for _ in range(5):  # Increase garbage collection iterations
            gc.collect()
            time.sleep(0.01)
        
        # Additional wait time to ensure file handle release
        time.sleep(0.1)
    else:
        # Basic cleanup for non-Windows platforms
        gc.collect()


def pytest_sessionfinish(session, exitstatus):
    """Final cleanup after entire test session"""
    if os.name == 'nt':
        # Final cleanup for Windows platform - ensure all resources released
        try:
            from numpack import force_cleanup_windows_handles
            # Execute final cleanup multiple times
            for _ in range(3):
                force_cleanup_windows_handles()
                time.sleep(0.05)
        except ImportError:
            pass
        
        # Final forced garbage collection
        for _ in range(10):
            gc.collect()
            time.sleep(0.01)
        
        # Final wait time
        time.sleep(0.2)
    else:
        # Basic cleanup for non-Windows platforms
        gc.collect() 