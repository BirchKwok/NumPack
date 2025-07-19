#!/usr/bin/env python3
"""
NumPack conditional build script

Choose appropriate build method based on platform and environment variables:
- Windows platform: Use pure Python build (setuptools)
- Unix/Linux platform: Use Rust + Python build (maturin)
- Environment variable NUMPACK_PYTHON_ONLY=1: Force pure Python build
"""

import os
import sys
import platform
import shutil
import subprocess
from pathlib import Path


def is_windows():
    """Detect if running on Windows platform"""
    return platform.system().lower() == 'windows'


def should_use_python_only():
    """Decide whether to use pure Python build"""
    # Check environment variable
    if os.environ.get('NUMPACK_PYTHON_ONLY', '').lower() in ['1', 'true', 'yes']:
        return True
    
    # Windows platform defaults to pure Python
    if is_windows():
        return True
    
    return False


def backup_original_config():
    """Backup original configuration file"""
    if Path('pyproject.toml').exists():
        shutil.copy('pyproject.toml', 'pyproject.toml.backup')
        print("Backed up original pyproject.toml")


def restore_original_config():
    """Restore original configuration file"""
    if Path('pyproject.toml.backup').exists():
        shutil.copy('pyproject.toml.backup', 'pyproject.toml')
        Path('pyproject.toml.backup').unlink()
        print("Restored original pyproject.toml")


def setup_python_only_build():
    """Setup pure Python build"""
    print("Setting up pure Python build mode...")
    
    # Backup original configuration
    backup_original_config()
    
    # Use Windows-specific configuration
    if Path('pyproject.toml.windows').exists():
        shutil.copy('pyproject.toml.windows', 'pyproject.toml')
        print("Switched to pure Python build configuration")
    else:
        print("Error: pyproject.toml.windows file not found")
        return False
    
    return True


def setup_rust_build():
    """Setup Rust + Python build"""
    print("Setting up Rust + Python build mode...")
    
    # Use original configuration file (contains maturin)
    if Path('pyproject.toml.backup').exists():
        restore_original_config()
    
    return True


def run_build(build_args=None):
    """执行构建"""
    build_args = build_args or []
    
    if should_use_python_only():
        print(f"Executing pure Python build (Platform: {platform.system()})")
        
        if not setup_python_only_build():
            return False
        
        try:
            # Use standard Python build tools
            cmd = [sys.executable, '-m', 'build'] + build_args
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            print("Pure Python build successful")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Pure Python build failed: {e}")
            return False
        except FileNotFoundError:
            print("Error: 'build' module not found, please install: pip install build")
            return False
        finally:
            # Restore original configuration
            restore_original_config()
    
    else:
        print(f"Executing Rust + Python build (Platform: {platform.system()})")
        
        setup_rust_build()
        
        try:
            # Use maturin build
            cmd = ['maturin', 'build', '--release'] + build_args
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            print("Rust + Python build successful")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Rust build failed: {e}")
            return False
        except FileNotFoundError:
            print("Error: 'maturin' not found, please install: pip install maturin")
            return False


def run_develop():
    """Execute development mode installation"""
    if should_use_python_only():
        print(f"Executing pure Python development install (Platform: {platform.system()})")
        
        if not setup_python_only_build():
            return False
        
        try:
            # Use pip editable install
            cmd = [sys.executable, '-m', 'pip', 'install', '-e', '.']
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            print("Pure Python development install successful")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Pure Python development install failed: {e}")
            return False
        finally:
            # Restore original configuration
            restore_original_config()
    
    else:
        print(f"Executing Rust + Python development install (Platform: {platform.system()})")
        
        try:
            # Use maturin develop
            cmd = ['maturin', 'develop']
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            print("Rust + Python development install successful")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Rust development install failed: {e}")
            return False
        except FileNotFoundError:
            print("Error: 'maturin' not found, please install: pip install maturin")
            return False


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NumPack conditional build script")
    parser.add_argument('command', choices=['build', 'develop', 'info'], 
                        help='Command to execute')
    parser.add_argument('--python-only', action='store_true',
                        help='Force pure Python build')
    parser.add_argument('--out', help='Output directory (for build only)')
    
    args = parser.parse_args()
    
    # 设置环境变量
    if args.python_only:
        os.environ['NUMPACK_PYTHON_ONLY'] = '1'
    
    print(f"NumPack Build Script")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")
    print(f"Build mode: {'Pure Python' if should_use_python_only() else 'Rust + Python'}")
    print("-" * 50)
    
    if args.command == 'info':
        print(f"Current configuration:")
        print(f"  - Platform: {platform.system()}")
        print(f"  - Use Pure Python: {should_use_python_only()}")
        print(f"  - NUMPACK_PYTHON_ONLY: {os.environ.get('NUMPACK_PYTHON_ONLY', 'unset')}")
        return
    
    elif args.command == 'build':
        build_args = []
        if args.out:
            build_args.extend(['--out', args.out])
        
        success = run_build(build_args)
        sys.exit(0 if success else 1)
    
    elif args.command == 'develop':
        success = run_develop()
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main() 