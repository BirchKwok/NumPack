#!/usr/bin/env python3
"""
NumPack conditional build script

NumPack now only supports Rust backend builds (all platforms).

Backward compatibility notes:
- Early versions supported pure Python backend, now removed
- All platforms (including Windows) require Rust toolchain
- Environment variable NUMPACK_PYTHON_ONLY=1 is deprecated, kept for backward compatibility only

Recommended usage:
- Development mode: maturin develop --release
- Build wheel: maturin build --release
- Build sdist: maturin build --sdist
"""

import os
import sys
import platform
import shutil
import subprocess
from pathlib import Path


def is_called_as_module():
    """Check if this script is being called as 'python -m build'"""
    # Check if this is being run as a module
    # When called as 'python -m build', the script path will be 'build.py' and no command arguments
    return (
        len(sys.argv) >= 1 and 
        (sys.argv[0].endswith('build.py') or sys.argv[0] == 'build.py') and
        (len(sys.argv) == 1 or not any(cmd in sys.argv for cmd in ['build', 'develop', 'info']))
    )


def call_real_build():
    """Call the real build package, not this script"""
    try:
        # Remove the current directory from sys.path temporarily
        original_path = sys.path[:]
        original_argv = sys.argv[:]
        current_dir = os.getcwd()
        
        # Remove current directory from path to avoid recursion
        paths_to_remove = ['', '.', current_dir]
        for path in paths_to_remove:
            while path in sys.path:
                sys.path.remove(path)
        
        # Import and run the real build module
        import build.__main__
        
        # Prepare arguments (remove the script name, keep the rest)
        cli_args = sys.argv[1:] if len(sys.argv) > 1 else []
        build.__main__.main(cli_args)
        
    except ImportError:
        print("Error: The 'build' package is not installed. Please install it with: pip install build")
        sys.exit(1)
    except SystemExit as e:
        sys.exit(e.code)
    finally:
        # Restore original path and argv
        sys.path[:] = original_path
        sys.argv[:] = original_argv


def is_windows():
    """Detect if running on Windows platform"""
    return platform.system().lower() == 'windows'


def should_use_python_only():
    """Decide whether to use pure Python build
    
    Note: NumPack no longer supports pure Python builds. This function is kept for backward compatibility only.
    All platforms (including Windows) must now use the Rust backend.
    """
    # Check environment variable
    if os.environ.get('NUMPACK_PYTHON_ONLY', '').lower() in ['1', 'true', 'yes']:
        print("=" * 60)
        print("WARNING: NUMPACK_PYTHON_ONLY is deprecated")
        print("NumPack no longer supports pure Python backend, all platforms require Rust toolchain")
        print("This environment variable will be ignored and Rust build will be used")
        print("=" * 60)
        return False
    
    # Windows now also uses Rust backend
    if is_windows():
        print("Note: Windows platform now also uses Rust backend build")
    
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
    """Setup pure Python build
    
    Note: This function is deprecated, NumPack no longer supports pure Python builds.
    """
    print("=" * 60)
    print("ERROR: Attempting to use pure Python build mode")
    print("")
    print("NumPack has fully migrated to Rust backend and no longer supports pure Python builds.")
    print("")
    print("Please ensure Rust toolchain is installed:")
    print("  - Visit https://rustup.rs/ to install Rust")
    print("  - Install maturin: pip install maturin")
    print("  - Build with maturin: maturin develop --release")
    print("=" * 60)
    return False


def setup_rust_build():
    """Setup Rust + Python build"""
    print("Setting up Rust + Python build mode...")
    
    # Use original configuration file (contains maturin)
    if Path('pyproject.toml.backup').exists():
        restore_original_config()
    
    return True


def run_build(build_args=None):
    """Execute build"""
    build_args = build_args or []
    
    if should_use_python_only():
        print(f"Executing pure Python build (Platform: {platform.system()})")
        
        if not setup_python_only_build():
            return False
        
        try:
            # Call the real build module directly, not via command line
            original_argv = sys.argv[:]
            original_path = sys.path[:]
            current_dir = os.getcwd()
            
            try:
                # Remove current directory from path to avoid recursion, but keep other paths
                paths_to_remove = ['', '.', current_dir]
                removed_paths = []
                for path in paths_to_remove:
                    while path in sys.path:
                        idx = sys.path.index(path)
                        removed_paths.append((idx, path))
                        sys.path.remove(path)
                
                # Import and call the real build module
                import build.__main__
                
                # Prepare arguments for the real build module
                cli_args = build_args if build_args else []
                build.__main__.main(cli_args)
                
            finally:
                # Restore original state
                sys.argv[:] = original_argv
                sys.path[:] = original_path
            
            print("Pure Python build successful")
            return True
            
        except SystemExit as e:
            if e.code == 0:
                print("Pure Python build successful")
                return True
            else:
                print(f"Pure Python build failed with exit code: {e.code}")
                return False
        except ImportError:
            print("Error: 'build' module not found, please install: pip install build")
            return False
        except Exception as e:
            print(f"Pure Python build failed: {e}")
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
            cmd = ['maturin', 'develop', '--release']
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


# Check if we're being called as 'python -m build'
if is_called_as_module():
    # If we should use Python-only build, set up the configuration first
    if should_use_python_only():
        setup_python_only_build()
    
    try:
        call_real_build()
    finally:
        # Restore original configuration if we modified it
        if should_use_python_only():
            restore_original_config()
    
    sys.exit(0)

if __name__ == '__main__':
    main() 