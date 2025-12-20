#!/usr/bin/env python3
"""
NumPack smart build script.

Build with a high-performance configuration.

Features:
- Uses release mode and high-performance optimizations by default
- Automatically handles multi-Python-version environments
- Quick start: python build.py

Usage:
  python build.py              # smart build (release mode)
  python build.py --help       # show help
"""

import os
import sys
import platform
import subprocess
import argparse
import shutil
from pathlib import Path


def print_banner():
    """Print banner."""
    print("\n" + "=" * 70)
    print("NumPack Build System")
    print("=" * 70)


def detect_platform():
    """Detect platform information."""
    system = platform.system()
    machine = platform.machine()
    
    print(f"\nPlatform:")
    print(f"  OS: {system}")
    print(f"  Arch: {machine}")
    print(f"  Python: {platform.python_version()}")
    print(f"  Python executable: {sys.executable}")
    
    return system, machine


def build_feature_string():
    """
    Build Cargo features string.
    
    Returns:
        str: Features string, for example "extension-module,rayon"
    """
    # Default features
    default_features = ['extension-module', 'rayon']
    
    return ','.join(default_features)


def run_maturin_build_wheel(features_str, python_interpreter):
    """
    Build wheel and tar.gz (sdist) files using maturin.

    Args:
        features_str: Cargo features string
        python_interpreter: Python interpreter path

    Returns:
        list: List of built file paths (wheel and tar.gz); returns None on failure
    """
    # Use dist/ under the project root as output directory
    output_dir = Path(__file__).parent / 'dist'
    # Clear the output directory first
    if output_dir.exists():
        for file in output_dir.glob('*'):
            file.unlink()
    output_dir.mkdir(exist_ok=True)  # Ensure directory exists

    # Build command - use -i to specify Python version, generate wheel and sdist
    cmd = ['maturin', 'build', '--release', '--sdist', '-i', python_interpreter, '-o', str(output_dir)]
    
    # Add features
    if features_str:
        cmd.extend(['--features', features_str])
    
    print(f"\nRunning: {' '.join(cmd)}")
    print("=" * 70)
    
    try:
        # Run build
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        # Find built files (wheel and tar.gz)
        built_files = list(Path(output_dir).glob('*.whl')) + list(Path(output_dir).glob('*.tar.gz'))
        if built_files:
            # Return a list of built file paths
            return [str(f) for f in built_files]
        else:
            return None
        
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        return None
    except FileNotFoundError:
        print("Error: maturin not found")
        print("Please install: pip install maturin")
        return None


def install_wheel(wheel_paths, python_interpreter):
    """
    Install wheel files.

    Args:
        wheel_paths: A list of wheel file paths or a single path
        python_interpreter: Python interpreter path
    """
    print("\n" + "=" * 70)
    print("Installing wheel")
    print("=" * 70)

    # Ensure wheel_paths is a list
    if isinstance(wheel_paths, str):
        wheel_paths = [wheel_paths]

    # Install only wheel files; skip tar.gz
    wheel_files = [p for p in wheel_paths if p.endswith('.whl')]

    if not wheel_files:
        print("No wheel files found")
        return False

    # Current Python version (major.minor)
    python_version = f"{sys.version_info.major}{sys.version_info.minor}"

    # Install only wheels matching the current Python version
    compatible_wheels = [w for w in wheel_files if f"cp{python_version}" in w]

    if not compatible_wheels:
        print(f"No wheels compatible with Python {python_version} were found")
        return False

    print(f"  Found {len(compatible_wheels)} compatible wheel(s)")
    cmd = [python_interpreter, '-m', 'pip', 'install', '--force-reinstall'] + compatible_wheels
    
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True)
        print("Install succeeded")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Install failed: {e}")
        return False


def sync_extension_module(python_interpreter):
    """Sync installed extension module into source tree to avoid loading an old version in tests."""
    project_root = Path(__file__).parent
    source_dir = project_root / 'python' / 'numpack'
    if not source_dir.exists():
        return

    try:
        result = subprocess.run(
            [
                python_interpreter,
                '-c',
                (
                    'import numpack, pathlib; '
                    'print(pathlib.Path(numpack._lib_numpack.__file__).resolve())'
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Failed to locate installed extension module: {exc}")
        return

    extension_path = Path(result.stdout.strip())
    if not extension_path.exists():
        print(f"Extension file not found: {extension_path}")
        return

    destination = source_dir / extension_path.name
    try:
        shutil.copy2(extension_path, destination)
        print(f"Synced extension module into source tree: {destination.name}")
    except Exception as exc:
        print(f"Failed to sync extension module: {exc}")


def verify_installation(python_interpreter):
    """Verify installation."""
    print(f"\nVerify installation:")
    
    try:
            # Try importing numpack
        result = subprocess.run(
            [python_interpreter, '-c', 
             'import numpack; from numpack.vector_engine import VectorEngine; '
             'print("NumPack version:", numpack.__version__ if hasattr(numpack, "__version__") else "unknown"); '
             'engine = VectorEngine(); '
             'print("Capabilities:", engine.capabilities())'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("  NumPack import succeeded")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
            
            return True
        else:
            print("  NumPack import failed")
            print(f"  {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  Verification error: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NumPack build script (high-performance configuration)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build.py                # smart build (release mode)
  python build.py --verify-only  # verify installation only
        """
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Verify current installation only; do not build'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Verify-only mode
    if args.verify_only:
        verify_installation(sys.executable)
        return
    
    # Detect platform
    detect_platform()
    
    # Build features string
    features_str = build_feature_string()
    
    print(f"\nStarting build:")
    print(f"  Mode: release (high performance)")
    print(f"  Features: {features_str}")
    print(f"  Target Python: {sys.executable}")
    
    # Step 1: Build wheel and sdist
    built_files = run_maturin_build_wheel(features_str, sys.executable)

    if not built_files:
        print("\n" + "=" * 70)
        print("Build failed")
        print("=" * 70)
        sys.exit(1)

    print("=" * 70)
    print("Build succeeded. Generated files:")
    for file_path in built_files:
        print(f"  - {Path(file_path).name}")

    # Step 2: Install wheel
    if not install_wheel(built_files, sys.executable):
        print("\n" + "=" * 70)
        print("Install failed")
        print("=" * 70)
        sys.exit(1)

    # Step 2.5: Sync extension module into source tree to keep test environment consistent
    sync_extension_module(sys.executable)
    
    # Step 3: Verify installation
    verify_installation(sys.executable)

    # Print usage hints
    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70)
    
    print("\nNext steps:")
    print("  1. Quick test: python -c 'from numpack.vector_engine import VectorEngine; print(VectorEngine().capabilities())'")
    print("  2. Verify install: python build.py --verify-only")
    
    print("\nUsage hint:")
    print("  import numpack; from numpack.vector_engine import VectorEngine;")
    print("  engine = VectorEngine()")
    print("  scores = engine.batch_compute(query, candidates, 'dot')")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main() 
