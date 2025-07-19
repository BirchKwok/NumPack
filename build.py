#!/usr/bin/env python3
"""
NumPack 条件性构建脚本

根据平台和环境变量选择合适的构建方式：
- Windows 平台：使用纯 Python 构建（setuptools）
- Unix/Linux 平台：使用 Rust + Python 构建（maturin）
- 环境变量 NUMPACK_PYTHON_ONLY=1：强制使用纯 Python 构建
"""

import os
import sys
import platform
import shutil
import subprocess
from pathlib import Path


def is_windows():
    """检测是否为 Windows 平台"""
    return platform.system().lower() == 'windows'


def should_use_python_only():
    """决定是否使用纯 Python 构建"""
    # 检查环境变量
    if os.environ.get('NUMPACK_PYTHON_ONLY', '').lower() in ['1', 'true', 'yes']:
        return True
    
    # Windows 平台默认使用纯 Python
    if is_windows():
        return True
    
    return False


def backup_original_config():
    """备份原始配置文件"""
    if Path('pyproject.toml').exists():
        shutil.copy('pyproject.toml', 'pyproject.toml.backup')
        print("✅ 已备份原始 pyproject.toml")


def restore_original_config():
    """恢复原始配置文件"""
    if Path('pyproject.toml.backup').exists():
        shutil.copy('pyproject.toml.backup', 'pyproject.toml')
        Path('pyproject.toml.backup').unlink()
        print("✅ 已恢复原始 pyproject.toml")


def setup_python_only_build():
    """设置纯 Python 构建"""
    print("🐍 设置纯 Python 构建模式...")
    
    # 备份原始配置
    backup_original_config()
    
    # 使用 Windows 专用配置
    if Path('pyproject.toml.windows').exists():
        shutil.copy('pyproject.toml.windows', 'pyproject.toml')
        print("✅ 已切换到纯 Python 构建配置")
    else:
        print("❌ 错误：找不到 pyproject.toml.windows 文件")
        return False
    
    return True


def setup_rust_build():
    """设置 Rust + Python 构建"""
    print("🦀 设置 Rust + Python 构建模式...")
    
    # 使用原始配置文件（包含 maturin）
    if Path('pyproject.toml.backup').exists():
        restore_original_config()
    
    return True


def run_build(build_args=None):
    """执行构建"""
    build_args = build_args or []
    
    if should_use_python_only():
        print(f"🐍 执行纯 Python 构建 (平台: {platform.system()})")
        
        if not setup_python_only_build():
            return False
        
        try:
            # 使用标准的 Python 构建工具
            cmd = [sys.executable, '-m', 'build'] + build_args
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            print("✅ 纯 Python 构建成功")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 纯 Python 构建失败: {e}")
            return False
        except FileNotFoundError:
            print("❌ 错误：未找到 'build' 模块，请安装: pip install build")
            return False
        finally:
            # 恢复原始配置
            restore_original_config()
    
    else:
        print(f"🦀 执行 Rust + Python 构建 (平台: {platform.system()})")
        
        setup_rust_build()
        
        try:
            # 使用 maturin 构建
            cmd = ['maturin', 'build', '--release'] + build_args
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            print("✅ Rust + Python 构建成功")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Rust 构建失败: {e}")
            return False
        except FileNotFoundError:
            print("❌ 错误：未找到 'maturin'，请安装: pip install maturin")
            return False


def run_develop():
    """执行开发模式安装"""
    if should_use_python_only():
        print(f"🐍 执行纯 Python 开发安装 (平台: {platform.system()})")
        
        if not setup_python_only_build():
            return False
        
        try:
            # 使用 pip editable 安装
            cmd = [sys.executable, '-m', 'pip', 'install', '-e', '.']
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            print("✅ 纯 Python 开发安装成功")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 纯 Python 开发安装失败: {e}")
            return False
        finally:
            # 恢复原始配置
            restore_original_config()
    
    else:
        print(f"🦀 执行 Rust + Python 开发安装 (平台: {platform.system()})")
        
        try:
            # 使用 maturin develop
            cmd = ['maturin', 'develop']
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            print("✅ Rust + Python 开发安装成功")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Rust 开发安装失败: {e}")
            return False
        except FileNotFoundError:
            print("❌ 错误：未找到 'maturin'，请安装: pip install maturin")
            return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NumPack 条件性构建脚本")
    parser.add_argument('command', choices=['build', 'develop', 'info'], 
                        help='要执行的命令')
    parser.add_argument('--python-only', action='store_true',
                        help='强制使用纯 Python 构建')
    parser.add_argument('--out', help='输出目录（仅用于 build）')
    
    args = parser.parse_args()
    
    # 设置环境变量
    if args.python_only:
        os.environ['NUMPACK_PYTHON_ONLY'] = '1'
    
    print(f"🔧 NumPack 构建脚本")
    print(f"平台: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")
    print(f"构建模式: {'纯 Python' if should_use_python_only() else 'Rust + Python'}")
    print("-" * 50)
    
    if args.command == 'info':
        print(f"当前配置:")
        print(f"  - 平台: {platform.system()}")
        print(f"  - 使用纯 Python: {should_use_python_only()}")
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