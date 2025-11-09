#!/usr/bin/env python3
"""
NumPack 智能构建脚本

使用最高性能配置编译

特性:
- 默认使用 release 模式和最高性能优化
- 自动处理多 Python 版本环境
- 简单运行: python build.py

用法:
  python build.py              # 智能构建（release 模式）
  python build.py --help       # 显示帮助
"""

import os
import sys
import platform
import subprocess
import argparse
import tempfile
from pathlib import Path


def print_banner():
    """打印横幅"""
    print("\n" + "=" * 70)
    print("NumPack 智能构建系统")
    print("=" * 70)


def detect_platform():
    """检测平台信息"""
    system = platform.system()
    machine = platform.machine()
    
    print(f"\n🔍 平台检测:")
    print(f"  操作系统: {system}")
    print(f"  架构: {machine}")
    print(f"  Python: {platform.python_version()}")
    print(f"  Python 路径: {sys.executable}")
    
    return system, machine


def build_feature_string():
    """
    构建 Cargo features 字符串
    
    Returns:
        str: features 字符串，如 "extension-module,rayon"
    """
    # 默认特性
    default_features = ['extension-module', 'rayon']
    
    return ','.join(default_features)


def run_maturin_build_wheel(features_str, python_interpreter):
    """
    使用 maturin 构建 wheel 文件
    
    Args:
        features_str: Cargo features 字符串
        python_interpreter: Python 解释器路径
    
    Returns:
        str: 构建的 wheel 文件路径，失败则返回 None
    """
    # 创建临时输出目录
    output_dir = tempfile.mkdtemp(prefix='numpack_build_')
    
    # 构建命令 - 使用 -i 参数指定 Python 版本
    cmd = ['maturin', 'build', '--release', '-i', python_interpreter, '-o', output_dir]
    
    # 添加 features
    if features_str:
        cmd.extend(['--features', features_str])
    
    print(f"\n执行命令: {' '.join(cmd)}")
    print("=" * 70)
    
    try:
        # 运行构建
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        # 查找生成的 wheel 文件
        wheel_files = list(Path(output_dir).glob('*.whl'))
        if wheel_files:
            return str(wheel_files[0])
        else:
            return None
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 构建失败: {e}")
        return None
    except FileNotFoundError:
        print("❌ 错误: 未找到 maturin")
        print("请安装: pip install maturin")
        return None


def install_wheel(wheel_path, python_interpreter):
    """
    安装 wheel 文件
    
    Args:
        wheel_path: wheel 文件路径
        python_interpreter: Python 解释器路径
    """
    print("\n" + "=" * 70)
    print("📦 安装 wheel 文件")
    print("=" * 70)
    
    cmd = [python_interpreter, '-m', 'pip', 'install', '--force-reinstall', wheel_path]
    
    print(f"执行命令: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True)
        print("✓ 安装成功!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        return False


def verify_installation(python_interpreter):
    """验证安装"""
    print(f"\n🔍 验证安装:")
    
    try:
            # 尝试导入 numpack
        result = subprocess.run(
            [python_interpreter, '-c', 
             'import numpack; '
             'print("NumPack 版本:", numpack.__version__ if hasattr(numpack, "__version__") else "未知"); '
             'engine = numpack.VectorEngine(); '
             'print("能力:", engine.capabilities())'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("  ✓ NumPack 导入成功")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
            
            return True
        else:
            print("  ❌ NumPack 导入失败")
            print(f"  {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ⚠ 验证时出错: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="NumPack 智能构建脚本 - 使用最高性能配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python build.py                # 智能构建（release 模式）
  python build.py --verify-only  # 仅验证安装
        """
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='仅验证当前安装，不构建'
    )
    
    args = parser.parse_args()
    
    # 打印横幅
    print_banner()
    
    # 仅验证模式
    if args.verify_only:
        verify_installation(sys.executable)
        return
    
    # 检测平台
    detect_platform()
    
    # 构建 features 字符串
    features_str = build_feature_string()
    
    print(f"\n🔨 开始构建:")
    print(f"  模式: release (最高性能)")
    print(f"  特性: {features_str}")
    print(f"  目标 Python: {sys.executable}")
    
    # 步骤 1: 构建 wheel
    wheel_path = run_maturin_build_wheel(features_str, sys.executable)
    
    if not wheel_path:
        print("\n" + "=" * 70)
        print("❌ 构建失败")
        print("=" * 70)
        sys.exit(1)
    
    print("=" * 70)
    print(f"✓ Wheel 构建成功: {wheel_path}")
    
    # 步骤 2: 安装 wheel
    if not install_wheel(wheel_path, sys.executable):
        print("\n" + "=" * 70)
        print("❌ 安装失败")
        print("=" * 70)
        sys.exit(1)
    
    # 步骤 3: 验证安装
    verify_installation(sys.executable)
    
    # 清理临时文件
    try:
        Path(wheel_path).parent.rmdir()
    except:
        pass
    
    # 打印使用提示
    print("\n" + "=" * 70)
    print("🎉 完成!")
    print("=" * 70)
    
    print("\n📚 后续步骤:")
    print("  1. 快速测试: python quick_test.py")
    print("  2. 验证安装: python build.py --verify-only")
    
    print("\n💡 使用提示:")
    print("  import numpack")
    print("  engine = numpack.VectorEngine()")
    print("  scores = engine.batch_compute(query, candidates, metric='dot')")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main() 
