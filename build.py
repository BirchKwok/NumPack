#!/usr/bin/env python3
"""
NumPack 智能构建脚本

自动检测平台和 GPU 能力，使用最高性能配置编译

特性:
- 自动检测 GPU（MPS/WebGPU）并启用对应特性
- 默认使用 release 模式和最高性能优化
- 自动处理多 Python 版本环境
- 简单运行: python build.py

用法:
  python build.py              # 智能构建（自动检测 GPU + release 模式）
  python build.py --no-gpu     # 禁用 GPU，仅 CPU
  python build.py --gpu mps    # 强制使用 MPS
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


def detect_gpu_capability(system, machine):
    """
    自动检测 GPU 能力并返回推荐的特性
    
    Returns:
        list: GPU 特性列表，如 ['gpu-mps'] 或 []
    """
    print(f"\n🎮 GPU 检测:")
    
    gpu_features = []
    
    # 1. 检测 Apple Silicon (MPS)
    if system == "Darwin" and machine == "arm64":
        # Apple Silicon - 支持 MPS
        try:
            # 尝试检测 Metal 是否可用
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if "Metal" in result.stdout or result.returncode == 0:
                print("  ✓ 检测到 Apple Silicon GPU (Metal Performance Shaders)")
                gpu_features.append('gpu-mps')
            else:
                print("  ⚠ Apple Silicon 但未检测到 Metal")
        except:
            # 如果无法运行 system_profiler，仍然假设有 Metal
            print("  ✓ 检测到 Apple Silicon - 假设支持 MPS")
            gpu_features.append('gpu-mps')
    
    # 2. 检测 NVIDIA GPU (CUDA)
    elif system == "Linux" or system == "Windows":
        # 尝试检测 NVIDIA GPU
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print("  ✓ 检测到 NVIDIA GPU")
                print("  ⚠ CUDA 支持尚未实现，将使用 WebGPU")
                gpu_features.append('gpu-wgpu')
            else:
                print("  ℹ 未检测到 NVIDIA GPU")
        except:
            print("  ℹ 未检测到 NVIDIA GPU")
    
    # 3. 如果没有检测到特定 GPU，尝试 WebGPU（通用）
    if not gpu_features:
        print("  ✗ 未检测到 GPU - 将使用纯 CPU 构建")
    
    return gpu_features


def build_feature_string(gpu_features):
    """
    构建 Cargo features 字符串
    
    Args:
        gpu_features: GPU 特性列表
    
    Returns:
        str: features 字符串，如 "extension-module,rayon,gpu-mps"
    """
    # 默认特性
    default_features = ['extension-module', 'rayon']
    
    # 添加 GPU 特性
    all_features = default_features + gpu_features
    
    return ','.join(all_features)


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
             'print("能力:", engine.capabilities()); '
             'print("GPU 可用:", engine.is_gpu_available())'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("  ✓ NumPack 导入成功")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
            
            # 检查 stderr（GPU 初始化信息）
            if result.stderr:
                for line in result.stderr.strip().split('\n'):
                    if 'Metal' in line or 'GPU' in line:
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
        description="NumPack 智能构建脚本 - 自动检测 GPU 并使用最高性能配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python build.py                # 智能构建（自动检测）
  python build.py --no-gpu       # 禁用 GPU
  python build.py --gpu mps      # 强制使用 MPS
  python build.py --verify-only  # 仅验证安装
        """
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='禁用 GPU，使用纯 CPU 构建'
    )
    
    parser.add_argument(
        '--gpu',
        choices=['mps', 'wgpu', 'cuda', 'rocm', 'all', 'universal'],
        help='强制使用指定的 GPU 后端（覆盖自动检测）\n'
             'universal: 编译所有GPU后端（通用包）'
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
    system, machine = detect_platform()
    
    # 确定 GPU 特性
    gpu_features = []
    
    if args.no_gpu:
        print(f"\n⚙️  GPU 已禁用（用户指定）")
    elif args.gpu:
        # 用户指定 GPU
        print(f"\n⚙️  使用指定的 GPU: {args.gpu}")
        if args.gpu == 'mps':
            gpu_features = ['gpu-mps']
        elif args.gpu == 'wgpu':
            gpu_features = ['gpu-wgpu']
        elif args.gpu == 'cuda':
            gpu_features = ['gpu-cuda']
        elif args.gpu == 'rocm':
            gpu_features = ['gpu-rocm']
        elif args.gpu == 'all':
            gpu_features = ['gpu-all']
        elif args.gpu == 'universal':
            print("  ⚡ 通用包模式：启用所有 GPU 后端")
            print("  ℹ️  运行时会自动检测并选择可用的 GPU")
            gpu_features = ['gpu-universal']
    else:
        # 自动检测
        gpu_features = detect_gpu_capability(system, machine)
    
    # 构建 features 字符串
    features_str = build_feature_string(gpu_features)
    
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
    print("  2. 完整测试: python test_gpu_detection.py")
    print("  3. 运行示例: python examples/gpu_demo.py")
    print("  4. 验证安装: python build.py --verify-only")
    
    print("\n💡 使用提示:")
    print("  import numpack")
    print("  engine = numpack.VectorEngine()")
    print("  scores = engine.batch_compute(query, candidates, metric='dot', device='mps')")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main() 
