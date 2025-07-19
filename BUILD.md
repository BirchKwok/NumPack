# NumPack 构建指南

本项目支持根据平台自动选择不同的构建方式，实现跨平台的最佳性能和兼容性。

## 构建方式

### 自动平台检测
- **Windows 平台**: 自动使用纯 Python 实现，避免 Rust 编译复杂性
- **Unix/Linux/macOS 平台**: 使用 Rust + Python 混合实现，获得最佳性能

### 手动控制
可以通过环境变量强制使用特定构建方式：

```bash
# 强制使用纯 Python 构建（任何平台）
export NUMPACK_PYTHON_ONLY=1

# 使用默认构建方式
unset NUMPACK_PYTHON_ONLY
```

## 使用构建脚本

### 查看当前配置
```bash
python build.py info
```

### 开发模式安装
```bash
# 自动选择构建方式
python build.py develop

# 强制使用纯 Python
python build.py develop --python-only
```

### 构建分发包
```bash
# 自动选择构建方式
python build.py build

# 指定输出目录
python build.py build --out dist

# 强制使用纯 Python
python build.py build --python-only
```

## 传统构建方式

### Windows 平台（纯 Python）
```bash
# 安装依赖
pip install build

# 开发安装
pip install -e .

# 构建 wheel
python -m build
```

### Unix/Linux 平台（Rust + Python）
```bash
# 安装依赖
pip install maturin

# 开发安装
maturin develop

# 构建 wheel
maturin build --release
```

## 依赖说明

### 纯 Python 构建依赖
- `setuptools>=61.0`
- `wheel`
- `build` (用于构建)
- `numpy>=1.26.0`
- `filelock>=3.0.0`

### Rust 构建依赖
- `maturin>=1.0,<2.0`
- Rust 工具链
- `numpy>=1.26.0`
- `filelock>=3.0.0`

## 文件格式兼容性

无论使用哪种构建方式，NumPack 都确保：
- ✅ 完全相同的 Python API
- ✅ 完全兼容的文件格式
- ✅ 跨平台文件互操作性

## 性能对比

| 平台 | 构建方式 | 编译复杂度 | 运行性能 | 兼容性 |
|------|----------|------------|----------|--------|
| Windows | 纯 Python | 低 | 中等 | 极高 |
| Unix/Linux | Rust + Python | 中等 | 高 | 高 |
| macOS | Rust + Python | 中等 | 高 | 高 |

## 故障排除

### Windows 上 Rust 编译问题
如果在 Windows 上遇到 Rust 编译问题，可以强制使用纯 Python：
```bash
set NUMPACK_PYTHON_ONLY=1
python build.py develop
```

### 缺少依赖
```bash
# 纯 Python 构建缺少依赖
pip install build setuptools wheel

# Rust 构建缺少依赖  
pip install maturin
# 安装 Rust: https://rustup.rs/
```

### 开发环境设置
```bash
# 克隆仓库
git clone <repo-url>
cd NumPack

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
python -m pytest

# 检查构建配置
python build.py info
```

## CI/CD 集成

项目的 GitHub Actions 工作流会自动：
- Windows: 使用纯 Python 构建
- Linux/macOS: 使用 Rust + Python 构建
- 生成对应平台的 wheel 包

这确保了最佳的用户体验和最小的构建复杂度。 