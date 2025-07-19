# NumPack Build Guide

This project supports automatic selection of different build methods based on platform, achieving optimal cross-platform performance and compatibility.

## Build Methods

### Automatic Platform Detection
- **Windows Platform**: Automatically use pure Python implementation to avoid Rust compilation complexity
- **Unix/Linux/macOS Platform**: Use Rust + Python hybrid implementation for optimal performance

### Manual Control
You can force specific build methods using environment variables:

```bash
# Force pure Python build (any platform)
export NUMPACK_PYTHON_ONLY=1

# Use default build method
unset NUMPACK_PYTHON_ONLY
```

## Using the Build Script

### View Current Configuration
```bash
python build.py info
```

### Development Mode Installation
```bash
# Automatically select build method
python build.py develop

# Force pure Python
python build.py develop --python-only
```

### Build Distribution Packages
```bash
# Automatically select build method
python build.py build

# Specify output directory
python build.py build --out dist

# Force pure Python
python build.py build --python-only
```

## Traditional Build Methods

### Windows Platform (Pure Python)
```bash
# Install dependencies
pip install build

# Development install
pip install -e .

# Build wheel
python -m build
```

### Unix/Linux Platform (Rust + Python)
```bash
# Install dependencies
pip install maturin

# Development install
maturin develop

# Build wheel
maturin build --release
```

## Dependencies

### Pure Python Build Dependencies
- `setuptools>=61.0`
- `wheel`
- `build` (for building)
- `numpy>=1.26.0`
- `filelock>=3.0.0`

### Rust Build Dependencies
- `maturin>=1.0,<2.0`
- Rust toolchain
- `numpy>=1.26.0`
- `filelock>=3.0.0`

## File Format Compatibility

Regardless of the build method used, NumPack ensures:
- ✅ Identical Python API
- ✅ Fully compatible file format
- ✅ Cross-platform file interoperability

## Performance Comparison

| Platform | Build Method | Compile Complexity | Runtime Performance | Compatibility |
|----------|--------------|-------------------|-------------------|---------------|
| Windows | Pure Python | Low | Medium | Very High |
| Unix/Linux | Rust + Python | Medium | High | High |
| macOS | Rust + Python | Medium | High | High |

## Troubleshooting

### Rust Compilation Issues on Windows
If you encounter Rust compilation issues on Windows, you can force pure Python:
```bash
set NUMPACK_PYTHON_ONLY=1
python build.py develop
```

### Missing Dependencies
```bash
# Missing dependencies for pure Python build
pip install build setuptools wheel

# Missing dependencies for Rust build  
pip install maturin
# Install Rust: https://rustup.rs/
```

### Development Environment Setup
```bash
# Clone repository
git clone <repo-url>
cd NumPack

# Install development dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest

# Check build configuration
python build.py info
```

## CI/CD Integration

The project's GitHub Actions workflow automatically:
- Windows: Use pure Python build
- Linux/macOS: Use Rust + Python build
- Generate platform-specific wheel packages

This ensures the best user experience and minimal build complexity. 