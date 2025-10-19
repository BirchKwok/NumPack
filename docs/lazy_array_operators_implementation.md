# LazyArray 算术操作符实现详解

## 概述

本文档详细说明了 LazyArray 算术操作符的实现方式、与 NumPy memmap 的区别，以及如何尽量避免数据复制的优化策略。

## 实现架构

### 1. 核心设计原理

LazyArray 的算术操作符采用**透明转换**的设计模式：

```
LazyArray 操作 → 内部转换为 NumPy 数组 → NumPy 计算 → 返回 NumPy 结果
```

这种设计确保了：
- **完全兼容性**: 利用 NumPy 成熟的数学运算能力
- **简单性**: 避免重新实现复杂的数学运算
- **一致性**: 结果与 NumPy 完全一致

### 2. 实现细节

#### 基本算术操作符

```rust
// 加法操作符示例
fn __add__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let self_array = self.to_numpy_array(py)?;           // 转换为 NumPy 数组
    let result = self_array.call_method1(py, "__add__", (other,))?;  // NumPy 计算
    Ok(result.into())
}
```

所有算术操作符都遵循相同的模式：
1. 调用 `to_numpy_array()` 转换数据
2. 委托给 NumPy 的对应方法
3. 返回 NumPy 结果

#### 支持的操作符类型

| 类型 | 操作符 | 示例 | 备注 |
|------|--------|------|------|
| **算术** | `+ - * / // % **` | `lazy_array * 2.5` | 支持标量和数组 |
| **比较** | `== != < <= > >=` | `lazy_array > 5` | 返回布尔数组 |
| **一元** | `+ - ~` | `-lazy_array` | `~` 仅限整数类型 |
| **位操作** | `& \| ^ << >>` | `lazy_array & 0xFF` | 仅限整数类型 |
| **原地** | `+= -= *= /= //= %= **=` | `lazy_array += 1` | 抛出 NotImplementedError |

#### 数据转换机制

```rust
fn to_numpy_array(&self, py: Python) -> PyResult<PyObject> {
    let total_size = self.size()?;
    let mut all_data = Vec::with_capacity(total_size * self.itemsize);

    // 批量读取所有数据
    let logical_length = self.len_logical();
    for i in 0..logical_length {
        let row_data = self.get_row_data(i)?;  // 从内存映射读取
        all_data.extend(row_data);
    }

    self.create_numpy_array(py, all_data, &self.logical_shape())
}
```

## 与 NumPy memmap 的区别

### 1. **数据存储和访问**

| 特性 | LazyArray | NumPy memmap |
|------|-----------|--------------|
| **存储** | NumPack 专有格式 + 压缩 | 标准 NumPy .npy/.npz 格式 |
| **访问模式** | 懒加载 + 零拷贝内存映射 | 直接内存映射 |
| **压缩** | 支持多种压缩算法 | 不支持压缩 |
| **元数据** | 丰富的元数据索引 | 基本形状和类型信息 |

### 2. **操作符行为**

| 方面 | LazyArray | NumPy memmap |
|------|-----------|--------------|
| **原地操作** | 不支持（抛出明确错误） | 支持（直接修改文件） |
| **写操作** | 只读设计 | 读写都支持 |
| **内存使用** | 优化的批量读取 | 依赖操作系统的页面缓存 |
| **性能** | 高度优化的批量访问 | 依赖操作系统 |

### 3. **错误处理和安全性**

```python
# LazyArray: 明确的错误信息
try:
    lazy_array *= 2.5
except NotImplementedError as e:
    print(e)  # "In-place operations are not supported for read-only LazyArray..."

# NumPy memmap: 直接修改文件
memmap_array *= 2.5  # 直接修改磁盘文件
```

## 数据复制优化策略

### 1. **零拷贝设计原则**

LazyArray 在以下几个层面尽量避免数据复制：

#### a) 内存映射访问
```rust
// 零拷贝数据访问
pub(crate) fn get_row_data(&self, row_idx: usize) -> PyResult<Vec<u8>> {
    let offset = physical_idx * row_size;
    Ok(self.mmap[offset..offset + row_size].to_vec())  // 仅复制需要的行
}
```

#### b) 批量操作
```rust
fn mega_batch_get_rows(&self, py: Python, indices: Vec<usize>, batch_size: usize) -> PyResult<Vec<PyObject>> {
    let chunk_size = batch_size.max(100);  // 批量处理减少 FFI 调用

    for chunk in indices.chunks(chunk_size) {
        // 批量处理多个索引
    }
}
```

#### c) 视图操作（reshape）
```rust
fn reshape(&self, py: Python, new_shape: &Bound<'_, PyAny>) -> PyResult<Py<LazyArray>> {
    let reshaped_array = LazyArray {
        mmap: Arc::clone(&self.mmap),        // 共享同一个内存映射
        shape: final_shape,                   // 只改变形状
        // ... 其他字段共享
    };
    Py::new(py, reshaped_array)
}
```

### 2. **数据复制场景**

虽然我们尽量优化，但在某些场景下数据复制是必要的：

| 场景 | 是否复制 | 原因 |
|------|----------|------|
| **算术运算** | ✅ 是 | 需要创建 NumPy 数组进行计算 |
| **索引访问** | ⚠️ 部分复制 | 只复制访问的数据行 |
| **视图操作** | ❌ 否 | 共享底层内存映射 |
| **类型转换** | ✅ 是 | 需要转换数据格式 |

### 3. **性能优化技术**

#### a) FFI 调用优化
```rust
// 减少跨 Rust-Python 边界的调用次数
fn vectorized_gather(&self, py: Python, indices: Vec<usize>) -> PyResult<PyObject> {
    if indices.len() >= 10 {
        return self.batch_get_rows_optimized(py, &indices);  // 批量操作
    }
    // ... 小量数据直接处理
}
```

#### b) 内存预分配
```rust
fn to_numpy_array(&self, py: Python) -> PyResult<PyObject> {
    let total_size = self.size()?;
    let mut all_data = Vec::with_capacity(total_size * self.itemsize);  // 预分配内存
    // ... 批量读取
}
```

#### c) SIMD 优化（可选）
```rust
// 使用 SIMD 指令加速数据复制和处理
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
```

## 使用建议

### 1. **最佳实践**

```python
import numpack as npk
import numpy as np

# ✅ 推荐：批量操作
lazy_array = npk.load('data', lazy=True)
result = (lazy_array + 2) * 3  # 单次转换，多次运算

# ✅ 推荐：使用 NumPy 函数
result = np.sqrt(lazy_array ** 2 + 1)

# ❌ 避免：频繁的小操作
for i in range(1000):
    temp = lazy_array + i  # 每次都会转换数据
```

### 2. **内存管理**

```python
# 使用上下文管理器确保资源清理
with npk.load('large_data', lazy=True) as lazy_array:
    result = lazy_array * 2.5
    # 自动清理资源

# 或者手动清理
lazy_array = npk.load('data', lazy=True)
result = lazy_array + 1
lazy_array.close()  # 显式关闭
```

### 3. **性能考虑**

```python
# ✅ 大数组：LazyArray 优势明显
large_lazy = npk.load('huge_data', lazy=True)  # 不立即加载到内存
result = large_lazy * 1.5  # 按需加载

# ✅ 小数组：NumPy 可能更快
small_data = np.array([1, 2, 3, 4, 5])  # 直接在内存中
result = small_data * 1.5  # 无转换开销
```

## 总结

LazyArray 的算术操作符实现采用了一种实用的折中方案：

**优势：**
- 完全兼容 NumPy 的运算能力
- 保持懒加载的内存效率
- 提供清晰的错误处理
- 支持多种优化策略

**权衡：**
- 算术运算需要数据复制（无法避免）
- 不支持原地操作（安全性考虑）
- 依赖 NumPy 生态系统（非重新实现）

这种设计使得 LazyArray 在保持内存效率的同时，提供了与 NumPy memmap 相当的易用性，是一个平衡性能、安全性和开发效率的解决方案。