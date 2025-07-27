# 项目备忘录
这是一个旨在替换numpy的npy/npz格式的文件协议格式代码库，它拥有两个api功能完全一致，并且文件协议一致的后端实现（Rust和Python后端）。它希望借助Rust后端（同时提供Python后端以保证最大的兼容性）实现让人非常认可的I/O速度、零拷贝的懒加载功能、以及可选的高性能压缩功能（zstd）。

## 项目要求
- **四个极高**: 极高的I/O速度、极高的内存映射访问速度、极高的随机访问速度、极高的索引速度

## 开发原则
- 当前的python api坚决不能变化，只能新增，不能改变现有api的逻辑。
- 非必要不要创建总结文档。
- 保持最小化修改。
- 总是进行测试，确保不会引入bug和性能衰减。
- 每个性能优化策略，确保可以实现性能优化后，就可以转为默认行为。
- 确保两个后端对python api的功能支持一致
- **使用dev conda环境进行测试**
- 使用build.py脚本进行编译安装，**注意，一定要确保安装环境正确，并且成功安装**

## API基本用法

### 核心操作
```python
import numpy as np
from numpack import NumPack

# 创建 NumPack 实例
npk = NumPack("data_directory")

# 保存数组
arrays = {
    'array1': np.random.rand(1000, 100).astype(np.float32),
    'array2': np.random.rand(500, 200).astype(np.float32)
}
npk.save(arrays)                    # 默认不压缩

# 加载数组
loaded = npk.load("array1")           # 普通加载
lazy_array = npk.load("array1", lazy=True)  # 懒加载

# 高级操作
npk.replace({'array1': replacement}, [0, 1, 2])  # 替换特定行
npk.append({"new_arrays": new_arrays})                            # 追加数组
npk.drop('array1')                               # 删除数组
npk.drop("array1", [0,3])  # 删除特定行
data = npk.getitem('array1', [0, 1, 2])         # 随机访问
shapes = npk.get_shape()                         # 获取形状信息
metadata = npk.get_metadata()                    # 获取元数据
member_list = npk.get_member_list()  # 获取当前所有数组名

# 流式处理大数组
for batch in npk.stream_load('array1', buffer_size=1000):
    process_batch(batch)
```

### 懒加载和缓冲操作
```python
# 懒加载适用于大规模数据集
lazy_array = npk.load("large_array", lazy=True)
similarity_scores = np.inner(data[0], lazy_array)  # 只加载需要的数据
```

## 当前性能状况

### 基准测试结果总结
- **保存操作**: Python后端整体表现更稳定，Rust后端在某些数据类型上仍需优化
- **加载操作**: 两个后端性能相近，Rust后端在uint8等小数据类型上有优势
- **懒加载**: Rust后端略优于Python后端（110K vs 95K操作/秒）
- **压缩功能**: 目前NumPack压缩比为1.0x，而NumPy npz可达229x压缩比
- **内存使用**: Rust后端在某些场景下内存效率更高

### 主要性能瓶颈
1. **int64保存性能**: Rust后端比Python慢（0.277s vs 0.186s）
2. **缺乏可选的zstd数据压缩**: 没有像numpy npz那样的高效压缩
3. **序列化开销**: MessagePack在某些数据类型上性能不够理想，需要使用自定义二进制格式加速
4. **FFI边界开销**: Python-Rust调用频繁导致性能损失

## 正在进行的性能优化工作

### 已实现的优化模块
- ✅ **SIMD处理器**: `src/memory/simd_processor.rs` - 向量化数据处理
- ✅ **多平台SIMD优化**: `src/memory/numpack_simd/` - 完整的SIMD抽象层
  - x86_64支持 (SSE2/AVX2/AVX512) 和 ARM支持 (NEON)
  - 数据类型特异化SIMD策略，针对14种NumPy数据类型优化
  - 访问模式感知调度器，智能选择最优SIMD策略
  - 实测性能：流式处理4,417 MB/s，聚集访问1,041 MB/s
- ✅ **批量访问引擎**: `src/batch_access_engine.rs` - 高效批量数据访问
- ✅ **索引优化**: `src/indexing/optimizations.rs` - 多种索引优化策略
- ✅ **缓存系统**: `src/cache/` - 多级缓存和压缩缓存
- ✅ **性能监控**: `src/performance/` - 全面的性能指标追踪

## .npk文件格式协议

- .npk格式的开发，必须遵循NumPack_File_Format_Specification.spec中的约定，并保证python后端和rust后端的文件协议格式一致，文件读写不受后端切换的影响。
- 如果需要更改文件格式，必须同步修改python、rust后端，确保两端文件读写兼容
- 当前统一使用MessagePack格式确保跨平台兼容性
- 文件结构: `metadata.npkm`(元数据) + `data_<array_name>.npkd`(原始数据) + 锁文件

### 数据类型映射
支持完整的NumPy数据类型，包括bool、int8-64、uint8-64、float16/32/64、complex64/128，所有多字节数据使用小端序存储确保跨平台兼容。

## 用户意图尊重的FFI优化策略

### 核心设计原则
- **尊重用户意图**：不曲解用户的使用方式，在自然的边界进行优化
- **透明优化**：对用户完全透明，API保持简单性
- **智能识别**：自动区分不同的访问模式并采用相应优化

### 用户意图分类和优化策略

| 使用方式 | 用户意图 | 系统行为 | FFI开销 | 应用场景 |
|----------|----------|----------|---------|----------|
| `lazy_array[i]` | 明确的单次访问 | 尊重意图，不干预 | 1x | 探索性访问、条件访问 |
| `lazy_array[indices_list]` | 明确的批量访问 | 一次性FFI调用 | 0.1x | 批量数据处理 |
| `lazy_array[np.array([...])]` | NumPy数组索引 | 批量处理优化 | 0.1x | 科学计算、数据分析 |
| `lazy_array[start:end]` | 切片访问 | 现有优化逻辑 | 0.5x | 范围数据访问 |

### 实现细节
- **索引类型识别**：在Rust层面区分 `UserIntent::SingleAccess`、`UserIntent::BatchAccess`、`UserIntent::ComplexIndex`
- **性能测试修正**：移除了测试中的错误使用模式（循环单次访问），确保基准测试反映真实使用场景
- **一次性批量处理**：列表和NumPy数组索引使用单次FFI调用，大幅减少跨语言边界开销

### 错误使用模式（已在测试中修正）
```python
# ❌ 错误模式：不应该在循环中进行单次访问
for i in indices:
    row = lazy_array[i]  # 每次FFI调用

# ✅ 正确模式：批量访问
rows = lazy_array[indices]  # 单次FFI调用
```

## 防止过度优化
- SIMD指令优化、多线程、数据预取等操作虽然是大部分情况下都通用的优化策略，但不一定适合当前的二进制文件内部布局，需要谨慎处理，每一步优化都要确认不会产生负面影响
- 性能优化必须基于实际基准测试结果，避免理论上的优化导致实际性能下降
- 优化策略应该是渐进式的，每个模块独立验证后再整合
- 保持代码的可维护性，复杂的优化应该有充分的文档和测试覆盖
- **FFI优化应基于用户真实意图，而非强制改变用户使用习惯**

## 目标规划
- **短期目标**: 让Rust后端在所有基准测试中大幅领先Python后端
- **中期目标**: 实现166x性能提升声明，特别在数据修改和随机访问操作上
- **长期目标**: 成为NumPy生态系统中最高性能的数组存储解决方案 