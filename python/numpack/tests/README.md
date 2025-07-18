# NumPack 测试说明

## 运行常规测试

默认情况下，运行以下命令会执行所有非基准测试：

```bash
pytest
```

或者：

```bash
python -m pytest
```

## 运行基准测试

基准测试被标记为 `@pytest.mark.benchmark`，这些测试涉及大量数据处理，可能需要较长时间和较多内存。

### 只运行基准测试

```bash
pytest -m benchmark
```

### 运行所有测试（包括基准测试）

```bash
pytest -m ""
```

## 测试配置

### 超时设置

所有测试都有 300 秒的超时限制。可以通过以下方式修改：

```bash
pytest --timeout=600  # 设置为600秒
```

### Windows 平台注意事项

在 Windows 平台上，测试会自动进行额外的资源清理以确保文件句柄正确释放。

## 基准测试详情

基准测试包括以下性能测试：

- `test_very_large_array`: 测试1亿行数据的处理性能
- `test_large_data`: 测试大数据处理性能  
- `test_append_operations`: 测试追加操作性能
- `test_random_access`: 测试随机访问性能
- `test_replace_operations`: 测试替换操作性能
- `test_drop_operations`: 测试删除操作性能
- `test_append_rows_operations`: 测试行追加性能
- `test_matrix_computation`: 测试矩阵计算性能

⚠️ **警告**: 基准测试会创建大型数组和文件，请确保有足够的可用内存和磁盘空间。 