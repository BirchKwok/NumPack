# NumPack vs NumPy 性能基准测试报告

测试环境信息:
- Python 版本: 3.12.4 | packaged by Anaconda, Inc. | (main, Jun 18 2024, 10:07:17) [Clang 14.0.6 ]
- NumPy 版本: 2.3.1
- 系统信息: posix.uname_result(sysname='Darwin', nodename='MacBook-Pro-3.local', release='24.5.0', version='Darwin Kernel Version 24.5.0: Tue Apr 22 19:54:49 PDT 2025; root:xnu-11417.121.6~2/RELEASE_ARM64_T6000', machine='arm64')
- CPU 核心数: 10
- 总内存: 32.0 GB

## NumPack Save

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Save | python | 0.068 | 67.0 | 57.2 | 846.5 |  |
| NumPack Save | rust | 0.051 | 25.2 | 57.2 | 1132.6 |  |
| NumPack Save float32 | python | 0.014 | 0.0 | 9.5 | 695.8 |  |
| NumPack Save float32 | rust | 0.013 | 1.5 | 9.5 | 749.8 |  |
| NumPack Save float64 | python | 0.017 | 0.0 | 19.1 | 1130.7 |  |
| NumPack Save float64 | rust | 0.017 | 0.0 | 19.1 | 1134.9 |  |
| NumPack Save int32 | python | 0.013 | 1.5 | 9.5 | 729.1 |  |
| NumPack Save int32 | rust | 0.014 | 0.0 | 9.5 | 669.6 |  |
| NumPack Save int64 | python | 0.015 | 0.0 | 19.1 | 1304.3 |  |
| NumPack Save int64 | rust | 0.018 | 0.2 | 19.1 | 1073.5 |  |
| NumPack Save uint8 | python | 0.011 | 0.0 | 2.4 | 225.5 |  |
| NumPack Save uint8 | rust | 0.010 | 1.4 | 2.4 | 235.2 |  |

## NumPack Load

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Load | python | 0.010 | 0.0 | 0.0 | 5729.4 |  |
| NumPack Load | rust | 0.011 | 0.0 | 0.0 | 5234.7 |  |
| NumPack Load float32 | python | 0.003 | 0.0 | 0.0 | 3372.8 |  |
| NumPack Load float32 | rust | 0.002 | 0.0 | 0.0 | 5207.2 |  |
| NumPack Load float64 | python | 0.004 | 0.0 | 0.0 | 4419.6 |  |
| NumPack Load float64 | rust | 0.004 | 0.0 | 0.0 | 5448.3 |  |
| NumPack Load int32 | python | 0.002 | 0.0 | 0.0 | 4634.1 |  |
| NumPack Load int32 | rust | 0.002 | 0.0 | 0.0 | 4511.7 |  |
| NumPack Load int64 | python | 0.004 | 0.0 | 0.0 | 5328.6 |  |
| NumPack Load int64 | rust | 0.004 | 0.0 | 0.0 | 5330.7 |  |
| NumPack Load uint8 | python | 0.001 | 0.0 | 0.0 | 2823.2 |  |
| NumPack Load uint8 | rust | 0.001 | 0.0 | 0.0 | 3897.3 |  |

## NumPack Lazy

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Lazy Load | python | 0.000 | 0.0 | 0.0 | 208169.0 |  |
| NumPack Lazy Load | rust | 0.001 | 0.0 | 0.0 | 78545.6 |  |
| NumPack Lazy Load Memory | python | 0.000 | 0.0 | 0.0 | 0.0 | data_size_mb=19.07, memory_ratio=0.00 |
| NumPack Lazy Load Memory | rust | 0.000 | 0.0 | 0.0 | 0.0 | data_size_mb=19.07, memory_ratio=0.00 |
| NumPack Lazy Matrix Dot | python | 0.001 | 12.4 | 0.0 | 0.0 |  |
| NumPack Lazy Matrix Dot | rust | 0.001 | 12.2 | 0.0 | 0.0 |  |
| NumPack Lazy Matrix Inner | python | 0.000 | 0.0 | 0.0 | 0.0 |  |
| NumPack Lazy Matrix Inner | rust | 0.000 | 0.0 | 0.0 | 0.0 |  |

## NumPy .npy

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy .npy Compression Random Data | numpy | 0.000 | 0.0 | 9.5 | 0.0 | original_size_mb=9.54, compression_ratio=1.00 |
| NumPy .npy Compression Repeated Pattern | numpy | 0.000 | 0.0 | 9.5 | 0.0 | original_size_mb=9.54, compression_ratio=1.00 |
| NumPy .npy Compression Sparse Data | numpy | 0.000 | 0.0 | 9.5 | 0.0 | original_size_mb=9.54, compression_ratio=1.00 |
| NumPy .npy Load | numpy | 0.012 | 0.0 | 0.0 | 4927.8 |  |
| NumPy .npy Save | numpy | 0.018 | 0.0 | 57.2 | 3113.8 |  |
| NumPy .npy mmap Load | numpy | 0.008 | 0.1 | 0.0 | 7577.5 |  |

## NumPy .npz

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy .npz Compression Random Data | numpy | 0.000 | 0.0 | 8.8 | 0.0 | original_size_mb=9.54, compression_ratio=1.08 |
| NumPy .npz Compression Repeated Pattern | numpy | 0.000 | 0.0 | 0.0 | 0.0 | original_size_mb=9.54, compression_ratio=255.44 |
| NumPy .npz Compression Sparse Data | numpy | 0.000 | 0.0 | 0.7 | 0.0 | original_size_mb=9.54, compression_ratio=13.35 |
| NumPy .npz Load | numpy | 0.019 | 1.4 | 0.0 | 2971.4 |  |
| NumPy .npz Save | numpy | 0.026 | 9.5 | 57.2 | 2220.0 |  |
| NumPy .npz mmap Load | numpy | 0.016 | 0.8 | 0.0 | 3487.4 |  |

## NumPack Single

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Single Random Access | python | 0.023 | 0.0 | 0.0 | 0.0 |  |
| NumPack Single Random Access | rust | 0.022 | 0.0 | 0.0 | 0.0 |  |

## NumPack Batch

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Batch Random Access | python | 0.001 | 0.0 | 0.0 | 0.0 |  |
| NumPack Batch Random Access | rust | 0.001 | 0.0 | 0.0 | 0.0 |  |

## NumPy In-Memory

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy In-Memory Batch Random Access | numpy | 0.000 | 0.0 | 0.0 | 0.0 |  |
| NumPy In-Memory Matrix Dot | numpy | 0.001 | 0.0 | 0.0 | 0.0 |  |
| NumPy In-Memory Matrix Inner | numpy | 0.000 | 0.0 | 0.0 | 0.0 |  |
| NumPy In-Memory Random Access | numpy | 0.000 | 0.0 | 0.0 | 0.0 |  |

## NumPy mmap

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy mmap Batch Random Access | numpy | 0.001 | 10.6 | 0.0 | 0.0 |  |
| NumPy mmap Chunk Read | numpy | 0.000 | 0.0 | 0.0 | 0.0 |  |
| NumPy mmap Fibonacci Access | numpy | 0.000 | 0.1 | 0.0 | 0.0 |  |
| NumPy mmap Load Memory | numpy | 0.000 | 0.0 | 0.0 | 0.0 | data_size_mb=19.07, memory_ratio=0.00 |
| NumPy mmap Matrix Dot | numpy | 0.002 | 12.2 | 0.0 | 0.0 |  |
| NumPy mmap Matrix Inner | numpy | 0.001 | 0.0 | 0.0 | 0.0 |  |
| NumPy mmap Reverse Access | numpy | 0.001 | 4.0 | 0.0 | 0.0 |  |
| NumPy mmap Single Random Access | numpy | 0.001 | 0.0 | 0.0 | 0.0 |  |
| NumPy mmap Stride Access | numpy | 0.001 | 7.9 | 0.0 | 0.0 |  |

## NumPy memmap

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy memmap Batch Random Access | numpy | 0.001 | 10.5 | 0.0 | 0.0 |  |
| NumPy memmap Random Access | numpy | 0.001 | 0.0 | 0.0 | 0.0 |  |

## NumPack Stream

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Stream Read | python | 0.003 | 2.0 | 0.0 | 0.0 |  |
| NumPack Stream Read | rust | 0.003 | 0.1 | 0.0 | 0.0 |  |

## NumPack Append

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Append | python | 0.005 | 0.0 | 0.0 | 0.0 |  |
| NumPack Append | rust | 0.005 | 0.0 | 0.0 | 0.0 |  |

## NumPack Replace

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Replace | python | 0.001 | 1.3 | 0.0 | 0.0 |  |
| NumPack Replace | rust | 0.001 | 0.0 | 0.0 | 0.0 |  |

## NumPack Delete

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Delete | python | 0.011 | 5.0 | 0.0 | 0.0 |  |
| NumPack Delete | rust | 0.009 | 0.0 | 0.0 | 0.0 |  |

## NumPy Append

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Append (vstack) | numpy | 0.002 | 0.0 | 0.0 | 0.0 |  |

## NumPy Replace

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Replace | numpy | 0.000 | 0.0 | 0.0 | 0.0 |  |

## NumPy Delete

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Delete | numpy | 0.002 | 0.0 | 0.0 | 0.0 |  |

## NumPy Full

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Full Load Memory | numpy | 0.000 | 0.0 | 0.0 | 0.0 | data_size_mb=19.07, memory_ratio=0.00 |

## NumPack Stride

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Stride Access | python | 0.001 | 0.2 | 0.0 | 0.0 |  |
| NumPack Stride Access | rust | 0.001 | 0.0 | 0.0 | 0.0 |  |

## NumPack Reverse

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Reverse Access | python | 0.001 | 0.0 | 0.0 | 0.0 |  |
| NumPack Reverse Access | rust | 0.001 | 0.0 | 0.0 | 0.0 |  |

## NumPack Fibonacci

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Fibonacci Access | python | 0.000 | 0.0 | 0.0 | 0.0 |  |
| NumPack Fibonacci Access | rust | 0.000 | 0.0 | 0.0 | 0.0 |  |

## NumPy Save

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Save float32 | numpy | 0.003 | 0.0 | 9.5 | 3781.2 |  |
| NumPy Save float64 | numpy | 0.005 | 0.0 | 19.1 | 3608.8 |  |
| NumPy Save int32 | numpy | 0.002 | 0.0 | 9.5 | 3884.8 |  |
| NumPy Save int64 | numpy | 0.005 | 0.0 | 19.1 | 3975.7 |  |
| NumPy Save uint8 | numpy | 0.001 | 0.0 | 2.4 | 2603.7 |  |

## NumPy Load

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Load float32 | numpy | 0.002 | 0.0 | 0.0 | 4077.1 |  |
| NumPy Load float64 | numpy | 0.004 | 0.0 | 0.0 | 5091.1 |  |
| NumPy Load int32 | numpy | 0.002 | 0.0 | 0.0 | 4631.2 |  |
| NumPy Load int64 | numpy | 0.004 | 0.0 | 0.0 | 4561.5 |  |
| NumPy Load uint8 | numpy | 0.001 | 2.4 | 0.0 | 3414.3 |  |

## NumPack Compression

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Compression Random Data | python | 0.000 | 0.0 | 9.5 | 0.0 | original_size_mb=9.54, compression_ratio=1.00 |
| NumPack Compression Random Data | rust | 0.000 | 0.0 | 9.5 | 0.0 | original_size_mb=9.54, compression_ratio=1.00 |
| NumPack Compression Repeated Pattern | python | 0.000 | 0.0 | 9.5 | 0.0 | original_size_mb=9.54, compression_ratio=1.00 |
| NumPack Compression Repeated Pattern | rust | 0.000 | 0.0 | 9.5 | 0.0 | original_size_mb=9.54, compression_ratio=1.00 |
| NumPack Compression Sparse Data | python | 0.000 | 0.0 | 9.5 | 0.0 | original_size_mb=9.54, compression_ratio=1.00 |
| NumPack Compression Sparse Data | rust | 0.000 | 0.0 | 9.5 | 0.0 | original_size_mb=9.54, compression_ratio=1.00 |

## 性能总结

### IO 性能总结
- 最快保存: NumPy Save uint8 (numpy) - 0.001秒
- 最快加载: NumPack Lazy Load Memory (python) - 0.000秒
- 最高保存吞吐量: NumPy Save int64 (numpy) - 3975.7MB/s
- 最高加载吞吐量: NumPack Lazy Load (python) - 208169.0MB/s

### 随机访问性能总结
- 最快随机访问: NumPack Fibonacci Access (python) - 0.000秒
- 最快非连续访问: NumPack Fibonacci Access (python) - 0.000秒

### 内存效率总结
- 最节省内存: NumPy In-Memory Random Access (numpy) - 0.0MB
  - NumPack Lazy Load Memory (python): 内存占用比 0.00
  - NumPack Lazy Load Memory (rust): 内存占用比 0.00
  - NumPy Full Load Memory (numpy): 内存占用比 0.00
  - NumPy mmap Load Memory (numpy): 内存占用比 0.00

### 压缩效率总结
- 最佳压缩比: NumPy .npz Compression Repeated Pattern (numpy) - 255.44x
  - Random Data 最佳压缩: NumPy .npz (numpy) - 1.08x
  - Repeated Pattern 最佳压缩: NumPy .npz (numpy) - 255.44x
  - Sparse Data 最佳压缩: NumPy .npz (numpy) - 13.35x

### 矩阵运算性能总结
- 最快矩阵运算: NumPack Lazy Matrix Inner (rust) - 0.000秒

### 数据类型性能总结
- 最快数据类型保存: NumPy Save uint8 (numpy) - 0.001秒
- 最快数据类型加载: NumPack Load uint8 (rust) - 0.001秒

---
报告生成时间: 2025-07-22 18:13:39
