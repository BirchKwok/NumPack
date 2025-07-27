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
| NumPack Save | python | 0.651 | 1161.4 | 1144.4 | 1757.3 |  |
| NumPack Save | rust | 0.912 | 387.0 | 1144.4 | 1255.5 |  |
| NumPack Save float32 | python | 0.099 | 1.3 | 190.7 | 1922.7 |  |
| NumPack Save float32 | rust | 0.088 | 0.0 | 190.7 | 2165.4 |  |
| NumPack Save float64 | python | 0.203 | 0.0 | 381.5 | 1878.9 |  |
| NumPack Save float64 | rust | 0.155 | 0.0 | 381.5 | 2465.5 |  |
| NumPack Save int32 | python | 0.082 | 0.0 | 190.7 | 2312.2 |  |
| NumPack Save int32 | rust | 0.090 | 0.0 | 190.7 | 2111.2 |  |
| NumPack Save int64 | python | 0.164 | 0.0 | 381.5 | 2332.6 |  |
| NumPack Save int64 | rust | 0.168 | 0.0 | 381.5 | 2270.5 |  |
| NumPack Save uint8 | python | 0.031 | 0.0 | 47.7 | 1533.8 |  |
| NumPack Save uint8 | rust | 0.025 | 0.0 | 47.7 | 1938.7 |  |

## NumPack Load

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Load | python | 0.246 | 0.0 | 0.0 | 4657.1 |  |
| NumPack Load | rust | 0.250 | 0.0 | 0.0 | 4580.0 |  |
| NumPack Load float32 | python | 0.036 | 0.0 | 0.0 | 5304.0 |  |
| NumPack Load float32 | rust | 0.041 | 0.0 | 0.0 | 4619.4 |  |
| NumPack Load float64 | python | 0.082 | 0.0 | 0.0 | 4677.4 |  |
| NumPack Load float64 | rust | 0.091 | 0.0 | 0.0 | 4185.2 |  |
| NumPack Load int32 | python | 0.042 | 0.0 | 0.0 | 4522.0 |  |
| NumPack Load int32 | rust | 0.045 | 0.0 | 0.0 | 4235.5 |  |
| NumPack Load int64 | python | 0.082 | 0.0 | 0.0 | 4639.2 |  |
| NumPack Load int64 | rust | 0.082 | 0.0 | 0.0 | 4641.7 |  |
| NumPack Load uint8 | python | 0.015 | 0.0 | 0.0 | 3143.9 |  |
| NumPack Load uint8 | rust | 0.010 | 0.0 | 0.0 | 4683.8 |  |

## NumPack Lazy

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Lazy Load | python | 0.011 | 0.0 | 0.0 | 108598.8 |  |
| NumPack Lazy Load | rust | 0.011 | 0.0 | 0.0 | 102516.5 |  |
| NumPack Lazy Load Memory | python | 0.000 | 0.0 | 0.0 | 0.0 | data_size_mb=381.47, memory_ratio=0.00 |
| NumPack Lazy Load Memory | rust | 0.000 | 0.0 | 0.0 | 0.0 | data_size_mb=381.47, memory_ratio=0.00 |
| NumPack Lazy Matrix Dot | python | 0.022 | 244.7 | 0.0 | 0.0 |  |
| NumPack Lazy Matrix Dot | rust | 0.022 | 244.2 | 0.0 | 0.0 |  |
| NumPack Lazy Matrix Inner | python | 0.006 | 1.9 | 0.0 | 0.0 |  |
| NumPack Lazy Matrix Inner | rust | 0.006 | 0.0 | 0.0 | 0.0 |  |

## NumPy .npy

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy .npy Compression Random Data | numpy | 0.000 | 0.0 | 190.7 | 0.0 | original_size_mb=190.73, compression_ratio=1.00 |
| NumPy .npy Compression Repeated Pattern | numpy | 0.000 | 0.0 | 190.7 | 0.0 | original_size_mb=190.73, compression_ratio=1.00 |
| NumPy .npy Compression Sparse Data | numpy | 0.000 | 0.0 | 190.7 | 0.0 | original_size_mb=190.73, compression_ratio=1.00 |
| NumPy .npy Load | numpy | 0.210 | 0.1 | 0.0 | 5443.4 |  |
| NumPy .npy Save | numpy | 0.392 | 0.0 | 1144.4 | 2921.2 |  |
| NumPy .npy mmap Load | numpy | 0.016 | 0.1 | 0.0 | 70173.1 |  |

## NumPy .npz

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy .npz Compression Random Data | numpy | 0.000 | 0.0 | 176.7 | 0.0 | original_size_mb=190.73, compression_ratio=1.08 |
| NumPy .npz Compression Repeated Pattern | numpy | 0.000 | 0.0 | 0.8 | 0.0 | original_size_mb=190.73, compression_ratio=229.04 |
| NumPy .npz Compression Sparse Data | numpy | 0.000 | 0.0 | 14.3 | 0.0 | original_size_mb=190.73, compression_ratio=13.36 |
| NumPy .npz Load | numpy | 0.336 | 3.3 | 0.0 | 3402.5 |  |
| NumPy .npz Save | numpy | 0.465 | 16.3 | 1144.4 | 2462.7 |  |
| NumPy .npz mmap Load | numpy | 0.329 | 0.0 | 0.0 | 3477.4 |  |

## NumPack Single

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Single Random Access | python | 0.039 | 0.0 | 0.0 | 0.0 |  |
| NumPack Single Random Access | rust | 0.039 | 0.0 | 0.0 | 0.0 |  |

## NumPack Batch

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Batch Random Access | python | 0.013 | 7.9 | 0.0 | 0.0 |  |
| NumPack Batch Random Access | rust | 0.012 | 0.0 | 0.0 | 0.0 |  |

## NumPy In-Memory

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy In-Memory Batch Random Access | numpy | 0.002 | 0.0 | 0.0 | 0.0 |  |
| NumPy In-Memory Matrix Dot | numpy | 0.005 | 0.0 | 0.0 | 0.0 |  |
| NumPy In-Memory Matrix Inner | numpy | 0.006 | 0.0 | 0.0 | 0.0 |  |
| NumPy In-Memory Random Access | numpy | 0.000 | 0.0 | 0.0 | 0.0 |  |

## NumPy mmap

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy mmap Batch Random Access | numpy | 0.014 | 130.3 | 0.0 | 0.0 |  |
| NumPy mmap Chunk Read | numpy | 0.000 | 0.0 | 0.0 | 0.0 |  |
| NumPy mmap Fibonacci Access | numpy | 0.000 | 0.2 | 0.0 | 0.0 |  |
| NumPy mmap Load Memory | numpy | 0.000 | 0.0 | 0.0 | 0.0 | data_size_mb=381.47, memory_ratio=0.00 |
| NumPy mmap Matrix Dot | numpy | 0.034 | 244.2 | 0.0 | 0.0 |  |
| NumPy mmap Matrix Inner | numpy | 0.032 | 0.0 | 0.0 | 0.0 |  |
| NumPy mmap Reverse Access | numpy | 0.004 | 41.2 | 0.0 | 0.0 |  |
| NumPy mmap Single Random Access | numpy | 0.001 | 0.0 | 0.0 | 0.0 |  |
| NumPy mmap Stride Access | numpy | 0.007 | 80.0 | 0.0 | 0.0 |  |

## NumPy memmap

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy memmap Batch Random Access | numpy | 0.010 | 130.1 | 0.0 | 0.0 |  |
| NumPy memmap Random Access | numpy | 0.001 | 0.0 | 0.0 | 0.0 |  |

## NumPack Stream

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Stream Read | python | 0.081 | 27.6 | 0.0 | 0.0 |  |
| NumPack Stream Read | rust | 0.081 | 0.0 | 0.0 | 0.0 |  |

## NumPack Append

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Append | python | 0.016 | 0.0 | 0.0 | 0.0 |  |
| NumPack Append | rust | 0.017 | 0.0 | 0.0 | 0.0 |  |

## NumPack Replace

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Replace | python | 0.022 | 47.7 | 0.0 | 0.0 |  |
| NumPack Replace | rust | 0.015 | 0.0 | 0.0 | 0.0 |  |

## NumPack Delete

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Delete | python | 0.091 | 57.2 | 0.0 | 0.0 |  |
| NumPack Delete | rust | 0.067 | 16.9 | 0.0 | 0.0 |  |

## NumPy Append

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Append (vstack) | numpy | 0.029 | 0.0 | 0.0 | 0.0 |  |

## NumPy Replace

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Replace | numpy | 0.001 | 0.0 | 0.0 | 0.0 |  |

## NumPy Delete

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Delete | numpy | 0.045 | 0.0 | 0.0 | 0.0 |  |

## NumPy Full

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Full Load Memory | numpy | 0.000 | 0.0 | 0.0 | 0.0 | data_size_mb=381.47, memory_ratio=0.00 |

## NumPack Stride

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Stride Access | python | 0.008 | 0.2 | 0.0 | 0.0 |  |
| NumPack Stride Access | rust | 0.008 | 0.0 | 0.0 | 0.0 |  |

## NumPack Reverse

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Reverse Access | python | 0.007 | 0.0 | 0.0 | 0.0 |  |
| NumPack Reverse Access | rust | 0.007 | 0.0 | 0.0 | 0.0 |  |

## NumPack Fibonacci

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Fibonacci Access | python | 0.000 | 0.0 | 0.0 | 0.0 |  |
| NumPack Fibonacci Access | rust | 0.000 | 0.0 | 0.0 | 0.0 |  |

## NumPy Save

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Save float32 | numpy | 0.058 | 0.0 | 190.7 | 3276.0 |  |
| NumPy Save float64 | numpy | 0.094 | 0.0 | 381.5 | 4064.1 |  |
| NumPy Save int32 | numpy | 0.062 | 0.0 | 190.7 | 3092.2 |  |
| NumPy Save int64 | numpy | 0.108 | 0.0 | 381.5 | 3536.6 |  |
| NumPy Save uint8 | numpy | 0.013 | 0.0 | 47.7 | 3540.0 |  |

## NumPy Load

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Load float32 | numpy | 0.040 | 0.0 | 0.0 | 4802.4 |  |
| NumPy Load float64 | numpy | 0.069 | 0.0 | 0.0 | 5542.5 |  |
| NumPy Load int32 | numpy | 0.039 | 0.0 | 0.0 | 4926.2 |  |
| NumPy Load int64 | numpy | 0.070 | 0.0 | 0.0 | 5430.1 |  |
| NumPy Load uint8 | numpy | 0.009 | 0.0 | 0.0 | 5441.0 |  |

## NumPack Compression

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Compression Random Data | python | 0.000 | 0.0 | 190.7 | 0.0 | original_size_mb=190.73, compression_ratio=1.00 |
| NumPack Compression Random Data | rust | 0.000 | 0.0 | 190.7 | 0.0 | original_size_mb=190.73, compression_ratio=1.00 |
| NumPack Compression Repeated Pattern | python | 0.000 | 0.0 | 190.7 | 0.0 | original_size_mb=190.73, compression_ratio=1.00 |
| NumPack Compression Repeated Pattern | rust | 0.000 | 0.0 | 190.7 | 0.0 | original_size_mb=190.73, compression_ratio=1.00 |
| NumPack Compression Sparse Data | python | 0.000 | 0.0 | 190.7 | 0.0 | original_size_mb=190.73, compression_ratio=1.00 |
| NumPack Compression Sparse Data | rust | 0.000 | 0.0 | 190.7 | 0.0 | original_size_mb=190.73, compression_ratio=1.00 |

## 性能总结

### IO 性能总结
- 最快保存: NumPy Save uint8 (numpy) - 0.013秒
- 最快加载: NumPack Lazy Load Memory (python) - 0.000秒
- 最高保存吞吐量: NumPy Save float64 (numpy) - 4064.1MB/s
- 最高加载吞吐量: NumPack Lazy Load (python) - 108598.8MB/s

### 随机访问性能总结
- 最快随机访问: NumPy mmap Fibonacci Access (numpy) - 0.000秒
- 最快非连续访问: NumPy mmap Fibonacci Access (numpy) - 0.000秒

### 内存效率总结
- 最节省内存: NumPy In-Memory Random Access (numpy) - 0.0MB
  - NumPack Lazy Load Memory (python): 内存占用比 0.00
  - NumPack Lazy Load Memory (rust): 内存占用比 0.00
  - NumPy Full Load Memory (numpy): 内存占用比 0.00
  - NumPy mmap Load Memory (numpy): 内存占用比 0.00

### 压缩效率总结
- 最佳压缩比: NumPy .npz Compression Repeated Pattern (numpy) - 229.04x
  - Random Data 最佳压缩: NumPy .npz (numpy) - 1.08x
  - Repeated Pattern 最佳压缩: NumPy .npz (numpy) - 229.04x
  - Sparse Data 最佳压缩: NumPy .npz (numpy) - 13.36x

### 矩阵运算性能总结
- 最快矩阵运算: NumPy In-Memory Matrix Dot (numpy) - 0.005秒

### 数据类型性能总结
- 最快数据类型保存: NumPy Save uint8 (numpy) - 0.013秒
- 最快数据类型加载: NumPy Load uint8 (numpy) - 0.009秒

---
报告生成时间: 2025-07-27 15:47:00
