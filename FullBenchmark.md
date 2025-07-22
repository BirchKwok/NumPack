# NumPack vs NumPy 性能基准测试报告

测试环境信息:
- Python 版本: 3.12.4 | packaged by Anaconda, Inc. | (main, Jun 18 2024, 10:07:17) [Clang 14.0.6 ]
- NumPy 版本: 2.3.1
- 系统信息: posix.uname_result(sysname='Darwin', nodename='guobingmingdeMacBook-Pro-3.local', release='24.5.0', version='Darwin Kernel Version 24.5.0: Tue Apr 22 19:54:49 PDT 2025; root:xnu-11417.121.6~2/RELEASE_ARM64_T6000', machine='arm64')
- CPU 核心数: 10
- 总内存: 32.0 GB

## NumPack Save

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Save | python | 0.573 | 1128.1 | 1144.4 | 1995.5 |  |
| NumPack Save | rust | 0.715 | 238.2 | 1144.4 | 1601.3 |  |
| NumPack Save float32 | python | 0.096 | 0.0 | 190.7 | 1985.6 |  |
| NumPack Save float32 | rust | 0.091 | 0.0 | 190.7 | 2098.4 |  |
| NumPack Save float64 | python | 0.182 | 0.0 | 381.5 | 2099.8 |  |
| NumPack Save float64 | rust | 0.293 | 0.0 | 381.5 | 1303.2 |  |
| NumPack Save int32 | python | 0.092 | 0.0 | 190.7 | 2079.6 |  |
| NumPack Save int32 | rust | 0.100 | 0.9 | 190.7 | 1898.9 |  |
| NumPack Save int64 | python | 0.180 | 0.0 | 381.5 | 2121.5 |  |
| NumPack Save int64 | rust | 0.179 | 0.0 | 381.5 | 2125.5 |  |
| NumPack Save uint8 | python | 0.032 | 0.0 | 47.7 | 1481.0 |  |
| NumPack Save uint8 | rust | 0.032 | 0.0 | 47.7 | 1503.8 |  |

## NumPack Load

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Load | python | 0.223 | 0.0 | 0.0 | 5129.8 |  |
| NumPack Load | rust | 0.303 | 42.5 | 0.0 | 3781.6 |  |
| NumPack Load float32 | python | 0.034 | 0.0 | 0.0 | 5572.4 |  |
| NumPack Load float32 | rust | 0.041 | 0.0 | 0.0 | 4668.9 |  |
| NumPack Load float64 | python | 0.076 | 0.0 | 0.0 | 5046.3 |  |
| NumPack Load float64 | rust | 0.076 | 0.0 | 0.0 | 4999.7 |  |
| NumPack Load int32 | python | 0.042 | 0.0 | 0.0 | 4559.4 |  |
| NumPack Load int32 | rust | 0.040 | 0.0 | 0.0 | 4812.0 |  |
| NumPack Load int64 | python | 0.072 | 0.0 | 0.0 | 5331.1 |  |
| NumPack Load int64 | rust | 0.075 | 0.0 | 0.0 | 5070.9 |  |
| NumPack Load uint8 | python | 0.013 | 0.0 | 0.0 | 3598.4 |  |
| NumPack Load uint8 | rust | 0.009 | 0.0 | 0.0 | 5333.9 |  |

## NumPack Lazy

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Lazy Load | python | 0.010 | 0.0 | 0.0 | 109390.7 |  |
| NumPack Lazy Load | rust | 0.011 | 0.0 | 0.0 | 107553.0 |  |
| NumPack Lazy Load Memory | python | 0.000 | 0.0 | 0.0 | 0.0 | data_size_mb=381.47, memory_ratio=0.00 |
| NumPack Lazy Load Memory | rust | 0.000 | 0.0 | 0.0 | 0.0 | data_size_mb=381.47, memory_ratio=0.00 |
| NumPack Lazy Matrix Dot | python | 0.021 | 244.5 | 0.0 | 0.0 |  |
| NumPack Lazy Matrix Dot | rust | 0.020 | 244.2 | 0.0 | 0.0 |  |
| NumPack Lazy Matrix Inner | python | 0.006 | 2.0 | 0.0 | 0.0 |  |
| NumPack Lazy Matrix Inner | rust | 0.006 | 0.0 | 0.0 | 0.0 |  |

## NumPy .npy

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy .npy Compression Random Data | numpy | 0.000 | 0.0 | 190.7 | 0.0 | original_size_mb=190.73, compression_ratio=1.00 |
| NumPy .npy Compression Repeated Pattern | numpy | 0.000 | 0.0 | 190.7 | 0.0 | original_size_mb=190.73, compression_ratio=1.00 |
| NumPy .npy Compression Sparse Data | numpy | 0.000 | 0.0 | 190.7 | 0.0 | original_size_mb=190.73, compression_ratio=1.00 |
| NumPy .npy Load | numpy | 0.207 | 9.9 | 0.0 | 5540.2 |  |
| NumPy .npy Save | numpy | 0.371 | 65.6 | 1144.4 | 3083.1 |  |
| NumPy .npy mmap Load | numpy | 0.012 | 0.3 | 0.0 | 93946.8 |  |

## NumPy .npz

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy .npz Compression Random Data | numpy | 0.000 | 0.0 | 176.7 | 0.0 | original_size_mb=190.73, compression_ratio=1.08 |
| NumPy .npz Compression Repeated Pattern | numpy | 0.000 | 0.0 | 0.8 | 0.0 | original_size_mb=190.73, compression_ratio=229.04 |
| NumPy .npz Compression Sparse Data | numpy | 0.000 | 0.0 | 14.3 | 0.0 | original_size_mb=190.73, compression_ratio=13.36 |
| NumPy .npz Load | numpy | 0.332 | 198.4 | 0.0 | 3448.1 |  |
| NumPy .npz Save | numpy | 0.409 | 0.0 | 1144.4 | 2798.7 |  |
| NumPy .npz mmap Load | numpy | 0.324 | 50.4 | 0.0 | 3533.5 |  |

## NumPack Single

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Single Random Access | python | 0.030 | 0.0 | 0.0 | 0.0 |  |
| NumPack Single Random Access | rust | 0.030 | 0.0 | 0.0 | 0.0 |  |

## NumPack Batch

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Batch Random Access | python | 0.012 | 7.7 | 0.0 | 0.0 |  |
| NumPack Batch Random Access | rust | 0.011 | 3.6 | 0.0 | 0.0 |  |

## NumPy In-Memory

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy In-Memory Batch Random Access | numpy | 0.001 | 0.0 | 0.0 | 0.0 |  |
| NumPy In-Memory Matrix Dot | numpy | 0.005 | 0.0 | 0.0 | 0.0 |  |
| NumPy In-Memory Matrix Inner | numpy | 0.006 | 0.0 | 0.0 | 0.0 |  |
| NumPy In-Memory Random Access | numpy | 0.000 | 0.0 | 0.0 | 0.0 |  |

## NumPy mmap

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy mmap Batch Random Access | numpy | 0.013 | 131.1 | 0.0 | 0.0 |  |
| NumPy mmap Chunk Read | numpy | 0.000 | 0.0 | 0.0 | 0.0 |  |
| NumPy mmap Fibonacci Access | numpy | 0.000 | 0.2 | 0.0 | 0.0 |  |
| NumPy mmap Load Memory | numpy | 0.000 | 0.0 | 0.0 | 0.0 | data_size_mb=381.47, memory_ratio=0.00 |
| NumPy mmap Matrix Dot | numpy | 0.032 | 244.2 | 0.0 | 0.0 |  |
| NumPy mmap Matrix Inner | numpy | 0.032 | 0.0 | 0.0 | 0.0 |  |
| NumPy mmap Reverse Access | numpy | 0.003 | 41.2 | 0.0 | 0.0 |  |
| NumPy mmap Single Random Access | numpy | 0.001 | 0.0 | 0.0 | 0.0 |  |
| NumPy mmap Stride Access | numpy | 0.006 | 80.0 | 0.0 | 0.0 |  |

## NumPy memmap

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy memmap Batch Random Access | numpy | 0.009 | 130.9 | 0.0 | 0.0 |  |
| NumPy memmap Random Access | numpy | 0.001 | 0.0 | 0.0 | 0.0 |  |

## NumPack Stream

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Stream Read | python | 0.069 | 13.9 | 0.0 | 0.0 |  |
| NumPack Stream Read | rust | 0.074 | 11.0 | 0.0 | 0.0 |  |

## NumPack Append

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Append | python | 0.021 | 0.0 | 0.0 | 0.0 |  |
| NumPack Append | rust | 0.021 | 0.0 | 0.0 | 0.0 |  |

## NumPack Replace

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Replace | python | 0.015 | 38.6 | 0.0 | 0.0 |  |
| NumPack Replace | rust | 0.018 | 1.0 | 0.0 | 0.0 |  |

## NumPack Delete

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Delete | python | 0.136 | 20.7 | 0.0 | 0.0 |  |
| NumPack Delete | rust | 0.106 | 0.0 | 0.0 | 0.0 |  |

## NumPy Append

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Append (vstack) | numpy | 0.030 | 57.4 | 0.0 | 0.0 |  |

## NumPy Replace

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Replace | numpy | 0.001 | 0.0 | 0.0 | 0.0 |  |

## NumPy Delete

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Delete | numpy | 0.040 | 1.2 | 0.0 | 0.0 |  |

## NumPy Full

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Full Load Memory | numpy | 0.000 | 0.0 | 0.0 | 0.0 | data_size_mb=381.47, memory_ratio=0.00 |

## NumPack Stride

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Stride Access | python | 0.006 | 0.0 | 0.0 | 0.0 |  |
| NumPack Stride Access | rust | 0.006 | 0.0 | 0.0 | 0.0 |  |

## NumPack Reverse

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Reverse Access | python | 0.007 | 0.3 | 0.0 | 0.0 |  |
| NumPack Reverse Access | rust | 0.008 | 0.2 | 0.0 | 0.0 |  |

## NumPack Fibonacci

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Fibonacci Access | python | 0.001 | 0.0 | 0.0 | 0.0 |  |
| NumPack Fibonacci Access | rust | 0.000 | 0.0 | 0.0 | 0.0 |  |

## NumPy Save

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Save float32 | numpy | 0.061 | 0.0 | 190.7 | 3119.7 |  |
| NumPy Save float64 | numpy | 0.113 | 0.0 | 381.5 | 3364.4 |  |
| NumPy Save int32 | numpy | 0.064 | 0.0 | 190.7 | 2986.4 |  |
| NumPy Save int64 | numpy | 0.121 | 0.0 | 381.5 | 3143.9 |  |
| NumPy Save uint8 | numpy | 0.013 | 0.0 | 47.7 | 3787.0 |  |

## NumPy Load

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Load float32 | numpy | 0.035 | 0.0 | 0.0 | 5374.5 |  |
| NumPy Load float64 | numpy | 0.066 | 0.0 | 0.0 | 5784.0 |  |
| NumPy Load int32 | numpy | 0.037 | 0.0 | 0.0 | 5221.0 |  |
| NumPy Load int64 | numpy | 0.068 | 0.0 | 0.0 | 5621.3 |  |
| NumPy Load uint8 | numpy | 0.008 | 0.0 | 0.0 | 5651.3 |  |

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
- 最高保存吞吐量: NumPy Save uint8 (numpy) - 3787.0MB/s
- 最高加载吞吐量: NumPack Lazy Load (python) - 109390.7MB/s

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
- 最快数据类型加载: NumPy Load uint8 (numpy) - 0.008秒

---
报告生成时间: 2025-07-22 18:19:57
