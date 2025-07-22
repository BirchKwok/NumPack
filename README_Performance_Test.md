# NumPack vs NumPy 性能基准测试报告

测试环境信息:
- Python 版本: 3.11.10 (main, Oct  3 2024, 02:26:51) [Clang 14.0.6 ]
- NumPy 版本: 2.3.1
- 系统信息: posix.uname_result(sysname='Darwin', nodename='guobingmingdeMacBook-Pro-3.local', release='24.5.0', version='Darwin Kernel Version 24.5.0: Tue Apr 22 19:54:49 PDT 2025; root:xnu-11417.121.6~2/RELEASE_ARM64_T6000', machine='arm64')
- CPU 核心数: 10
- 总内存: 32.0 GB

## NumPack Save

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Save | python | 0.463 | 1163.0 | 1144.4 | 2471.6 |  |
| NumPack Save | rust | 0.411 | 387.0 | 1144.4 | 2786.2 |  |
| NumPack Save float32 | python | 0.077 | 0.0 | 190.7 | 2469.3 |  |
| NumPack Save float32 | rust | 0.079 | 0.0 | 190.7 | 2428.1 |  |
| NumPack Save float64 | python | 0.133 | 0.0 | 381.5 | 2861.4 |  |
| NumPack Save float64 | rust | 0.135 | 0.0 | 381.5 | 2819.6 |  |
| NumPack Save int32 | python | 0.071 | 0.0 | 190.7 | 2668.7 |  |
| NumPack Save int32 | rust | 0.073 | 1.1 | 190.7 | 2614.9 |  |
| NumPack Save int64 | python | 0.134 | 0.0 | 381.5 | 2847.1 |  |
| NumPack Save int64 | rust | 0.161 | 0.0 | 381.5 | 2365.9 |  |
| NumPack Save uint8 | python | 0.029 | 7.7 | 47.7 | 1645.2 |  |
| NumPack Save uint8 | rust | 0.027 | 0.0 | 47.7 | 1770.5 |  |

## NumPack Load

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Load | python | 0.223 | 0.0 | 0.0 | 5131.5 |  |
| NumPack Load | rust | 0.228 | 0.0 | 0.0 | 5015.1 |  |
| NumPack Load float32 | python | 0.034 | 0.0 | 0.0 | 5671.3 |  |
| NumPack Load float32 | rust | 0.041 | 0.0 | 0.0 | 4707.8 |  |
| NumPack Load float64 | python | 0.075 | 0.0 | 0.0 | 5073.2 |  |
| NumPack Load float64 | rust | 0.075 | 0.0 | 0.0 | 5101.8 |  |
| NumPack Load int32 | python | 0.039 | 0.0 | 0.0 | 4840.3 |  |
| NumPack Load int32 | rust | 0.040 | 0.0 | 0.0 | 4763.9 |  |
| NumPack Load int64 | python | 0.074 | 0.0 | 0.0 | 5164.4 |  |
| NumPack Load int64 | rust | 0.076 | 0.0 | 0.0 | 5011.5 |  |
| NumPack Load uint8 | python | 0.014 | 0.0 | 0.0 | 3377.3 |  |
| NumPack Load uint8 | rust | 0.009 | 0.0 | 0.0 | 5129.3 |  |

## NumPack Lazy

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Lazy Load | python | 0.011 | 0.0 | 0.0 | 102830.9 |  |
| NumPack Lazy Load | rust | 0.010 | 0.0 | 0.0 | 112494.3 |  |
| NumPack Lazy Load Memory | python | 0.000 | 0.0 | 0.0 | 0.0 | data_size_mb=381.47, memory_ratio=0.00 |
| NumPack Lazy Load Memory | rust | 0.000 | 0.0 | 0.0 | 0.0 | data_size_mb=381.47, memory_ratio=0.00 |
| NumPack Lazy Matrix Dot | python | 0.014 | 244.5 | 0.0 | 0.0 |  |
| NumPack Lazy Matrix Dot | rust | 0.015 | 244.1 | 0.0 | 0.0 |  |
| NumPack Lazy Matrix Inner | python | 0.006 | 0.0 | 0.0 | 0.0 |  |
| NumPack Lazy Matrix Inner | rust | 0.006 | 0.0 | 0.0 | 0.0 |  |

## NumPy .npy

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy .npy Compression Random Data | numpy | 0.000 | 0.0 | 190.7 | 0.0 | original_size_mb=190.73, compression_ratio=1.00 |
| NumPy .npy Compression Repeated Pattern | numpy | 0.000 | 0.0 | 190.7 | 0.0 | original_size_mb=190.73, compression_ratio=1.00 |
| NumPy .npy Compression Sparse Data | numpy | 0.000 | 0.0 | 190.7 | 0.0 | original_size_mb=190.73, compression_ratio=1.00 |
| NumPy .npy Load | numpy | 0.203 | 0.0 | 0.0 | 5647.4 |  |
| NumPy .npy Save | numpy | 0.185 | 0.0 | 1144.4 | 6182.7 |  |
| NumPy .npy mmap Load | numpy | 0.009 | 0.1 | 0.0 | 131962.8 |  |

## NumPy .npz

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy .npz Compression Random Data | numpy | 0.000 | 0.0 | 176.7 | 0.0 | original_size_mb=190.73, compression_ratio=1.08 |
| NumPy .npz Compression Repeated Pattern | numpy | 0.000 | 0.0 | 0.8 | 0.0 | original_size_mb=190.73, compression_ratio=229.04 |
| NumPy .npz Compression Sparse Data | numpy | 0.000 | 0.0 | 14.3 | 0.0 | original_size_mb=190.73, compression_ratio=13.36 |
| NumPy .npz Load | numpy | 0.309 | 2.5 | 0.0 | 3703.0 |  |
| NumPy .npz Save | numpy | 0.282 | 16.3 | 1144.4 | 4060.5 |  |
| NumPy .npz mmap Load | numpy | 0.305 | 1.1 | 0.0 | 3751.2 |  |

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
| NumPy In-Memory Batch Random Access | numpy | 0.001 | 0.0 | 0.0 | 0.0 |  |
| NumPy In-Memory Matrix Dot | numpy | 0.006 | 0.0 | 0.0 | 0.0 |  |
| NumPy In-Memory Matrix Inner | numpy | 0.005 | 0.0 | 0.0 | 0.0 |  |
| NumPy In-Memory Random Access | numpy | 0.000 | 0.0 | 0.0 | 0.0 |  |

## NumPy mmap

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy mmap Batch Random Access | numpy | 0.010 | 131.9 | 0.0 | 0.0 |  |
| NumPy mmap Chunk Read | numpy | 0.000 | 0.0 | 0.0 | 0.0 |  |
| NumPy mmap Fibonacci Access | numpy | 0.000 | 0.2 | 0.0 | 0.0 |  |
| NumPy mmap Load Memory | numpy | 0.000 | 0.0 | 0.0 | 0.0 | data_size_mb=381.47, memory_ratio=0.00 |
| NumPy mmap Matrix Dot | numpy | 0.032 | 244.2 | 0.0 | 0.0 |  |
| NumPy mmap Matrix Inner | numpy | 0.032 | 0.0 | 0.0 | 0.0 |  |
| NumPy mmap Reverse Access | numpy | 0.004 | 41.2 | 0.0 | 0.0 |  |
| NumPy mmap Single Random Access | numpy | 0.001 | 0.0 | 0.0 | 0.0 |  |
| NumPy mmap Stride Access | numpy | 0.006 | 80.0 | 0.0 | 0.0 |  |

## NumPy memmap

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy memmap Batch Random Access | numpy | 0.009 | 131.6 | 0.0 | 0.0 |  |
| NumPy memmap Random Access | numpy | 0.001 | 0.0 | 0.0 | 0.0 |  |

## NumPack Stream

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Stream Read | python | 0.073 | 8.4 | 0.0 | 0.0 |  |
| NumPack Stream Read | rust | 0.073 | 7.7 | 0.0 | 0.0 |  |

## NumPack Append

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Append | python | 0.014 | 0.0 | 0.0 | 0.0 |  |
| NumPack Append | rust | 0.012 | 0.0 | 0.0 | 0.0 |  |

## NumPack Replace

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Replace | python | 0.023 | 47.7 | 0.0 | 0.0 |  |
| NumPack Replace | rust | 0.022 | 0.0 | 0.0 | 0.0 |  |

## NumPack Delete

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Delete | python | 0.089 | 74.2 | 0.0 | 0.0 |  |
| NumPack Delete | rust | 0.106 | 23.2 | 0.0 | 0.0 |  |

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
| NumPy Delete | numpy | 0.041 | 0.0 | 0.0 | 0.0 |  |

## NumPy Full

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Full Load Memory | numpy | 0.000 | 0.0 | 0.0 | 0.0 | data_size_mb=381.47, memory_ratio=0.00 |

## NumPack Stride

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Stride Access | python | 0.007 | 0.0 | 0.0 | 0.0 |  |
| NumPack Stride Access | rust | 0.007 | 0.0 | 0.0 | 0.0 |  |

## NumPack Reverse

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Reverse Access | python | 0.007 | 0.0 | 0.0 | 0.0 |  |
| NumPack Reverse Access | rust | 0.007 | 0.0 | 0.0 | 0.0 |  |

## NumPack Fibonacci

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPack Fibonacci Access | python | 0.000 | 0.0 | 0.0 | 0.0 |  |
| NumPack Fibonacci Access | rust | 0.001 | 0.0 | 0.0 | 0.0 |  |

## NumPy Save

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Save float32 | numpy | 0.108 | 0.0 | 190.7 | 1765.8 |  |
| NumPy Save float64 | numpy | 0.062 | 0.0 | 381.5 | 6150.5 |  |
| NumPy Save int32 | numpy | 0.032 | 0.0 | 190.7 | 5977.5 |  |
| NumPy Save int64 | numpy | 0.065 | 0.0 | 381.5 | 5893.3 |  |
| NumPy Save uint8 | numpy | 0.008 | 0.0 | 47.7 | 5648.4 |  |

## NumPy Load

| 操作 | 后端 | 时间(秒) | 内存使用(MB) | 文件大小(MB) | 吞吐量(MB/s) | 额外信息 |
|------|------|----------|--------------|--------------|-------------|----------|
| NumPy Load float32 | numpy | 0.037 | 0.0 | 0.0 | 5113.3 |  |
| NumPy Load float64 | numpy | 0.066 | 0.0 | 0.0 | 5771.2 |  |
| NumPy Load int32 | numpy | 0.036 | 0.0 | 0.0 | 5249.7 |  |
| NumPy Load int64 | numpy | 0.067 | 0.0 | 0.0 | 5666.7 |  |
| NumPy Load uint8 | numpy | 0.009 | 0.0 | 0.0 | 5118.4 |  |

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
- 最快保存: NumPy Save uint8 (numpy) - 0.008秒
- 最快加载: NumPack Lazy Load Memory (python) - 0.000秒
- 最高保存吞吐量: NumPy .npy Save (numpy) - 6182.7MB/s
- 最高加载吞吐量: NumPy .npy mmap Load (numpy) - 131962.8MB/s

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
- 最快矩阵运算: NumPy In-Memory Matrix Inner (numpy) - 0.005秒

### 数据类型性能总结
- 最快数据类型保存: NumPy Save uint8 (numpy) - 0.008秒
- 最快数据类型加载: NumPack Load uint8 (rust) - 0.009秒

---
报告生成时间: 2025-07-22 18:21:25
