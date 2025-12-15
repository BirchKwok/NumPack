//! NumPack SIMD性能基准测试
//!
//! 专门测试NumPack的SIMD优化效果，验证不同访问模式和数据类型的性能提升

use super::{AccessPattern, AccessPatternSIMD, DataType, NumPackSIMD, SIMDStrategy};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// 基准测试结果
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub operation: String,
    pub data_type: DataType,
    pub data_size: usize,
    pub strategy: SIMDStrategy,
    pub access_pattern: AccessPattern,
    pub duration: Duration,
    pub throughput_mb_s: f64,
    pub speedup_factor: f64,
    pub cpu_utilization: f64,
}

/// SIMD基准测试套件
pub struct SIMDBenchmark {
    simd: NumPackSIMD,
    pattern_simd: AccessPatternSIMD,
}

impl SIMDBenchmark {
    pub fn new() -> Self {
        Self {
            simd: NumPackSIMD::new(),
            pattern_simd: AccessPatternSIMD::new(),
        }
    }

    /// 运行完整的SIMD基准测试套件
    pub fn run_comprehensive_benchmark(&mut self) -> Vec<BenchmarkResult> {
        let mut results = Vec::new();

        println!("开始NumPack SIMD性能基准测试...");

        // 1. 数据类型特异化测试
        results.extend(self.benchmark_data_type_specialization());

        // 2. 访问模式优化测试
        results.extend(self.benchmark_access_patterns());

        // 3. 行拷贝性能测试
        results.extend(self.benchmark_row_copy_performance());

        // 4. 数据转换性能测试
        results.extend(self.benchmark_data_conversion());

        // 5. 大规模数据测试
        results.extend(self.benchmark_large_scale_operations());

        // 6. 多平台兼容性测试
        results.extend(self.benchmark_cross_platform_compatibility());

        println!("SIMD基准测试完成，共{}个测试用例", results.len());
        results
    }

    /// 数据类型特异化基准测试
    fn benchmark_data_type_specialization(&mut self) -> Vec<BenchmarkResult> {
        println!("测试数据类型特异化SIMD优化...");
        let mut results = Vec::new();

        let data_types = vec![
            DataType::Bool,
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::Uint8,
            DataType::Uint16,
            DataType::Uint32,
            DataType::Uint64,
            DataType::Float16,
            DataType::Float32,
            DataType::Float64,
            DataType::Complex64,
            DataType::Complex128,
        ];

        let test_sizes = vec![1024, 4096, 16384, 65536];

        for &dtype in &data_types {
            for &size in &test_sizes {
                // 生成测试数据
                let test_data = self.generate_test_data(dtype, size);
                let indices = (0..size / 4).step_by(2).collect::<Vec<_>>();

                // 测试不同的SIMD策略
                let strategies = self.get_applicable_strategies(dtype);

                for strategy in strategies {
                    let result = self.benchmark_row_copy_operation(
                        dtype,
                        &test_data,
                        &indices,
                        strategy,
                        AccessPattern::BatchRandom,
                    );
                    results.push(result);
                }
            }
        }

        results
    }

    /// 访问模式优化基准测试
    fn benchmark_access_patterns(&mut self) -> Vec<BenchmarkResult> {
        println!("测试访问模式感知SIMD优化...");
        let mut results = Vec::new();

        let data_size = 32768;
        let dtype = DataType::Float32;
        let test_data = self.generate_test_data(dtype, data_size);

        // 测试不同的访问模式
        let access_patterns = vec![
            (AccessPattern::SingleRandom, vec![1024]),
            (
                AccessPattern::BatchRandom,
                (0..data_size).step_by(17).take(100).collect(),
            ),
            (AccessPattern::Sequential, (1000..1100).collect()),
            (
                AccessPattern::Strided,
                (0..data_size).step_by(16).take(50).collect(),
            ),
            (
                AccessPattern::Clustered,
                [
                    (100..120).collect::<Vec<_>>(),
                    (500..520).collect::<Vec<_>>(),
                    (1000..1020).collect::<Vec<_>>(),
                ]
                .concat(),
            ),
            (
                AccessPattern::Streaming,
                (0..data_size).step_by(1).take(8192).collect(),
            ),
        ];

        for (pattern, indices) in access_patterns {
            // 自动选择策略
            let auto_strategy =
                self.pattern_simd
                    .select_optimal_strategy(pattern, dtype, indices.len() * 4);

            // 与基础策略对比
            let base_strategy = self.simd.get_optimal_strategy(dtype, indices.len() * 4);

            let auto_result = self.benchmark_row_copy_operation(
                dtype,
                &test_data,
                &indices,
                auto_strategy,
                pattern,
            );
            let base_result = self.benchmark_row_copy_operation(
                dtype,
                &test_data,
                &indices,
                base_strategy,
                pattern,
            );

            results.push(auto_result);
            results.push(base_result);
        }

        results
    }

    /// 行拷贝性能基准测试
    fn benchmark_row_copy_performance(&mut self) -> Vec<BenchmarkResult> {
        println!("📋 测试行拷贝SIMD性能...");
        let mut results = Vec::new();

        let dtype = DataType::Float32;
        let row_sizes = vec![64, 256, 1024, 4096];
        let row_counts = vec![100, 500, 1000, 2000];

        for &row_size in &row_sizes {
            for &row_count in &row_counts {
                let total_size = row_size * row_count;
                let test_data = self.generate_test_data(dtype, total_size);
                let indices = (0..row_count).step_by(3).collect::<Vec<_>>();

                // 测试不同SIMD策略的行拷贝性能
                let strategies = vec![
                    SIMDStrategy::Scalar,
                    SIMDStrategy::SSE2DWord,
                    SIMDStrategy::AVX2DWord,
                    SIMDStrategy::AVX512DWord,
                ];

                for strategy in strategies {
                    if self.is_strategy_supported(strategy) {
                        let result = self.benchmark_row_copy_with_timing(
                            dtype, &test_data, &indices, row_size, strategy,
                        );
                        results.push(result);
                    }
                }
            }
        }

        results
    }

    /// 数据转换性能基准测试
    fn benchmark_data_conversion(&mut self) -> Vec<BenchmarkResult> {
        println!("🔄 测试数据转换SIMD性能...");
        let mut results = Vec::new();

        let conversion_pairs = vec![
            (DataType::Int32, DataType::Float32),
            (DataType::Int64, DataType::Float64),
            (DataType::Float32, DataType::Int32),
            (DataType::Int16, DataType::Float32),
            (DataType::Int8, DataType::Float32),
            (DataType::Uint32, DataType::Float32),
        ];

        let test_sizes = vec![1024, 4096, 16384];

        for (src_dtype, dst_dtype) in conversion_pairs {
            for &size in &test_sizes {
                let src_data = self.generate_test_data(src_dtype, size);
                let mut dst_data = vec![0u8; size * dst_dtype.size_bytes() as usize];

                // 测试SIMD vs 标量转换
                let simd_result = self.benchmark_conversion_operation(
                    &src_data,
                    &mut dst_data,
                    src_dtype,
                    dst_dtype,
                    true,
                );

                let scalar_result = self.benchmark_conversion_operation(
                    &src_data,
                    &mut dst_data,
                    src_dtype,
                    dst_dtype,
                    false,
                );

                results.push(simd_result);
                results.push(scalar_result);
            }
        }

        results
    }

    /// 大规模数据基准测试
    fn benchmark_large_scale_operations(&mut self) -> Vec<BenchmarkResult> {
        println!("🌊 测试大规模数据SIMD性能...");
        let mut results = Vec::new();

        let large_sizes = vec![1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]; // 1MB, 4MB, 16MB
        let dtype = DataType::Float32;

        for &size in &large_sizes {
            let test_data = self.generate_test_data(dtype, size);
            let indices = (0..size / 1024).step_by(4).collect::<Vec<_>>();

            // 测试流式访问性能
            let streaming_result = self.benchmark_row_copy_operation(
                dtype,
                &test_data,
                &indices,
                SIMDStrategy::AVX512DWord,
                AccessPattern::Streaming,
            );
            results.push(streaming_result);

            // 测试批量随机访问性能
            let random_indices = self.generate_random_indices(size, 1000);
            let batch_result = self.benchmark_row_copy_operation(
                dtype,
                &test_data,
                &random_indices,
                SIMDStrategy::AVX2DWord,
                AccessPattern::BatchRandom,
            );
            results.push(batch_result);
        }

        results
    }

    /// 多平台兼容性基准测试
    fn benchmark_cross_platform_compatibility(&mut self) -> Vec<BenchmarkResult> {
        println!("🌐 测试多平台SIMD兼容性...");
        let mut results = Vec::new();

        let dtype = DataType::Float32;
        let size = 8192;
        let test_data = self.generate_test_data(dtype, size);
        let indices = (0..size / 8).collect::<Vec<_>>();

        // 测试不同指令集的兼容性
        let platform_strategies = vec![
            ("Scalar", SIMDStrategy::Scalar),
            ("SSE2", SIMDStrategy::SSE2DWord),
            ("AVX2", SIMDStrategy::AVX2DWord),
            ("AVX512", SIMDStrategy::AVX512DWord),
            ("NEON", SIMDStrategy::NEONDWord),
        ];

        for (platform_name, strategy) in platform_strategies {
            if self.is_strategy_supported(strategy) {
                let mut result = self.benchmark_row_copy_operation(
                    dtype,
                    &test_data,
                    &indices,
                    strategy,
                    AccessPattern::BatchRandom,
                );
                result.operation = format!("Cross-Platform-{}", platform_name);
                results.push(result);
            }
        }

        results
    }

    /// 执行行拷贝操作的基准测试
    fn benchmark_row_copy_operation(
        &self,
        dtype: DataType,
        data: &[u8],
        indices: &[usize],
        strategy: SIMDStrategy,
        pattern: AccessPattern,
    ) -> BenchmarkResult {
        let item_size = dtype.size_bytes() as usize;
        let row_size = item_size * 16; // 假设每行16个元素
        let mut dst = vec![0u8; indices.len() * row_size];

        let start = Instant::now();

        // 根据策略执行操作
        let _ = match strategy {
            SIMDStrategy::Scalar => self
                .simd
                .scalar_copy_rows(data, &mut dst, indices, row_size),
            SIMDStrategy::AVX512DWord
            | SIMDStrategy::AVX512QWord
            | SIMDStrategy::AVX512Byte
            | SIMDStrategy::AVX512Word => self
                .simd
                .avx512_copy_rows(data, &mut dst, indices, row_size),
            SIMDStrategy::AVX2DWord
            | SIMDStrategy::AVX2QWord
            | SIMDStrategy::AVX2Byte
            | SIMDStrategy::AVX2Word => self.simd.avx2_copy_rows(data, &mut dst, indices, row_size),
            SIMDStrategy::SSE2DWord
            | SIMDStrategy::SSE2QWord
            | SIMDStrategy::SSE2Byte
            | SIMDStrategy::SSE2Word => self.simd.sse2_copy_rows(data, &mut dst, indices, row_size),
            SIMDStrategy::NEONDWord
            | SIMDStrategy::NEONQWord
            | SIMDStrategy::NEONByte
            | SIMDStrategy::NEONWord => self.simd.neon_copy_rows(data, &mut dst, indices, row_size),
            _ => self
                .simd
                .scalar_copy_rows(data, &mut dst, indices, row_size),
        };

        let duration = start.elapsed();
        let data_size_mb = (indices.len() * row_size) as f64 / (1024.0 * 1024.0);
        let throughput = data_size_mb / duration.as_secs_f64();

        BenchmarkResult {
            operation: format!("RowCopy-{:?}", strategy),
            data_type: dtype,
            data_size: indices.len() * row_size,
            strategy,
            access_pattern: pattern,
            duration,
            throughput_mb_s: throughput,
            speedup_factor: 1.0,  // 将在后处理中计算
            cpu_utilization: 0.0, // 简化实现，实际应用中可测量CPU使用率
        }
    }

    /// 执行行拷贝操作的详细计时基准测试
    fn benchmark_row_copy_with_timing(
        &self,
        dtype: DataType,
        data: &[u8],
        indices: &[usize],
        row_size: usize,
        strategy: SIMDStrategy,
    ) -> BenchmarkResult {
        let mut dst = vec![0u8; indices.len() * row_size];

        // 预热
        for _ in 0..3 {
            let _ = self
                .simd
                .scalar_copy_rows(data, &mut dst, indices, row_size);
        }

        let start = Instant::now();
        let iterations = 100;

        for _ in 0..iterations {
            let _ = match strategy {
                SIMDStrategy::Scalar => self
                    .simd
                    .scalar_copy_rows(data, &mut dst, indices, row_size),
                _ => self
                    .simd
                    .copy_rows(data, &mut dst, indices, row_size, dtype),
            };
        }

        let duration = start.elapsed() / iterations;
        let data_size_mb = (indices.len() * row_size) as f64 / (1024.0 * 1024.0);
        let throughput = data_size_mb / duration.as_secs_f64();

        BenchmarkResult {
            operation: format!("DetailedRowCopy-{:?}", strategy),
            data_type: dtype,
            data_size: indices.len() * row_size,
            strategy,
            access_pattern: AccessPattern::BatchRandom,
            duration,
            throughput_mb_s: throughput,
            speedup_factor: 1.0,
            cpu_utilization: 0.0,
        }
    }

    /// 执行数据转换操作的基准测试
    fn benchmark_conversion_operation(
        &self,
        src_data: &[u8],
        dst_data: &mut [u8],
        src_dtype: DataType,
        dst_dtype: DataType,
        use_simd: bool,
    ) -> BenchmarkResult {
        let start = Instant::now();
        let iterations = 50;

        for _ in 0..iterations {
            if use_simd {
                let _ = self
                    .simd
                    .batch_convert(src_data, dst_data, src_dtype, dst_dtype);
            } else {
                let _ = self
                    .simd
                    .scalar_convert(src_data, dst_data, src_dtype, dst_dtype);
            }
        }

        let duration = start.elapsed() / iterations;
        let data_size_mb = src_data.len() as f64 / (1024.0 * 1024.0);
        let throughput = data_size_mb / duration.as_secs_f64();

        BenchmarkResult {
            operation: format!(
                "Convert-{:?}-to-{:?}-{}",
                src_dtype,
                dst_dtype,
                if use_simd { "SIMD" } else { "Scalar" }
            ),
            data_type: src_dtype,
            data_size: src_data.len(),
            strategy: if use_simd {
                SIMDStrategy::AVX2DWord
            } else {
                SIMDStrategy::Scalar
            },
            access_pattern: AccessPattern::Sequential,
            duration,
            throughput_mb_s: throughput,
            speedup_factor: 1.0,
            cpu_utilization: 0.0,
        }
    }

    /// 生成测试数据
    fn generate_test_data(&self, dtype: DataType, element_count: usize) -> Vec<u8> {
        let item_size = dtype.size_bytes() as usize;
        let mut data = vec![0u8; element_count * item_size];

        // 填充测试数据
        for i in 0..data.len() {
            data[i] = (i % 256) as u8;
        }

        data
    }

    /// 生成随机索引
    fn generate_random_indices(&self, max_index: usize, count: usize) -> Vec<usize> {
        let mut indices = Vec::with_capacity(count);
        let step = max_index / count;

        for i in 0..count {
            indices.push((i * step + (i % 7) * 13) % max_index);
        }

        indices
    }

    /// 获取数据类型适用的SIMD策略
    fn get_applicable_strategies(&self, dtype: DataType) -> Vec<SIMDStrategy> {
        let mut strategies = vec![SIMDStrategy::Scalar];

        match dtype {
            DataType::Float32 | DataType::Int32 | DataType::Uint32 => {
                strategies.extend_from_slice(&[
                    SIMDStrategy::SSE2DWord,
                    SIMDStrategy::AVX2DWord,
                    SIMDStrategy::AVX512DWord,
                    SIMDStrategy::NEONDWord,
                ]);
            }
            DataType::Float64 | DataType::Int64 | DataType::Uint64 => {
                strategies.extend_from_slice(&[
                    SIMDStrategy::SSE2QWord,
                    SIMDStrategy::AVX2QWord,
                    SIMDStrategy::AVX512QWord,
                    SIMDStrategy::NEONQWord,
                ]);
            }
            DataType::Int8 | DataType::Uint8 => {
                strategies.extend_from_slice(&[
                    SIMDStrategy::SSE2Byte,
                    SIMDStrategy::AVX2Byte,
                    SIMDStrategy::AVX512Byte,
                    SIMDStrategy::NEONByte,
                ]);
            }
            DataType::Int16 | DataType::Uint16 | DataType::Float16 => {
                strategies.extend_from_slice(&[
                    SIMDStrategy::SSE2Word,
                    SIMDStrategy::AVX2Word,
                    SIMDStrategy::AVX512Word,
                    SIMDStrategy::NEONWord,
                ]);
            }
            DataType::Bool => {
                strategies.extend_from_slice(&[
                    SIMDStrategy::PackedBool,
                    SIMDStrategy::AVX2Bool,
                    SIMDStrategy::AVX512Bool,
                ]);
            }
            DataType::Complex64 => {
                strategies
                    .extend_from_slice(&[SIMDStrategy::SSE2Complex64, SIMDStrategy::AVX2Complex64]);
            }
            DataType::Complex128 => {
                strategies.extend_from_slice(&[
                    SIMDStrategy::SSE2Complex128,
                    SIMDStrategy::AVX2Complex128,
                ]);
            }
        }

        strategies
            .into_iter()
            .filter(|&s| self.is_strategy_supported(s))
            .collect()
    }

    /// 检查SIMD策略是否被支持
    fn is_strategy_supported(&self, strategy: SIMDStrategy) -> bool {
        match strategy {
            SIMDStrategy::Scalar => true,
            SIMDStrategy::SSE2Byte
            | SIMDStrategy::SSE2Word
            | SIMDStrategy::SSE2DWord
            | SIMDStrategy::SSE2QWord
            | SIMDStrategy::SSE2Complex64
            | SIMDStrategy::SSE2Complex128 => self.simd.capabilities.sse2,
            SIMDStrategy::AVX2Bool
            | SIMDStrategy::AVX2Byte
            | SIMDStrategy::AVX2Word
            | SIMDStrategy::AVX2DWord
            | SIMDStrategy::AVX2QWord
            | SIMDStrategy::AVX2Complex64
            | SIMDStrategy::AVX2Complex128 => self.simd.capabilities.avx2,
            SIMDStrategy::AVX512Bool
            | SIMDStrategy::AVX512Byte
            | SIMDStrategy::AVX512Word
            | SIMDStrategy::AVX512DWord
            | SIMDStrategy::AVX512QWord => self.simd.capabilities.avx512f,
            SIMDStrategy::NEONByte
            | SIMDStrategy::NEONWord
            | SIMDStrategy::NEONDWord
            | SIMDStrategy::NEONQWord => self.simd.capabilities.neon,
            SIMDStrategy::PackedBool => true,
        }
    }

    /// 计算加速比并生成报告
    pub fn generate_performance_report(&self, results: &mut [BenchmarkResult]) -> String {
        // 计算加速比
        let mut baseline_performance: HashMap<String, f64> = HashMap::new();

        // 收集标量基线性能
        for result in results.iter() {
            if result.strategy == SIMDStrategy::Scalar {
                let key = format!(
                    "{}-{:?}-{}",
                    result.operation.split('-').next().unwrap_or(""),
                    result.data_type,
                    result.data_size
                );
                baseline_performance.insert(key, result.throughput_mb_s);
            }
        }

        // 计算加速比
        for result in results.iter_mut() {
            let key = format!(
                "{}-{:?}-{}",
                result.operation.split('-').next().unwrap_or(""),
                result.data_type,
                result.data_size
            );
            if let Some(&baseline) = baseline_performance.get(&key) {
                result.speedup_factor = result.throughput_mb_s / baseline;
            }
        }

        // 生成报告
        let mut report = String::new();
        report.push_str("# NumPack SIMD性能基准测试报告\n\n");

        // 按操作类型分组
        let mut operations: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
        for result in results.iter() {
            let op_type = result
                .operation
                .split('-')
                .next()
                .unwrap_or("Unknown")
                .to_string();
            operations
                .entry(op_type)
                .or_insert_with(Vec::new)
                .push(result);
        }

        for (op_type, op_results) in operations {
            report.push_str(&format!("## {} Operation Performance\n\n", op_type));
            report.push_str("| 策略 | 数据类型 | 大小 | 吞吐量(MB/s) | 加速比 | 延迟(μs) |\n");
            report.push_str("|------|----------|------|-------------|--------|----------|\n");

            for result in op_results {
                report.push_str(&format!(
                    "| {:?} | {:?} | {} | {:.2} | {:.2}x | {:.2} |\n",
                    result.strategy,
                    result.data_type,
                    result.data_size,
                    result.throughput_mb_s,
                    result.speedup_factor,
                    result.duration.as_micros()
                ));
            }
            report.push_str("\n");
        }

        // 总结最佳性能
        let best_result = results.iter().max_by(|a, b| {
            a.speedup_factor
                .partial_cmp(&b.speedup_factor)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(best) = best_result {
            report.push_str(&format!(
                "## 性能总结\n\n最佳加速比: {:.2}x ({})\n",
                best.speedup_factor, best.operation
            ));
        }

        report
    }
}
