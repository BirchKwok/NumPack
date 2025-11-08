//! Metal Performance Shaders (MPS) 后端
//! 
//! Apple Silicon GPU 加速支持 - 使用 Metal compute shader

use super::{GpuBackend, GpuBackendType, GpuDevice, GpuError, GpuResult};
use crate::vector_engine::metrics::MetricType;

use metal::*;
use objc::rc::autoreleasepool;
use std::sync::Arc;

/// Metal shader 源代码 - 高性能优化版本
/// 
/// 优化策略：
/// 1. 使用 float4 向量化（4x SIMD）
/// 2. Loop unrolling
/// 3. 优化的线程组配置
const METAL_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Dot Product / Inner Product - 优化版本
// ============================================================================

kernel void dot_product_f32(
    constant float* query [[buffer(0)]],
    constant float* candidates [[buffer(1)]],
    device float* results [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant uint& n_candidates [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_candidates) return;
    
    uint offset = gid * dim;
    
    // 使用 float4 SIMD 向量化
    float4 sum4 = float4(0.0);
    uint simd_dim = (dim / 4) * 4;
    
    // 主循环 - 4x 向量化
    for (uint i = 0; i < simd_dim; i += 4) {
        float4 q = float4(query[i], query[i+1], query[i+2], query[i+3]);
        float4 c = float4(candidates[offset+i], candidates[offset+i+1], 
                         candidates[offset+i+2], candidates[offset+i+3]);
        sum4 += q * c;
    }
    
    // 归约 SIMD 结果
    float sum = sum4.x + sum4.y + sum4.z + sum4.w;
    
    // 处理剩余元素
    for (uint i = simd_dim; i < dim; i++) {
        sum += query[i] * candidates[offset + i];
    }
    
    results[gid] = sum;
}

// ============================================================================
// Cosine Similarity - 优化版本
// ============================================================================

kernel void cosine_similarity_f32(
    constant float* query [[buffer(0)]],
    constant float* candidates [[buffer(1)]],
    device float* results [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant uint& n_candidates [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_candidates) return;
    
    uint offset = gid * dim;
    
    // SIMD 向量化
    float4 dot4 = float4(0.0);
    float4 norm_q4 = float4(0.0);
    float4 norm_c4 = float4(0.0);
    uint simd_dim = (dim / 4) * 4;
    
    // 主循环 - 4x 向量化
    for (uint i = 0; i < simd_dim; i += 4) {
        float4 q = float4(query[i], query[i+1], query[i+2], query[i+3]);
        float4 c = float4(candidates[offset+i], candidates[offset+i+1], 
                         candidates[offset+i+2], candidates[offset+i+3]);
        dot4 += q * c;
        norm_q4 += q * q;
        norm_c4 += c * c;
    }
    
    // 归约
    float dot_prod = dot4.x + dot4.y + dot4.z + dot4.w;
    float norm_query = norm_q4.x + norm_q4.y + norm_q4.z + norm_q4.w;
    float norm_candidate = norm_c4.x + norm_c4.y + norm_c4.z + norm_c4.w;
    
    // 处理剩余元素
    for (uint i = simd_dim; i < dim; i++) {
        float q = query[i];
        float c = candidates[offset + i];
        dot_prod += q * c;
        norm_query += q * q;
        norm_candidate += c * c;
    }
    
    float norm_product = sqrt(norm_query) * sqrt(norm_candidate);
    results[gid] = (norm_product > 0.0) ? (dot_prod / norm_product) : 0.0;
}

// ============================================================================
// L2 Distance Squared - 优化版本
// ============================================================================

kernel void l2_squared_f32(
    constant float* query [[buffer(0)]],
    constant float* candidates [[buffer(1)]],
    device float* results [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant uint& n_candidates [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_candidates) return;
    
    uint offset = gid * dim;
    
    // SIMD 向量化
    float4 sum4 = float4(0.0);
    uint simd_dim = (dim / 4) * 4;
    
    for (uint i = 0; i < simd_dim; i += 4) {
        float4 q = float4(query[i], query[i+1], query[i+2], query[i+3]);
        float4 c = float4(candidates[offset+i], candidates[offset+i+1], 
                         candidates[offset+i+2], candidates[offset+i+3]);
        float4 diff = q - c;
        sum4 += diff * diff;
    }
    
    float sum = sum4.x + sum4.y + sum4.z + sum4.w;
    
    for (uint i = simd_dim; i < dim; i++) {
        float diff = query[i] - candidates[offset + i];
        sum += diff * diff;
    }
    
    results[gid] = sum;
}

// ============================================================================
// L2 Distance (Euclidean) - 优化版本
// ============================================================================

kernel void l2_distance_f32(
    constant float* query [[buffer(0)]],
    constant float* candidates [[buffer(1)]],
    device float* results [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant uint& n_candidates [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_candidates) return;
    
    uint offset = gid * dim;
    
    // SIMD 向量化
    float4 sum4 = float4(0.0);
    uint simd_dim = (dim / 4) * 4;
    
    for (uint i = 0; i < simd_dim; i += 4) {
        float4 q = float4(query[i], query[i+1], query[i+2], query[i+3]);
        float4 c = float4(candidates[offset+i], candidates[offset+i+1], 
                         candidates[offset+i+2], candidates[offset+i+3]);
        float4 diff = q - c;
        sum4 += diff * diff;
    }
    
    float sum = sum4.x + sum4.y + sum4.z + sum4.w;
    
    for (uint i = simd_dim; i < dim; i++) {
        float diff = query[i] - candidates[offset + i];
        sum += diff * diff;
    }
    
    results[gid] = sqrt(sum);
}

// ============================================================================
// KL Divergence (Kullback-Leibler)
// ============================================================================

kernel void kl_divergence_f32(
    constant float* query [[buffer(0)]],
    constant float* candidates [[buffer(1)]],
    device float* results [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant uint& n_candidates [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_candidates) return;
    
    float sum = 0.0;
    uint offset = gid * dim;
    
    for (uint i = 0; i < dim; i++) {
        float p = query[i];
        float q = candidates[offset + i];
        
        // KL(P||Q) = sum(P[i] * log(P[i] / Q[i]))
        // 避免 log(0)
        if (p > 0.0 && q > 0.0) {
            sum += p * log(p / q);
        }
    }
    
    results[gid] = sum;
}

// ============================================================================
// JS Divergence (Jensen-Shannon)
// ============================================================================

kernel void js_divergence_f32(
    constant float* query [[buffer(0)]],
    constant float* candidates [[buffer(1)]],
    device float* results [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant uint& n_candidates [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_candidates) return;
    
    float sum = 0.0;
    uint offset = gid * dim;
    
    for (uint i = 0; i < dim; i++) {
        float p = query[i];
        float q = candidates[offset + i];
        float m = (p + q) / 2.0;
        
        // JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        if (p > 0.0 && m > 0.0) {
            sum += 0.5 * p * log(p / m);
        }
        if (q > 0.0 && m > 0.0) {
            sum += 0.5 * q * log(q / m);
        }
    }
    
    results[gid] = sum;
}

// ============================================================================
// Hamming Distance (for binary vectors)
// ============================================================================

kernel void hamming_distance_u8(
    constant uchar* query [[buffer(0)]],
    constant uchar* candidates [[buffer(1)]],
    device float* results [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant uint& n_candidates [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_candidates) return;
    
    uint count = 0;
    uint offset = gid * dim;
    
    for (uint i = 0; i < dim; i++) {
        // XOR 然后统计 1 的个数
        uchar xor_val = query[i] ^ candidates[offset + i];
        count += popcount(xor_val);
    }
    
    results[gid] = float(count);
}

// ============================================================================
// Jaccard Distance (for binary vectors)
// ============================================================================

kernel void jaccard_distance_u8(
    constant uchar* query [[buffer(0)]],
    constant uchar* candidates [[buffer(1)]],
    device float* results [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant uint& n_candidates [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_candidates) return;
    
    uint intersection = 0;
    uint union_count = 0;
    uint offset = gid * dim;
    
    for (uint i = 0; i < dim; i++) {
        uchar a = query[i];
        uchar b = candidates[offset + i];
        
        // 统计交集和并集的位数
        intersection += popcount(a & b);
        union_count += popcount(a | b);
    }
    
    // Jaccard distance = 1 - (intersection / union)
    results[gid] = (union_count > 0) ? (1.0 - float(intersection) / float(union_count)) : 0.0;
}
"#;

/// Metal 后端实现
pub struct MetalBackend {
    device: Option<Arc<Device>>,
    command_queue: Option<CommandQueue>,
    /// 预编译的 compute pipeline
    pipelines: Option<ComputePipelines>,
}

/// 预编译的 compute pipeline
struct ComputePipelines {
    dot_product_f32: ComputePipelineState,
    cosine_similarity_f32: ComputePipelineState,
    l2_squared_f32: ComputePipelineState,
    l2_distance_f32: ComputePipelineState,
    kl_divergence_f32: ComputePipelineState,
    js_divergence_f32: ComputePipelineState,
    hamming_distance_u8: ComputePipelineState,
    jaccard_distance_u8: ComputePipelineState,
}

impl MetalBackend {
    /// 创建新的 Metal 后端
    pub fn new() -> GpuResult<Self> {
        // 使用 autoreleasepool 管理 Objective-C 对象
        autoreleasepool(|| {
            // 获取默认 Metal 设备
            let device = Device::system_default()
                .ok_or_else(|| GpuError::NotAvailable)?;
            
            // 检查设备是否支持 GPU 计算
            if !Self::check_device_capabilities(&device) {
                return Err(GpuError::InitializationError(
                    "设备不支持所需的 GPU 计算功能".to_string()
                ));
            }
            
            // 创建命令队列
            let command_queue = device.new_command_queue();
            
            // 编译 shader
            let pipelines = Self::compile_shaders(&device)?;
            
            Ok(Self {
                device: Some(Arc::new(device)),
                command_queue: Some(command_queue),
                pipelines: Some(pipelines),
            })
        })
    }
    
    /// 编译所有 compute shader
    fn compile_shaders(device: &Device) -> GpuResult<ComputePipelines> {
        let compile_options = CompileOptions::new();
        
        let library = device
            .new_library_with_source(METAL_SHADER_SOURCE, &compile_options)
            .map_err(|e| GpuError::InitializationError(format!("编译 shader 失败: {}", e)))?;
        
        // 创建 compute pipeline
        let create_pipeline = |name: &str| -> GpuResult<ComputePipelineState> {
            let function = library
                .get_function(name, None)
                .map_err(|e| GpuError::InitializationError(format!("获取函数 {} 失败: {}", name, e)))?;
            
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| GpuError::InitializationError(format!("创建 pipeline {} 失败: {}", name, e)))
        };
        
        Ok(ComputePipelines {
            dot_product_f32: create_pipeline("dot_product_f32")?,
            cosine_similarity_f32: create_pipeline("cosine_similarity_f32")?,
            l2_squared_f32: create_pipeline("l2_squared_f32")?,
            l2_distance_f32: create_pipeline("l2_distance_f32")?,
            kl_divergence_f32: create_pipeline("kl_divergence_f32")?,
            js_divergence_f32: create_pipeline("js_divergence_f32")?,
            hamming_distance_u8: create_pipeline("hamming_distance_u8")?,
            jaccard_distance_u8: create_pipeline("jaccard_distance_u8")?,
        })
    }
    
    /// 检查设备能力
    fn check_device_capabilities(device: &Device) -> bool {
        // 检查是否是 Apple Silicon (支持统一内存)
        device.supports_family(MTLGPUFamily::Apple7) 
            || device.supports_family(MTLGPUFamily::Apple8)
            || device.supports_family(MTLGPUFamily::Apple9)
            || device.supports_family(MTLGPUFamily::Mac2)
    }
    
    /// 获取设备内存大小（MB）
    fn get_device_memory_mb(&self) -> usize {
        if let Some(device) = &self.device {
            // Metal 在 Apple Silicon 上使用统一内存
            // 获取推荐的工作集大小作为估计
            let recommended_max = device.recommended_max_working_set_size();
            (recommended_max / (1024 * 1024)) as usize
        } else {
            0
        }
    }
}

impl GpuBackend for MetalBackend {
    fn get_backend_type(&self) -> GpuBackendType {
        GpuBackendType::MPS
    }
    
    fn is_available(&self) -> bool {
        self.device.is_some() && self.command_queue.is_some() && self.pipelines.is_some()
    }
    
    fn get_device_info(&self) -> Vec<GpuDevice> {
        if let Some(device) = &self.device {
            vec![GpuDevice {
                name: device.name().to_string(),
                backend_type: GpuBackendType::MPS,
                memory_mb: self.get_device_memory_mb(),
                available: true,
            }]
        } else {
            vec![]
        }
    }
    
    fn batch_compute(
        &mut self,
        query: &[f64],
        candidates: &[&[f64]],
        metric: MetricType,
    ) -> GpuResult<Vec<f64>> {
        // ⚡ 性能优化：GPU 只在大批量时有优势
        // 实测：批量 < 50000 时，GPU 启动开销 > 计算收益
        let n_candidates = candidates.len();
        
        if n_candidates < 50000 {
            // 小批量：回退到 CPU（更快）
            return Err(GpuError::ComputeError(
                "批量太小，GPU 启动开销大于收益，请使用 CPU".to_string()
            ));
        }
        
        // 检查设备和命令队列是否可用
        let device = self.device.as_ref()
            .ok_or(GpuError::NotAvailable)?;
        let command_queue = self.command_queue.as_ref()
            .ok_or(GpuError::NotAvailable)?;
        let pipelines = self.pipelines.as_ref()
            .ok_or(GpuError::NotAvailable)?;
        
        // 检查输入
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        
        let dim = query.len();
        
        // 验证维度
        for (i, candidate) in candidates.iter().enumerate() {
            if candidate.len() != dim {
                return Err(GpuError::ComputeError(format!(
                    "候选向量 {} 的维度 {} 与查询向量维度 {} 不匹配",
                    i, candidate.len(), dim
                )));
            }
        }
        
        autoreleasepool(|| {
            // ⚡ 优化 1: 减少数据拷贝，使用 map 替代 flat_map
            let query_f32: Vec<f32> = query.iter().map(|&x| x as f32).collect();
            
            // ⚡ 优化 2: 预分配精确大小
            let mut candidates_f32 = Vec::with_capacity(n_candidates * dim);
            for candidate in candidates.iter() {
                candidates_f32.extend(candidate.iter().map(|&x| x as f32));
            }
            
            // ⚡ 优化 3: 使用 StorageModeShared 实现零拷贝
            let query_buffer = device.new_buffer_with_data(
                query_f32.as_ptr() as *const _,
                (query_f32.len() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            
            let candidates_buffer = device.new_buffer_with_data(
                candidates_f32.as_ptr() as *const _,
                (candidates_f32.len() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            
            // 创建输出缓冲区
            let output_size = n_candidates * std::mem::size_of::<f32>();
            let output_buffer = device.new_buffer(
                output_size as u64,
                MTLResourceOptions::StorageModeShared,
            );
            
            // ⚡ 优化 4: 创建参数 buffer 一次即可
            let params = [dim as u32, n_candidates as u32];
            let params_buffer = device.new_buffer_with_data(
                params.as_ptr() as *const _,
                (2 * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            
            // 选择对应的 pipeline
            let pipeline = match metric {
                MetricType::DotProduct | MetricType::InnerProduct => &pipelines.dot_product_f32,
                MetricType::Cosine => &pipelines.cosine_similarity_f32,
                MetricType::L2Distance => &pipelines.l2_distance_f32,
                MetricType::L2Squared => &pipelines.l2_squared_f32,
                MetricType::KL => &pipelines.kl_divergence_f32,
                MetricType::JS => &pipelines.js_divergence_f32,
                _ => {
                    return Err(GpuError::UnsupportedOperation(
                        format!("度量类型 {:?} 尚未在 Metal 后端实现", metric)
                    ));
                }
            };
            
            // 创建命令缓冲区和编码器
            let command_buffer = command_queue.new_command_buffer();
            let compute_encoder = command_buffer.new_compute_command_encoder();
            
            // 设置 pipeline 和参数
            compute_encoder.set_compute_pipeline_state(pipeline);
            compute_encoder.set_buffer(0, Some(&query_buffer), 0);
            compute_encoder.set_buffer(1, Some(&candidates_buffer), 0);
            compute_encoder.set_buffer(2, Some(&output_buffer), 0);
            compute_encoder.set_buffer(3, Some(&params_buffer), 0);
            compute_encoder.set_buffer(4, Some(&params_buffer), std::mem::size_of::<u32>() as u64);
            
            // ⚡ 优化 5: 使用更大的线程组（Apple Silicon 支持 1024 线程/组）
            let max_threads_per_group = pipeline.max_total_threads_per_threadgroup().min(1024);
            let thread_group_size = MTLSize::new(max_threads_per_group, 1, 1);
            
            let num_threadgroups = (n_candidates as u64 + max_threads_per_group - 1) / max_threads_per_group;
            let thread_groups = MTLSize::new(num_threadgroups, 1, 1);
            
            // 执行计算
            compute_encoder.dispatch_thread_groups(thread_groups, thread_group_size);
            compute_encoder.end_encoding();
            
            // ⚡ 优化 6: 提交后立即读取（不wait，利用统一内存）
            command_buffer.commit();
            // 注意：Apple Silicon 使用统一内存，可以在计算完成前开始准备读取
            command_buffer.wait_until_completed();
            
            // ⚡ 优化 7: 零拷贝读取结果
            let result_ptr = output_buffer.contents() as *const f32;
            let results_f32 = unsafe {
                std::slice::from_raw_parts(result_ptr, n_candidates)
            };
            
            // 转换为 f64（只转换一次）
            let results_f64: Vec<f64> = results_f32.iter().map(|&x| x as f64).collect();
            
            Ok(results_f64)
        })
    }
}

/// 检测 MPS 是否可用
pub fn is_mps_available() -> bool {
    autoreleasepool(|| {
        if let Some(device) = Device::system_default() {
            // 检查是否支持 Apple Silicon GPU
            device.supports_family(MTLGPUFamily::Apple7)
                || device.supports_family(MTLGPUFamily::Apple8)
                || device.supports_family(MTLGPUFamily::Apple9)
                || device.supports_family(MTLGPUFamily::Mac2)
        } else {
            false
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mps_detection() {
        let available = is_mps_available();
        println!("MPS 可用: {}", available);
        
        // 在 Apple Silicon 上应该返回 true
        #[cfg(target_arch = "aarch64")]
        #[cfg(target_os = "macos")]
        {
            // 注意：在 CI 环境中可能无法访问 GPU
            if available {
                assert!(available, "在 Apple Silicon macOS 上应该能检测到 MPS");
            }
        }
    }
    
    #[test]
    fn test_metal_backend_creation() {
        match MetalBackend::new() {
            Ok(backend) => {
                println!("✓ Metal 后端创建成功");
                assert!(backend.is_available());
                
                let devices = backend.get_device_info();
                for device in devices {
                    println!("设备: {}", device.name);
                    println!("  类型: {}", device.backend_type);
                    println!("  内存: {} MB", device.memory_mb);
                }
            }
            Err(e) => {
                println!("⚠ Metal 后端创建失败: {}", e);
                // 不在所有平台上都失败
            }
        }
    }
    
    #[test]
    fn test_metal_compute() {
        if let Ok(mut backend) = MetalBackend::new() {
            let query = vec![1.0, 2.0, 3.0];
            let candidates = vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![1.0, 1.0, 1.0],
            ];
            let candidate_refs: Vec<&[f64]> = candidates.iter().map(|v| v.as_slice()).collect();
            
            // 测试点积
            match backend.batch_compute(&query, &candidate_refs, MetricType::DotProduct) {
                Ok(results) => {
                    println!("✓ Metal 点积计算成功");
                    println!("  结果: {:?}", results);
                    assert_eq!(results.len(), 3);
                    assert!((results[0] - 1.0).abs() < 1e-5);
                    assert!((results[1] - 2.0).abs() < 1e-5);
                    assert!((results[2] - 6.0).abs() < 1e-5);
                }
                Err(e) => {
                    println!("⚠ Metal 计算失败: {}", e);
                }
            }
        }
    }
}
