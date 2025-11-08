//! WebGPU 后端
//! 
//! 通用跨平台 GPU 加速支持 - 使用 WGSL compute shader

use super::{GpuBackend, GpuBackendType, GpuDevice, GpuError, GpuResult};
use crate::vector_engine::metrics::MetricType;

use wgpu::*;
use wgpu::util::DeviceExt;
use std::sync::Arc;

/// WGSL shader 源代码
/// 
/// 实现所有 SIMSIMD 支持的度量类型
const WGSL_SHADER_SOURCE: &str = r#"
// ============================================================================
// Dot Product / Inner Product
// ============================================================================

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> candidates: array<f32>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    dim: u32,
    n_candidates: u32,
}

@compute @workgroup_size(64)
fn dot_product(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid >= params.n_candidates) { return; }
    
    var sum = 0.0;
    let offset = gid * params.dim;
    
    for (var i = 0u; i < params.dim; i = i + 1u) {
        sum += query[i] * candidates[offset + i];
    }
    
    results[gid] = sum;
}

// ============================================================================
// Cosine Similarity
// ============================================================================

@compute @workgroup_size(64)
fn cosine_similarity(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid >= params.n_candidates) { return; }
    
    var dot_prod = 0.0;
    var norm_query = 0.0;
    var norm_candidate = 0.0;
    let offset = gid * params.dim;
    
    for (var i = 0u; i < params.dim; i = i + 1u) {
        let q = query[i];
        let c = candidates[offset + i];
        dot_prod += q * c;
        norm_query += q * q;
        norm_candidate += c * c;
    }
    
    let norm_product = sqrt(norm_query) * sqrt(norm_candidate);
    results[gid] = select(0.0, dot_prod / norm_product, norm_product > 0.0);
}

// ============================================================================
// L2 Distance Squared
// ============================================================================

@compute @workgroup_size(64)
fn l2_squared(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid >= params.n_candidates) { return; }
    
    var sum = 0.0;
    let offset = gid * params.dim;
    
    for (var i = 0u; i < params.dim; i = i + 1u) {
        let diff = query[i] - candidates[offset + i];
        sum += diff * diff;
    }
    
    results[gid] = sum;
}

// ============================================================================
// L2 Distance
// ============================================================================

@compute @workgroup_size(64)
fn l2_distance(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid >= params.n_candidates) { return; }
    
    var sum = 0.0;
    let offset = gid * params.dim;
    
    for (var i = 0u; i < params.dim; i = i + 1u) {
        let diff = query[i] - candidates[offset + i];
        sum += diff * diff;
    }
    
    results[gid] = sqrt(sum);
}

// ============================================================================
// KL Divergence
// ============================================================================

@compute @workgroup_size(64)
fn kl_divergence(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid >= params.n_candidates) { return; }
    
    var sum = 0.0;
    let offset = gid * params.dim;
    
    for (var i = 0u; i < params.dim; i = i + 1u) {
        let p = query[i];
        let q = candidates[offset + i];
        
        // KL(P||Q) = sum(P[i] * log(P[i] / Q[i]))
        if (p > 0.0 && q > 0.0) {
            sum += p * log(p / q);
        }
    }
    
    results[gid] = sum;
}

// ============================================================================
// JS Divergence
// ============================================================================

@compute @workgroup_size(64)
fn js_divergence(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid >= params.n_candidates) { return; }
    
    var sum = 0.0;
    let offset = gid * params.dim;
    
    for (var i = 0u; i < params.dim; i = i + 1u) {
        let p = query[i];
        let q = candidates[offset + i];
        let m = (p + q) / 2.0;
        
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
"#;

/// WebGPU 后端实现
pub struct WebGpuBackend {
    device: Option<Arc<Device>>,
    queue: Option<Arc<Queue>>,
    adapter_info: Option<AdapterInfo>,
    /// 预编译的 compute pipeline
    pipelines: Option<ComputePipelines>,
}

/// 预编译的 compute pipeline
struct ComputePipelines {
    dot_product: ComputePipeline,
    cosine_similarity: ComputePipeline,
    l2_squared: ComputePipeline,
    l2_distance: ComputePipeline,
    kl_divergence: ComputePipeline,
    js_divergence: ComputePipeline,
}

impl WebGpuBackend {
    /// 创建新的 WebGPU 后端
    pub fn new() -> GpuResult<Self> {
        // 使用 pollster 来运行异步代码
        pollster::block_on(Self::new_async())
    }
    
    async fn new_async() -> GpuResult<Self> {
        // 创建 WebGPU 实例
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        
        // 请求适配器（GPU 设备）
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| GpuError::NotAvailable)?;
        
        // 获取适配器信息
        let adapter_info = adapter.get_info();
        
        eprintln!("✓ WebGPU 适配器: {}", adapter_info.name);
        eprintln!("  后端: {:?}", adapter_info.backend);
        eprintln!("  设备类型: {:?}", adapter_info.device_type);
        
        // 请求设备和队列
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("NumPack GPU Device"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    memory_hints: MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| GpuError::InitializationError(format!("请求设备失败: {}", e)))?;
        
        // 编译 shader
        let pipelines = Self::compile_shaders(&device)?;
        
        Ok(Self {
            device: Some(Arc::new(device)),
            queue: Some(Arc::new(queue)),
            adapter_info: Some(adapter_info),
            pipelines: Some(pipelines),
        })
    }
    
    /// 编译所有 compute shader
    fn compile_shaders(device: &Device) -> GpuResult<ComputePipelines> {
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("NumPack Compute Shaders"),
            source: ShaderSource::Wgsl(WGSL_SHADER_SOURCE.into()),
        });
        
        // 创建 pipeline layout
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("NumPack Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("NumPack Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // 创建各个 pipeline
        let create_pipeline = |entry_point: &str| -> ComputePipeline {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(&format!("NumPack {} Pipeline", entry_point)),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point,
                compilation_options: Default::default(),
                cache: None,
            })
        };
        
        Ok(ComputePipelines {
            dot_product: create_pipeline("dot_product"),
            cosine_similarity: create_pipeline("cosine_similarity"),
            l2_squared: create_pipeline("l2_squared"),
            l2_distance: create_pipeline("l2_distance"),
            kl_divergence: create_pipeline("kl_divergence"),
            js_divergence: create_pipeline("js_divergence"),
        })
    }
    
    /// 获取设备内存大小（MB）
    fn get_device_memory_mb(&self) -> usize {
        // WebGPU 不提供直接的内存查询 API
        // 返回一个合理的估计值
        8192 // 8GB 估计
    }
}

impl GpuBackend for WebGpuBackend {
    fn get_backend_type(&self) -> GpuBackendType {
        GpuBackendType::WebGPU
    }
    
    fn is_available(&self) -> bool {
        self.device.is_some() && self.queue.is_some() && self.pipelines.is_some()
    }
    
    fn get_device_info(&self) -> Vec<GpuDevice> {
        if let Some(info) = &self.adapter_info {
            vec![GpuDevice {
                name: info.name.clone(),
                backend_type: GpuBackendType::WebGPU,
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
        // 检查设备是否可用
        let device = self.device.as_ref()
            .ok_or(GpuError::NotAvailable)?;
        let queue = self.queue.as_ref()
            .ok_or(GpuError::NotAvailable)?;
        let pipelines = self.pipelines.as_ref()
            .ok_or(GpuError::NotAvailable)?;
        
        // 检查输入
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        
        let dim = query.len();
        let n_candidates = candidates.len();
        
        // 验证维度
        for (i, candidate) in candidates.iter().enumerate() {
            if candidate.len() != dim {
                return Err(GpuError::ComputeError(format!(
                    "候选向量 {} 的维度 {} 与查询向量维度 {} 不匹配",
                    i, candidate.len(), dim
                )));
            }
        }
        
        // 将 f64 转换为 f32
        let query_f32: Vec<f32> = query.iter().map(|&x| x as f32).collect();
        let candidates_f32: Vec<f32> = candidates
            .iter()
            .flat_map(|c| c.iter().map(|&x| x as f32))
            .collect();
        
        // 创建缓冲区
        let query_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Query Buffer"),
            contents: bytemuck::cast_slice(&query_f32),
            usage: BufferUsages::STORAGE,
        });
        
        let candidates_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Candidates Buffer"),
            contents: bytemuck::cast_slice(&candidates_f32),
            usage: BufferUsages::STORAGE,
        });
        
        let results_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Results Buffer"),
            size: (n_candidates * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // 创建参数缓冲区
        let params = [dim as u32, n_candidates as u32];
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&params),
            usage: BufferUsages::UNIFORM,
        });
        
        // 选择对应的 pipeline
        let pipeline = match metric {
            MetricType::DotProduct | MetricType::InnerProduct => &pipelines.dot_product,
            MetricType::Cosine => &pipelines.cosine_similarity,
            MetricType::L2Distance => &pipelines.l2_distance,
            MetricType::L2Squared => &pipelines.l2_squared,
            MetricType::KL => &pipelines.kl_divergence,
            MetricType::JS => &pipelines.js_divergence,
            _ => {
                return Err(GpuError::UnsupportedOperation(
                    format!("度量类型 {:?} 尚未在 WebGPU 后端实现（仅支持 f32 类型）", metric)
                ));
            }
        };
        
        // 创建 bind group
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("NumPack Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: query_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: candidates_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        
        // 创建输出缓冲区（用于读取结果）
        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (n_candidates * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // 创建命令编码器并执行计算
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            let workgroup_size = 64;
            let num_workgroups = (n_candidates as u32 + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        
        // 复制结果到 staging buffer
        encoder.copy_buffer_to_buffer(
            &results_buffer,
            0,
            &staging_buffer,
            0,
            (n_candidates * std::mem::size_of::<f32>()) as u64,
        );
        
        // 提交命令
        queue.submit(Some(encoder.finish()));
        
        // 读取结果
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        
        device.poll(Maintain::Wait);
        rx.recv().unwrap().map_err(|e| GpuError::ComputeError(format!("Buffer map failed: {:?}", e)))?;
        
        let data = buffer_slice.get_mapped_range();
        let results_f32: &[f32] = bytemuck::cast_slice(&data);
        let results_f64: Vec<f64> = results_f32.iter().map(|&x| x as f64).collect();
        
        drop(data);
        staging_buffer.unmap();
        
        Ok(results_f64)
    }
}

/// 检测 WebGPU 是否可用
pub fn is_webgpu_available() -> bool {
    pollster::block_on(async {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        
        instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .is_some()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_webgpu_detection() {
        let available = is_webgpu_available();
        println!("WebGPU 可用: {}", available);
    }
    
    #[test]
    fn test_webgpu_backend_creation() {
        match WebGpuBackend::new() {
            Ok(backend) => {
                println!("✓ WebGPU 后端创建成功");
                assert!(backend.is_available());
                
                let devices = backend.get_device_info();
                for device in devices {
                    println!("设备: {}", device.name);
                    println!("  类型: {}", device.backend_type);
                    println!("  内存: {} MB", device.memory_mb);
                }
            }
            Err(e) => {
                println!("⚠ WebGPU 后端创建失败: {}", e);
            }
        }
    }
}
