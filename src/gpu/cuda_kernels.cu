/*
 * CUDA Kernels for NumPack Vector Engine
 * 
 * 实现所有 SIMSIMD 支持的度量类型
 * 
 * 编译: nvcc -ptx cuda_kernels.cu -o cuda_kernels.ptx
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ============================================================================
// Dot Product / Inner Product
// ============================================================================

extern "C" __global__ void dot_product_f32(
    const float* __restrict__ query,
    const float* __restrict__ candidates,
    float* __restrict__ results,
    unsigned int dim,
    unsigned int n_candidates
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_candidates) return;
    
    unsigned int offset = gid * dim;
    
    // 使用 float4 向量化
    float sum = 0.0f;
    unsigned int vec_dim = (dim / 4) * 4;
    
    // 向量化主循环（4x unroll）
    for (unsigned int i = 0; i < vec_dim; i += 4) {
        float4 q = *reinterpret_cast<const float4*>(&query[i]);
        float4 c = *reinterpret_cast<const float4*>(&candidates[offset + i]);
        
        sum += q.x * c.x + q.y * c.y + q.z * c.z + q.w * c.w;
    }
    
    // 处理剩余元素
    for (unsigned int i = vec_dim; i < dim; i++) {
        sum += query[i] * candidates[offset + i];
    }
    
    results[gid] = sum;
}

// ============================================================================
// Cosine Similarity
// ============================================================================

extern "C" __global__ void cosine_similarity_f32(
    const float* __restrict__ query,
    const float* __restrict__ candidates,
    float* __restrict__ results,
    unsigned int dim,
    unsigned int n_candidates
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_candidates) return;
    
    unsigned int offset = gid * dim;
    
    float dot_prod = 0.0f;
    float norm_query = 0.0f;
    float norm_candidate = 0.0f;
    
    unsigned int vec_dim = (dim / 4) * 4;
    
    // 向量化主循环
    for (unsigned int i = 0; i < vec_dim; i += 4) {
        float4 q = *reinterpret_cast<const float4*>(&query[i]);
        float4 c = *reinterpret_cast<const float4*>(&candidates[offset + i]);
        
        dot_prod += q.x * c.x + q.y * c.y + q.z * c.z + q.w * c.w;
        norm_query += q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
        norm_candidate += c.x * c.x + c.y * c.y + c.z * c.z + c.w * c.w;
    }
    
    // 处理剩余元素
    for (unsigned int i = vec_dim; i < dim; i++) {
        float q = query[i];
        float c = candidates[offset + i];
        dot_prod += q * c;
        norm_query += q * q;
        norm_candidate += c * c;
    }
    
    float norm_product = sqrtf(norm_query) * sqrtf(norm_candidate);
    results[gid] = (norm_product > 0.0f) ? (dot_prod / norm_product) : 0.0f;
}

// ============================================================================
// L2 Distance Squared
// ============================================================================

extern "C" __global__ void l2_squared_f32(
    const float* __restrict__ query,
    const float* __restrict__ candidates,
    float* __restrict__ results,
    unsigned int dim,
    unsigned int n_candidates
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_candidates) return;
    
    unsigned int offset = gid * dim;
    
    float sum = 0.0f;
    unsigned int vec_dim = (dim / 4) * 4;
    
    for (unsigned int i = 0; i < vec_dim; i += 4) {
        float4 q = *reinterpret_cast<const float4*>(&query[i]);
        float4 c = *reinterpret_cast<const float4*>(&candidates[offset + i]);
        
        float4 diff;
        diff.x = q.x - c.x;
        diff.y = q.y - c.y;
        diff.z = q.z - c.z;
        diff.w = q.w - c.w;
        
        sum += diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + diff.w * diff.w;
    }
    
    for (unsigned int i = vec_dim; i < dim; i++) {
        float diff = query[i] - candidates[offset + i];
        sum += diff * diff;
    }
    
    results[gid] = sum;
}

// ============================================================================
// L2 Distance
// ============================================================================

extern "C" __global__ void l2_distance_f32(
    const float* __restrict__ query,
    const float* __restrict__ candidates,
    float* __restrict__ results,
    unsigned int dim,
    unsigned int n_candidates
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_candidates) return;
    
    unsigned int offset = gid * dim;
    
    float sum = 0.0f;
    unsigned int vec_dim = (dim / 4) * 4;
    
    for (unsigned int i = 0; i < vec_dim; i += 4) {
        float4 q = *reinterpret_cast<const float4*>(&query[i]);
        float4 c = *reinterpret_cast<const float4*>(&candidates[offset + i]);
        
        float4 diff;
        diff.x = q.x - c.x;
        diff.y = q.y - c.y;
        diff.z = q.z - c.z;
        diff.w = q.w - c.w;
        
        sum += diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + diff.w * diff.w;
    }
    
    for (unsigned int i = vec_dim; i < dim; i++) {
        float diff = query[i] - candidates[offset + i];
        sum += diff * diff;
    }
    
    results[gid] = sqrtf(sum);
}

// ============================================================================
// KL Divergence
// ============================================================================

extern "C" __global__ void kl_divergence_f32(
    const float* __restrict__ query,
    const float* __restrict__ candidates,
    float* __restrict__ results,
    unsigned int dim,
    unsigned int n_candidates
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_candidates) return;
    
    unsigned int offset = gid * dim;
    
    float sum = 0.0f;
    
    for (unsigned int i = 0; i < dim; i++) {
        float p = query[i];
        float q = candidates[offset + i];
        
        // KL(P||Q) = sum(P[i] * log(P[i] / Q[i]))
        if (p > 0.0f && q > 0.0f) {
            sum += p * logf(p / q);
        }
    }
    
    results[gid] = sum;
}

// ============================================================================
// JS Divergence
// ============================================================================

extern "C" __global__ void js_divergence_f32(
    const float* __restrict__ query,
    const float* __restrict__ candidates,
    float* __restrict__ results,
    unsigned int dim,
    unsigned int n_candidates
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_candidates) return;
    
    unsigned int offset = gid * dim;
    
    float sum = 0.0f;
    
    for (unsigned int i = 0; i < dim; i++) {
        float p = query[i];
        float q = candidates[offset + i];
        float m = (p + q) * 0.5f;
        
        // JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        if (p > 0.0f && m > 0.0f) {
            sum += 0.5f * p * logf(p / m);
        }
        if (q > 0.0f && m > 0.0f) {
            sum += 0.5f * q * logf(q / m);
        }
    }
    
    results[gid] = sum;
}

