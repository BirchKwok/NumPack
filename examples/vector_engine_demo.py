#!/usr/bin/env python3
"""
VectorEngine 完整功能演示

展示 VectorEngine 的所有功能：
1. 批量计算（batch_compute）
2. Top-K 搜索（top_k_search）
3. 多数据类型支持
4. 性能对比
"""

from numpack import VectorEngine
import numpy as np
import time

def demo_basic_usage():
    """基础使用演示"""
    print('=' * 80)
    print('📚 VectorEngine 基础使用')
    print('=' * 80)
    print()
    
    # 创建引擎
    engine = VectorEngine()
    print(f'引擎能力: {engine.capabilities()}')
    print()
    
    # 准备数据（推荐使用 float32）
    query = np.random.rand(768).astype(np.float32)
    candidates = np.random.rand(10000, 768).astype(np.float32)
    
    print('1️⃣  批量计算 - 计算所有候选向量的分数')
    print('-' * 80)
    scores = engine.batch_compute(query, candidates, 'cosine', device='cpu')
    print(f'   计算了 {len(scores)} 个分数')
    print(f'   分数范围: [{scores.min():.4f}, {scores.max():.4f}]')
    print()
    
    print('2️⃣  Top-K 搜索 - 直接找到最相似的 k 个')
    print('-' * 80)
    k = 10
    indices, top_scores = engine.top_k_search(query, candidates, 'cosine', k=k)
    print(f'   找到 Top-{k}:')
    for i in range(k):
        print(f'     #{i+1}: index={indices[i]:5d}, score={top_scores[i]:.6f}')
    print()

def demo_multi_dtype():
    """多数据类型演示"""
    print('=' * 80)
    print('🎨 VectorEngine 多数据类型支持')
    print('=' * 80)
    print()
    
    engine = VectorEngine()
    
    # float32 - 推荐用于通用场景
    print('1️⃣  float32（单精度） - 推荐配置 ⭐')
    print('-' * 80)
    q_f32 = np.random.rand(768).astype(np.float32)
    c_f32 = np.random.rand(10000, 768).astype(np.float32)
    
    start = time.perf_counter()
    indices, scores = engine.top_k_search(q_f32, c_f32, 'cosine', k=10)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f'   Top-10 搜索: {elapsed:.2f} ms')
    print(f'   内存占用: {c_f32.nbytes / 1024 / 1024:.1f} MB')
    print(f'   推荐场景: 通用文本/图像检索')
    print()
    
    # int8 - 量化向量
    print('2️⃣  int8（整数） - 量化向量 💾')
    print('-' * 80)
    q_i8 = np.random.randint(-100, 100, 768, dtype=np.int8)
    c_i8 = np.random.randint(-100, 100, (10000, 768), dtype=np.int8)
    
    start = time.perf_counter()
    indices, scores = engine.top_k_search(q_i8, c_i8, 'dot', k=10)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f'   Top-10 搜索: {elapsed:.2f} ms')
    print(f'   内存占用: {c_i8.nbytes / 1024 / 1024:.1f} MB')
    print(f'   内存节省: {(1 - c_i8.nbytes / c_f32.nbytes) * 100:.0f}% vs float32')
    print(f'   推荐场景: 量化模型嵌入')
    print()
    
    # uint8 - 二进制向量
    print('3️⃣  uint8（二进制） - 最快 ⚡')
    print('-' * 80)
    q_u8 = np.random.randint(0, 2, 1024, dtype=np.uint8)
    c_u8 = np.random.randint(0, 2, (10000, 1024), dtype=np.uint8)
    
    start = time.perf_counter()
    indices, scores = engine.top_k_search(q_u8, c_u8, 'hamming', k=10)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f'   Top-10 搜索: {elapsed:.2f} ms（最快！）')
    print(f'   内存占用: {c_u8.nbytes / 1024 / 1024:.1f} MB')
    print(f'   推荐场景: SimHash 指纹匹配')
    print()

def demo_real_world_scenario():
    """实际应用场景演示"""
    print('=' * 80)
    print('🌟 VectorEngine 实际应用场景')
    print('=' * 80)
    print()
    
    engine = VectorEngine()
    
    # 场景：大规模语义搜索
    print('场景: 大规模语义搜索（100万文档）')
    print('-' * 80)
    print()
    
    # 假设的文档数据
    n_docs = 1000000  # 100万文档
    dim = 768         # BERT base 维度
    
    print(f'文档库: {n_docs:,} 个文档')
    print(f'嵌入维度: {dim}')
    print()
    
    # 模拟查询
    query = np.random.rand(dim).astype(np.float32)
    
    # 实际应用中，这些会从磁盘或数据库加载
    # 这里我们只测试小规模以演示
    candidates = np.random.rand(10000, dim).astype(np.float32)
    
    # Step 1: Top-K 检索
    k = 100
    print(f'Step 1: Top-{k} 粗召回')
    start = time.perf_counter()
    indices, scores = engine.top_k_search(query, candidates, 'cosine', k=k)
    elapsed = (time.perf_counter() - start) * 1000
    print(f'  耗时: {elapsed:.2f} ms')
    print(f'  找到 {len(indices)} 个候选')
    print()
    
    # Step 2: 后处理（阈值过滤）
    print(f'Step 2: 阈值过滤')
    threshold = 0.7
    mask = scores >= threshold
    filtered_indices = indices[mask]
    filtered_scores = scores[mask]
    print(f'  阈值: {threshold}')
    print(f'  保留: {len(filtered_indices)}/{k} 个')
    print()
    
    # Step 3: 结果展示
    print(f'最终结果:')
    for i in range(min(5, len(filtered_indices))):
        doc_id = filtered_indices[i]
        similarity = filtered_scores[i]
        print(f'  {i+1}. 文档{doc_id:6d} (相似度: {similarity:.4f})')
    print()

def demo_performance_comparison():
    """性能对比演示"""
    print('=' * 80)
    print('⚡ VectorEngine 性能对比')
    print('=' * 80)
    print()
    
    engine = VectorEngine()
    
    # 测试数据
    query = np.random.rand(768)
    candidates = np.random.rand(10000, 768)
    
    print('测试配置: 10,000 个 768 维向量')
    print()
    
    # VectorEngine
    print('🔹 VectorEngine:')
    times = []
    for _ in range(10):
        start = time.perf_counter()
        _ = engine.batch_compute(query, candidates, 'dot', device='cpu')
        times.append((time.perf_counter() - start) * 1000)
    ve_time = np.mean(times)
    print(f'   batch_compute: {ve_time:.2f} ms ± {np.std(times):.2f} ms')
    
    times = []
    for _ in range(10):
        start = time.perf_counter()
        _ = engine.top_k_search(query, candidates, 'dot', k=10)
        times.append((time.perf_counter() - start) * 1000)
    topk_time = np.mean(times)
    print(f'   top_k_search:  {topk_time:.2f} ms ± {np.std(times):.2f} ms')
    print()
    
    # NumPy
    print('🔹 NumPy:')
    times = []
    for _ in range(10):
        start = time.perf_counter()
        scores = np.dot(candidates, query)
        indices = np.argsort(scores)[-10:][::-1]
        times.append((time.perf_counter() - start) * 1000)
    numpy_time = np.mean(times)
    print(f'   dot + argsort: {numpy_time:.2f} ms ± {np.std(times):.2f} ms')
    print()
    
    # 对比
    print('📊 加速比:')
    print(f'   VectorEngine vs NumPy: {numpy_time/ve_time:.2f}x 🚀')
    print()

if __name__ == '__main__':
    demo_basic_usage()
    demo_multi_dtype()
    demo_real_world_scenario()
    demo_performance_comparison()
    
    print('=' * 80)
    print('✅ 演示完成！')
    print()
    print('更多信息:')
    print('  • VECTOR_ENGINE_MULTI_DTYPE_GUIDE.md - 多数据类型指南')
    print('  • VECTOR_ENGINE_TOP_K_GUIDE.md - Top-K 搜索指南')
    print('=' * 80)

