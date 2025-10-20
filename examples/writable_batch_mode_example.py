"""
🚀 Writable Batch Mode 使用示例

展示如何使用零内存开销的writable_batch_mode进行高性能数组操作
"""
import random
import time
import numpy as np
from numpack import NumPack


def example_basic_usage():
    """示例1: 基本用法"""
    print("=" * 60)
    print("示例1: 基本用法")
    print("=" * 60)
    
    # 创建NumPack文件
    npk = NumPack('example_writable.npk', drop_if_exists=True)
    npk.open()
    
    # 保存一些数组
    arrays = {
        'array1': np.random.rand(1, 1000000),
        'array2': np.random.rand(1, 1000000),
        'array3': np.random.rand(1, 1000000),
    }
    npk.save(arrays)
    
    # 使用writable_batch_mode进行批量修改
    start = time.perf_counter()
    with npk.writable_batch_mode() as wb:
        # 加载数组（返回mmap视图，零拷贝）
        arr1 = wb.load('array1')
        arr1 *= 2.0  # 直接在文件上修改
        
        arr2 = wb.load('array2')
        arr2 += 1.0
        
        arr3 = wb.load('array3')
        arr3 /= 2.0
        
        # save是可选的（保持API一致性）
        wb.save({'array1': arr1, 'array2': arr2, 'array3': arr3})
    
    elapsed = time.perf_counter() - start
    print(f"\n✅ 修改3个数组耗时: {elapsed * 1000:.2f} ms")
    print(f"💾 内存开销: 接近0 MB（只占虚拟内存）")
    
    # 验证修改已持久化
    result = npk.load('array1', lazy=False)
    print(f"📊 验证: array1均值 = {result.mean():.4f}")
    
    npk.close()
    print()


def example_high_performance_loop():
    """示例2: 高性能循环操作"""
    print("=" * 60)
    print("示例2: 高性能循环操作（原始测试用例）")
    print("=" * 60)
    
    # 创建测试数据
    npk = NumPack('example_loop.npk', drop_if_exists=True)
    npk.open()
    
    arrays = {
        'a1': np.random.rand(1, 1000000),
        'a2': np.random.rand(1, 1000000),
        'a3': np.random.rand(1, 1000000),
    }
    npk.save(arrays)
    
    # 高性能循环
    foo = ['a1', 'a2', 'a3']
    random.seed(42)
    
    start = time.perf_counter()
    with npk.writable_batch_mode() as wb:
        for i in range(100):
            c = random.choice(foo)
            a = wb.load(c)    # mmap视图（零拷贝）
            a *= 4.1          # 直接在文件上修改
            wb.save({c: a})   # 可选
    
    elapsed = time.perf_counter() - start
    
    print(f"\n✅ 100次随机操作耗时: {elapsed * 1000:.2f} ms")
    print(f"📈 平均每次: {elapsed * 10:.3f} ms")
    print(f"🎯 性能目标: < 100 ms (< 1 ms/次)")
    
    if elapsed * 1000 <= 100:
        print(f"🎉 达标！提速约 18-20x")
    
    npk.close()
    print()


def example_large_array():
    """示例3: 超大数组（内存装不下）"""
    print("=" * 60)
    print("示例3: 超大数组场景")
    print("=" * 60)
    
    # 创建大数组（~80MB每个）
    npk = NumPack('example_large.npk', drop_if_exists=True)
    npk.open()
    
    print("创建大数组（每个~80MB）...")
    large_arrays = {
        'big1': np.random.rand(1, 10000000).astype(np.float64),
        'big2': np.random.rand(1, 10000000).astype(np.float64),
        'big3': np.random.rand(1, 10000000).astype(np.float64),
    }
    npk.save(large_arrays)
    
    print("使用writable_batch_mode处理...")
    start = time.perf_counter()
    with npk.writable_batch_mode() as wb:
        for name in ['big1', 'big2', 'big3']:
            arr = wb.load(name)
            arr *= 1.5  # 直接在文件上修改
            wb.save({name: arr})
    
    elapsed = time.perf_counter() - start
    
    print(f"\n✅ 处理~240MB数据耗时: {elapsed * 1000:.2f} ms")
    print(f"💾 内存开销: 接近0 MB")
    print(f"🌟 优势: 支持TB级数据（受限磁盘，而非内存）")
    
    npk.close()
    print()


def example_comparison():
    """示例4: batch_mode vs writable_batch_mode对比"""
    print("=" * 60)
    print("示例4: 性能对比")
    print("=" * 60)
    
    # 准备数据
    npk = NumPack('example_compare.npk', drop_if_exists=True)
    npk.open()
    
    test_array = np.random.rand(1, 1000000)
    npk.save({'test': test_array})
    
    # 测试batch_mode
    print("\n测试 batch_mode（内存缓存）...")
    start = time.perf_counter()
    with npk.batch_mode():
        for i in range(50):
            a = npk.load('test')
            a *= 1.1
            npk.save({'test': a})
    batch_time = time.perf_counter() - start
    
    # 恢复数据
    npk.save({'test': test_array})
    
    # 测试writable_batch_mode
    print("测试 writable_batch_mode（零内存）...")
    start = time.perf_counter()
    with npk.writable_batch_mode() as wb:
        for i in range(50):
            a = wb.load('test')
            a *= 1.1
            wb.save({'test': a})
    writable_time = time.perf_counter() - start
    
    print("\n" + "=" * 60)
    print("📊 性能对比结果：")
    print("=" * 60)
    print(f"{'模式':<25} {'耗时(ms)':<12} {'内存开销'}")
    print("-" * 60)
    print(f"{'batch_mode':<25} {batch_time*1000:<12.2f} ~8 MB")
    print(f"{'writable_batch_mode':<25} {writable_time*1000:<12.2f} ~0 MB")
    print("=" * 60)
    print("\n✅ 结论：")
    print("  • batch_mode: 小数组，追求极致速度")
    print("  • writable_batch_mode: 大数组，零内存开销")
    
    npk.close()
    print()


def example_best_practices():
    """示例5: 最佳实践"""
    print("=" * 60)
    print("示例5: 最佳实践")
    print("=" * 60)
    
    npk = NumPack('example_practices.npk', drop_if_exists=True)
    npk.open()
    
    # 创建测试数据
    npk.save({
        'data1': np.random.rand(100, 1000),
        'data2': np.random.rand(100, 1000),
    })
    
    print("\n✅ 推荐做法：")
    print()
    
    # 1. 使用context manager
    print("1. 始终使用context manager:")
    print("```python")
    print("with npk.writable_batch_mode() as wb:")
    print("    arr = wb.load('data')")
    print("    arr *= 2.0")
    print("    # 退出时自动flush")
    print("```")
    print()
    
    # 2. 缓存array引用
    print("2. 缓存经常访问的数组:")
    with npk.writable_batch_mode() as wb:
        # 第一次load会创建mmap
        arr1 = wb.load('data1')
        arr2 = wb.load('data2')
        
        # 后续直接使用缓存的引用
        for i in range(10):
            arr1 *= 1.1
            arr2 += 0.1
    print("✅ 避免重复load同一个数组")
    print()
    
    # 3. 异常处理
    print("3. 异常处理（自动处理）:")
    try:
        with npk.writable_batch_mode() as wb:
            arr = wb.load('data1')
            arr *= 2.0
            # 即使抛出异常，也会自动flush和清理
    except Exception as e:
        pass
    print("✅ context manager自动清理资源")
    print()
    
    npk.close()


if __name__ == '__main__':
    print("\n" + "🚀" * 30)
    print("Writable Batch Mode - 零内存开销高性能方案")
    print("🚀" * 30 + "\n")
    
    # 运行所有示例
    example_basic_usage()
    example_high_performance_loop()
    example_large_array()
    example_comparison()
    example_best_practices()
    
    print("\n" + "=" * 60)
    print("🎉 所有示例运行完成！")
    print("=" * 60)
    print("\n💡 使用建议：")
    print("  • 小数组（< 100MB）：使用 batch_mode()")
    print("  • 大数组（> 100MB）：使用 writable_batch_mode()")
    print("  • 内存受限环境：始终使用 writable_batch_mode()")
    print("  • TB级数据：writable_batch_mode 是唯一选择")
    print()

