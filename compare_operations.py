"""对比原地操作和非原地操作的性能"""

import numpy as np
import time
from numpack import NumPack
import shutil

def test_non_inplace():
    """测试非原地操作：a = a * 4.1"""
    numpack_dir = "test_npk_non_inplace"
    npk = NumPack(numpack_dir, drop_if_exists=True) 
    npk.open()
    
    arrays = {'a1': np.random.rand(1, 1000000)}
    npk.save(arrays)
    
    times = []
    for i in range(20):
        t0 = time.time()
        
        # 非原地操作
        a = npk.load('a1', lazy=True)
        a = a * 4.1  # 创建新数组
        npk.save({'a1': a})
        
        times.append(time.time() - t0)
    
    npk.close()
    shutil.rmtree(numpack_dir)
    
    return times

def test_inplace():
    """测试原地操作：a *= 4.1"""
    numpack_dir = "test_npk_inplace"
    npk = NumPack(numpack_dir, drop_if_exists=True) 
    npk.open()
    
    arrays = {'a1': np.random.rand(1, 1000000)}
    npk.save(arrays)
    
    times = []
    for i in range(20):
        t0 = time.time()
        
        # 原地操作（实际上也会创建新数组）
        a = npk.load('a1', lazy=True)
        a *= 4.1  # Python 自动转换为 a = a * 4.1
        npk.save({'a1': a})
        
        times.append(time.time() - t0)
    
    npk.close()
    shutil.rmtree(numpack_dir)
    
    return times

print("=" * 60)
print("性能对比测试")
print("=" * 60)

print("\n测试1: 非原地操作 (a = a * 4.1)")
times_non_inplace = test_non_inplace()
avg_non = np.mean(times_non_inplace)
std_non = np.std(times_non_inplace)
print(f"平均耗时: {avg_non*1000:.3f}ms ± {std_non*1000:.3f}ms")

print("\n测试2: 原地操作 (a *= 4.1)")
times_inplace = test_inplace()
avg_in = np.mean(times_inplace)
std_in = np.std(times_inplace)
print(f"平均耗时: {avg_in*1000:.3f}ms ± {std_in*1000:.3f}ms")

print("\n" + "=" * 60)
print("结论：")
diff_pct = abs(avg_in - avg_non) / avg_non * 100
if diff_pct < 5:
    print(f"✅ 两种操作性能相同（差异 {diff_pct:.1f}%）")
else:
    print(f"⚠️  性能差异：{diff_pct:.1f}%")
    if avg_in > avg_non:
        print(f"   原地操作慢 {(avg_in/avg_non - 1)*100:.1f}%")
    else:
        print(f"   非原地操作慢 {(avg_non/avg_in - 1)*100:.1f}%")

print("\n重要说明：")
print("1. 对于 LazyArray，两种写法本质上是相同的")
print("2. 'a *= 4.1' 会自动转换为 'a = a * 4.1'")
print("3. 都会将 LazyArray 转换为 numpy 数组")
print("=" * 60)

