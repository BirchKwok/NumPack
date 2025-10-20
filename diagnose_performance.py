"""诊断性能问题"""

import random
import numpy as np
import time
from numpack import NumPack

numpack_dir = "test_npk_perf"
npk = NumPack(numpack_dir, drop_if_exists=True) 
npk.open()
    
print("创建测试数据...")
arrays = {
    'a1': np.random.rand(1, 1000000),
    'a2': np.random.rand(1, 1000000),
    'a3': np.random.rand(1, 1000000),
}

print("保存初始数据...")
start = time.time()
npk.save(arrays)
print(f"初始保存耗时: {time.time() - start:.4f}s")

foo = ['a1', 'a2', 'a3']

print("\n开始循环测试...")
times = []

for i in range(10):  # 先测试10次
    c = random.choice(foo)
    
    # 计时：加载
    t0 = time.time()
    a = npk.load(c, lazy=True)
    t_load = time.time() - t0
    
    # 检查类型
    print(f"\n第 {i+1} 次循环，数组: {c}")
    print(f"  加载后类型: {type(a)}")
    print(f"  加载耗时: {t_load:.6f}s")
    
    # 计时：乘法
    t0 = time.time()
    a_result = a * 4.1
    t_mul = time.time() - t0
    print(f"  乘法后类型: {type(a_result)}")
    print(f"  乘法耗时: {t_mul:.6f}s")
    print(f"  数据形状: {a_result.shape}")
    
    # 计时：保存
    t0 = time.time()
    npk.save({c: a_result})
    t_save = time.time() - t0
    print(f"  保存耗时: {t_save:.6f}s")
    
    times.append({
        'load': t_load,
        'mul': t_mul,
        'save': t_save,
        'total': t_load + t_mul + t_save
    })

print("\n=== 性能统计 ===")
avg_load = np.mean([t['load'] for t in times])
avg_mul = np.mean([t['mul'] for t in times])
avg_save = np.mean([t['save'] for t in times])
avg_total = np.mean([t['total'] for t in times])

print(f"平均加载时间: {avg_load:.6f}s")
print(f"平均乘法时间: {avg_mul:.6f}s")
print(f"平均保存时间: {avg_save:.6f}s")
print(f"平均总时间: {avg_total:.6f}s")

npk.close()

# 清理
import shutil
shutil.rmtree(numpack_dir)

