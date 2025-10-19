import random
import numpy as np
from numpack import NumPack
import time

numpack_dir = "test_npk"
npk = NumPack(numpack_dir, drop_if_exists=True) 
npk.open()
    
arrays = {
    'a1': np.random.rand(1, 1000000),
    'a2': np.random.rand(1, 1000000),
    'a3': np.random.rand(1, 1000000),
}

npk.save(arrays)

foo = ['a1', 'a2', 'a3']

# 测试save的性能
print("测试 save 方法:")
save_times = []
for i in range(10):
    c = random.choice(foo)
    a = npk.load(c, lazy=False)
    b = a * 4.1
    
    start = time.time()
    npk.save({c:b}) 
    end = time.time()
    save_times.append(end - start)
    print(f"  轮次 {i+1}: {(end-start)*1000:.2f} ms")

print(f"save 平均时间: {np.mean(save_times)*1000:.2f} ms")

# 测试replace的性能
print("\n测试 replace 方法:")
replace_times = []
for i in range(10):
    c = random.choice(foo)
    a = npk.load(c, lazy=False)
    b = a * 4.1
    
    start = time.time()
    npk.replace({c:b}, np.arange(len(b)))
    end = time.time()
    replace_times.append(end - start)
    print(f"  轮次 {i+1}: {(end-start)*1000:.2f} ms")

print(f"replace 平均时间: {np.mean(replace_times)*1000:.2f} ms")

print(f"\n性能比较: save比replace慢 {np.mean(save_times)/np.mean(replace_times):.2f}x")

npk.close()

