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

print("使用save方法:")
save_times = []
for i in range(100):
    c = random.choice(foo)
    a = npk.load(c, lazy=False)
    b = a * 4.1
    
    start = time.time()
    npk.save({c:b}) 
    end = time.time()
    save_times.append(end - start)

print(f"save 平均时间: {np.mean(save_times)*1000:.2f} ms")

print("\n使用replace方法:")
replace_times = []
for i in range(100):
    c = random.choice(foo)
    a = npk.load(c, lazy=False)
    b = a * 4.1
    
    start = time.time()
    npk.replace({c:b}, np.arange(len(b)))
    end = time.time()
    replace_times.append(end - start)

print(f"replace 平均时间: {np.mean(replace_times)*1000:.2f} ms")

print(f"\n性能比较: save比replace慢 {np.mean(save_times)/np.mean(replace_times):.2f}x")
print(f"性能提升: save速度提升了 {(1 - np.mean(save_times)/4.55)*100:.1f}%")

npk.close()

