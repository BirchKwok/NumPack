import os
import numpy as np
import tempfile
import filelock
from concurrent.futures import ThreadPoolExecutor
from numpack import NumPack


def test_basic_operations(temp_dir=None):
    """基本操作测试"""
    # 创建测试数据
    array1 = np.random.rand(100, 100).astype(np.float32)
    array2 = np.random.rand(50, 200).astype(np.float32)
    arrays = {
        'array1': array1,
        'array2': array2
    }
    
    # 使用传入的目录或创建新的临时目录
    if temp_dir is None:
        temp_context = tempfile.TemporaryDirectory()
        temp_dir = temp_context.name
    else:
        temp_context = None
    
    try:
        # 保存数组
        print(f"保存数组到目录: {temp_dir}")
        print(f"array1 前5个元素: {array1.flatten()[:5]}")
        npk = NumPack(temp_dir)
        # 清理目录
        npk.reset()

        # 保存数组
        npk.save_arrays(arrays)
        
        # 测试普通加载
        print("\n测试普通加载...")
        loaded_arrays = npk.load_arrays(mmap_mode=False)
        loaded_array1 = loaded_arrays['array1']
        loaded_array2 = loaded_arrays['array2']
        
        print(f"加载的 array1 前5个元素: {loaded_array1.flatten()[:5]}")
        print(f"array1 和 loaded_array1 的最大差值: {np.max(np.abs(array1 - loaded_array1))}")
        
        # 验证数组内容
        assert np.array_equal(array1, loaded_array1), "array1 的内容不匹配"
        assert np.array_equal(array2, loaded_array2), "array2 的内容不匹配"
        
        # 验证数组形状
        assert array1.shape == loaded_array1.shape, "array1 的形状不匹配"
        assert array2.shape == loaded_array2.shape, "array2 的形状不匹配"
        
        # 测试延迟加载
        print("\n测试延迟加载...")
        lazy_arrays = npk.load_arrays(mmap_mode=True)
        print(f"延迟加载对象类型: {type(lazy_arrays)}")
        print(f"可用数组: {list(lazy_arrays.keys())}")
        
        # 测试按需加载
        print("\n测试按需加载...")
        lazy_array1 = lazy_arrays['array1']
        print(f"按需加载的 array1 前5个元素: {lazy_array1.flatten()[:5]}")
        assert np.array_equal(array1, lazy_array1), "延迟加载的 array1 内容不匹配"
        
        # 测试选择性加载
        print("\n测试选择性加载...")
        partial_arrays = npk.load_arrays(['array1'], mmap_mode=False)
        assert len(partial_arrays) == 1, "选择性加载的数组数量不正确"
        assert np.array_equal(array1, partial_arrays['array1']), "选择性加载的 array1 内容不匹配"
        
        # 测试替换整个数组
        print("\n测试替换整个数组...")
        new_array1 = np.random.rand(100, 100).astype(np.float32)
        npk.replace_arrays({'array1': new_array1}, slice(None))
        loaded_arrays = npk.load_arrays(mmap_mode=False)
        assert np.array_equal(new_array1, loaded_arrays['array1']), "替换后的 array1 不匹配"
        print("替换整个数组测试通过")
        
        # 测试替换部分行
        print("\n测试替换部分行...")
        # 替换第 10-15 行
        new_rows = np.random.rand(5, 100).astype(np.float32)
        npk.replace_arrays({'array1': new_rows}, slice(10, 15))
        loaded_arrays = npk.load_arrays(mmap_mode=False)
        assert np.array_equal(new_rows, loaded_arrays['array1'][10:15]), "替换的行不匹配"
        assert np.array_equal(new_array1[:10], loaded_arrays['array1'][:10]), "未替换的行被修���"
        assert np.array_equal(new_array1[15:], loaded_arrays['array1'][15:]), "未替换的行被修改"
        print("替换部分行测试通过")
        
        # 测试替换指定行
        print("\n测试替换指定行...")
        indices = [20, 25, 30]
        new_rows = np.random.rand(len(indices), 100).astype(np.float32)
        # 保存当前状态用于验证
        current_array = loaded_arrays['array1'].copy()
        npk.replace_arrays({'array1': new_rows}, indices)
        loaded_arrays = npk.load_arrays(mmap_mode=False)
        assert np.array_equal(new_rows, loaded_arrays['array1'][indices]), "替换的指定行不匹配"
        # 验证其他行未被修改
        mask = np.ones(100, dtype=bool)
        mask[indices] = False
        assert np.array_equal(loaded_arrays['array1'][mask], current_array[mask]), "未替换的行被修改"
        print("替换指定行测试通过")
        
        # 测试追加数组
        print("\n测试追加数组...")
        array3 = np.random.rand(75, 150).astype(np.float32)
        npk.save_arrays({'array3': array3})
        loaded_arrays = npk.load_arrays(mmap_mode=False)
        assert np.array_equal(array3, loaded_arrays['array3']), "追加的 array3 不匹配"
        print("追加数组测试通过")
        
        # 测试删除数组
        print("\n测试删除数组...")
        npk.drop_arrays(['array2'])
        loaded_arrays = npk.load_arrays(mmap_mode=False)
        print(f"loaded_arrays: {loaded_arrays}")
        assert 'array2' not in loaded_arrays, "array2 未被删除"
        assert len(loaded_arrays) == 2, "数组数量不正确"
        print("删除数组测试通过")
        
        # 测试元数据
        print("\n测试元数据...")
        shape = npk.get_shape('array1')
        assert shape == (100, 100), f"array1 形状不正确: {shape}"
        
        members = npk.get_member_list()
        assert set(members) == {'array1', 'array3'}, f"成员列表不正确: {members}"
        
        mtime = npk.get_modify_time('array1')
        assert mtime is not None, "修改时间不应为空"
        print("元数据测试通过")

    finally:
        if temp_context is not None:
            temp_context.cleanup()

def test_multithreading(temp_dir=None):
    """多线程测试"""
    print("\n开始多线程测试...")
    
    # 使用传入的目录或创建新的临时目录
    if temp_dir is None:
        temp_context = tempfile.TemporaryDirectory()
        temp_dir = temp_context.name
    else:
        temp_context = None
    
    try:
        # 创建初始数据
        array1 = np.random.rand(100, 100).astype(np.float32)
        arrays = {'array1': array1}
        
        # 保存初始数组
        print(f"保存初始数组到目录: {temp_dir}")
        npk = NumPack(temp_dir)
        npk.reset()

        npk.save_arrays(arrays)
        
        def concurrent_read_write(thread_id):
            """并发读写测试函数"""
            try:
                print(f"线程 {thread_id} 开始执行")
                
                # 创建新数组
                new_array = np.random.rand(100, 100).astype(np.float32)
                array_name = f'array_thread_{thread_id}'
                
                # 获取独占锁并写入
                print(f"线程 {thread_id} 开始写入 {array_name}")
                with filelock.FileLock(os.path.join(temp_dir, "write.lock")):
                    npk.save_arrays({array_name: new_array})
                    print(f"线程 {thread_id} 成功写入 {array_name}")

                # 验证写入
                arrays = npk.load_arrays(mmap_mode=False)
                assert array_name in arrays, f"{array_name} 未找到"
                assert np.array_equal(new_array, arrays[array_name]), f"{array_name} 内容不匹配"
                print(f"线程 {thread_id} 验证成功")

                return True
            except Exception as e:
                print(f"线程 {thread_id} 发生错误: {str(e)}")
                return False
        
        # 创建线程池
        num_threads = 4
        print(f"创建 {num_threads} 个线程")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(concurrent_read_write, i) 
                      for i in range(num_threads)]
            
            # 等待所有线程完成
            results = [future.result() for future in futures]
            
        # 验证所有线程都成功完成
        assert all(results), "某些线程操作失败"
        
        # 验证最终文件状态
        final_arrays = npk.load_arrays(mmap_mode=False)
        assert len(final_arrays) == num_threads + 1, "最终数组数量不正确"
        print("多线程测试完成")

    finally:
        if temp_context is not None:
            temp_context.cleanup()


if __name__ == '__main__':
    # 为每个测试创建独立的临时目录
    with tempfile.TemporaryDirectory() as basic_test_dir, \
         tempfile.TemporaryDirectory() as thread_test_dir, \
         tempfile.TemporaryDirectory() as process_test_dir:
             
        print("运行基本操作测试...")
        test_basic_operations(basic_test_dir)
        print("基本操作测试通过！")
        
        print("\n运行多线程测试...")
        test_multithreading(thread_test_dir)
        print("多线程测试通过！")
