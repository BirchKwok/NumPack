import os
import numpy as np
import tempfile
from numpack import save_nnp, load_nnp, replace_arrays, append_arrays, drop_arrays

def test_basic_operations():
    # 创建测试数据
    array1 = np.random.rand(100, 100).astype(np.float32)
    array2 = np.random.rand(50, 200).astype(np.float32)
    arrays = {
        'array1': array1,
        'array2': array2
    }
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.nnp', delete=False) as f:
        filename = f.name
    
    try:
        # 保存数组
        print(f"保存数组到文件: {filename}")
        print(f"array1 前5个元素: {array1.flatten()[:5]}")
        save_nnp(filename, arrays)
        
        # 加载数组
        loaded_arrays = load_nnp(filename)
        loaded_array1 = loaded_arrays['array1']
        loaded_array2 = loaded_arrays['array2']
        
        # print(f"加载的 array1 前5个元素: {loaded_array1.flatten()[:5]}")
        # print(f"array1 和 loaded_array1 的最大差值: {np.max(np.abs(array1 - loaded_array1))}")
        
        # 验证数组内容
        assert np.array_equal(array1, loaded_array1), "array1 的内容不匹配"
        assert np.array_equal(array2, loaded_array2), "array2 的内容不匹配"
        
        # 验证数组形状
        assert array1.shape == loaded_array1.shape, "array1 的形状不匹配"
        assert array2.shape == loaded_array2.shape, "array2 的形状不匹配"
        
        # 测试替换数组
        new_array1 = np.random.rand(100, 100).astype(np.float32)
        replace_arrays(filename, {'array1': new_array1}, slice(None))
        loaded_arrays = load_nnp(filename)
        assert np.array_equal(new_array1, loaded_arrays['array1']), "替换后的 array1 不匹配"
        
        # 测试追加数组
        array3 = np.random.rand(75, 150).astype(np.float32)
        append_arrays(filename, {'array3': array3})
        loaded_arrays = load_nnp(filename)
        assert np.array_equal(array3, loaded_arrays['array3']), "追加的 array3 不匹配"
        
        # 测试删除数组
        drop_arrays(filename, slice(None), ['array2'])
        loaded_arrays = load_nnp(filename)
        assert 'array2' not in loaded_arrays, "array2 未被删除"
        assert len(loaded_arrays) == 2, "数组数量不正确"
        
    finally:
        # 清理临时文件
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == '__main__':
    test_basic_operations() 