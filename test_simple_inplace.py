"""简化的原地操作符测试"""

import tempfile
import numpy as np
import numpack as npk
from pathlib import Path
import shutil

def test_simple():
    """最简单的测试"""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp) / 'test'
        test_dir.mkdir()
        
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        with npk.NumPack(test_dir) as pack:
            pack.save({'array': data})
        
        with npk.NumPack(test_dir) as pack:
            a = pack.load('array', lazy=True)
            print(f'\nBefore: type={type(a)}, has_imul={hasattr(a, "__imul__")}')
            a *= 2.0
            print(f'After: type={type(a)}')
            assert isinstance(a, np.ndarray)
            expected = data * 2.0
            np.testing.assert_array_equal(a, expected)

if __name__ == "__main__":
    test_simple()
    print("✅ 测试通过!")

