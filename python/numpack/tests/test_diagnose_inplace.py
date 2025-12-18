"""Diagnose issues related to in-place operators."""

import tempfile
import numpy as np
import numpack as npk
from pathlib import Path

def test_diagnose():
    """Diagnostic test."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp) / 'test'
        test_dir.mkdir()
        
        data = np.array([[1, 2]], dtype=np.float32)
        with npk.NumPack(test_dir) as pack:
            pack.save({'array': data})
        
        with npk.NumPack(test_dir) as pack:
            a = pack.load('array', lazy=True)
            
            print(f'\n=== Diagnostic Info ===')
            print(f'Type: {type(a)}')
            print(f'Type name: {type(a).__name__}')
            print(f'Type module: {type(a).__module__}')
            print(f'Has __imul__: {hasattr(a, "__imul__")}')
            print(f'Has __mul__: {hasattr(a, "__mul__")}')
            
            # Check MRO
            print(f'\\nMRO: {[c.__name__ for c in type(a).__mro__]}')
            
            # Try to get __imul__ directly
            try:
                imul = getattr(type(a), '__imul__', None)
                print(f'__imul__ from type: {imul}')
            except:
                print('Cannot get __imul__ from type')
            
            # Try multiplication first
            print(f'\\nTrying a * 2...')
            result_mul = a * 2
            print(f'Success: type={type(result_mul)}')
            
            # Now try in-place
            print(f'\\nTrying a *= 2...')
            b = pack.load('array', lazy=True)
            try:
                b *= 2
                print(f'Success: type={type(b)}')
            except Exception as e:
                print(f'Failed: {type(e).__name__}: {e}')
                import sys
                import traceback
                traceback.print_exc()

