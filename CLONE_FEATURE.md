# Clone Feature Implementation

## Overview
Added `clone` function to NumPack Python API that allows users to clone existing arrays with their data and metadata.

## Implementation Details

### 1. Rust Layer (`src/lib.rs`)
- Added `clone` method in the `#[pymethods] impl NumPack` block (lines 4269-4308)
- The method performs the following operations:
  1. Validates that the source array exists
  2. Checks that the target array name is not already in use
  3. Loads the source array data (eager mode)
  4. Saves the data with the new target name
  5. Clears metadata cache for the new array

### 2. Python API (`python/numpack/__init__.py`)
- Added `clone` method to the `NumPack` class (lines 386-421)
- Provides user-friendly docstring with examples
- Wraps the Rust implementation with proper error handling

### 3. Tests (`python/numpack/tests/test_numpack.py`)
- Added comprehensive test suite with 155 test cases covering:
  - **test_clone_basic**: Basic cloning functionality for all data types and dimensions (70 tests)
  - **test_clone_independence**: Verifies cloned arrays can be modified independently (70 tests)
  - **test_clone_errors**: Tests error handling for edge cases (1 test)
  - **test_clone_with_metadata**: Validates metadata is correctly cloned (14 tests)

## Usage Example

```python
import numpy as np
from numpack import NumPack

with NumPack('data.npk') as npk:
    # Create and save original array
    original = np.array([[1, 2, 3], [4, 5, 6]])
    npk.save({'original': original})
    
    # Clone the array
    npk.clone('original', 'backup')
    
    # Modify the clone independently
    backup = npk.load('backup')
    backup *= 2
    npk.save({'backup': backup})
    
    # Original remains unchanged
    original_reloaded = npk.load('original')
    # original_reloaded is still [[1, 2, 3], [4, 5, 6]]
```

## Features

- **Complete Data Copy**: Both array data and metadata are fully copied
- **Independence**: Cloned arrays can be modified without affecting the original
- **Error Handling**: 
  - Raises `KeyError` if source array doesn't exist
  - Raises `ValueError` if target array name already exists
- **Type Support**: Works with all NumPack supported data types:
  - Boolean, integers (int8-64, uint8-64)
  - Floating point (float16, float32, float64)
  - Complex numbers (complex64, complex128)
- **Dimension Support**: Works with arrays of any dimension (1D to 5D+)

## Test Results

All 155 tests passed successfully:
- ✓ Basic cloning across all data types and dimensions
- ✓ Independence verification (modifications don't affect original)
- ✓ Error handling for non-existent arrays and duplicate names
- ✓ Metadata preservation

## Performance Considerations

- Uses eager loading internally to ensure complete data copy
- Memory requirement: Temporarily needs memory for the full array during cloning
- Recommended for arrays that fit comfortably in memory
- For very large arrays, consider using other approaches like file-level copying

## API Reference

### Python API

```python
def clone(source_name: str, target_name: str) -> None:
    """
    Clone an existing array to a new array
    
    Parameters:
        source_name (str): Name of the source array to clone
        target_name (str): Name for the cloned array
    
    Raises:
        KeyError: If source array doesn't exist
        ValueError: If target array already exists
    """
```

### Rust API

```rust
fn clone(&self, py: Python, source_name: &str, target_name: &str) -> PyResult<()>
```

## Integration

The feature is fully integrated into NumPack v0.4.4 and requires no additional dependencies.
