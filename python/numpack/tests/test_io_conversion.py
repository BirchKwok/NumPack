"""Tests for the NumPack I/O module.

This file tests conversions between various data formats and NumPack.
"""

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path

# Import I/O helpers
from numpack import NumPack
from numpack.io import (
    DependencyError,
    get_file_size,
    is_large_file,
    estimate_chunk_rows,
    from_numpy,
    to_numpy,
    from_csv,
    to_csv,
    from_txt,
    to_txt,
    LARGE_FILE_THRESHOLD,
)


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_get_file_size_file(self, tmp_path):
        """Test file size for a file."""
        # Create a test file
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"0" * 1000)
        
        size = get_file_size(test_file)
        assert size == 1000
    
    def test_get_file_size_directory(self, tmp_path):
        """Test total size for a directory."""
        # Create a test directory and files
        sub_dir = tmp_path / "subdir"
        sub_dir.mkdir()
        
        (tmp_path / "file1.bin").write_bytes(b"0" * 500)
        (sub_dir / "file2.bin").write_bytes(b"0" * 500)
        
        size = get_file_size(tmp_path)
        assert size == 1000
    
    def test_is_large_file(self, tmp_path):
        """Test large-file detection."""
        # Create a small file
        small_file = tmp_path / "small.bin"
        small_file.write_bytes(b"0" * 100)
        
        assert not is_large_file(small_file)
        assert is_large_file(small_file, threshold=50) == True  # 100 > 50
        assert is_large_file(small_file, threshold=200) == False  # 100 < 200
    
    def test_estimate_chunk_rows(self):
        """Test chunk row estimation."""
        # 1D array
        shape_1d = (10000,)
        dtype_f64 = np.dtype('float64')
        rows = estimate_chunk_rows(shape_1d, dtype_f64, 1024 * 1024)  # 1MB
        assert rows > 0
        
        # 2D array
        shape_2d = (10000, 100)
        rows = estimate_chunk_rows(shape_2d, dtype_f64, 1024 * 1024)  # 1MB
        assert rows > 0
        assert rows <= 10000


class TestNumpyConversion:
    """Tests for NumPy format conversion."""
    
    def test_from_npy_small(self, tmp_path):
        """Test importing from a small .npy file."""
        # Create test data
        arr = np.random.rand(100, 10).astype(np.float64)
        npy_path = tmp_path / "test.npy"
        npk_path = tmp_path / "test.npk"
        
        np.save(npy_path, arr)
        
        # Convert
        from_numpy(npy_path, npk_path, drop_if_exists=True)
        
        # Verify
        with NumPack(npk_path) as npk:
            loaded = npk.load("test")
            np.testing.assert_array_almost_equal(arr, loaded)
    
    def test_from_npy_with_name(self, tmp_path):
        """Test importing with an explicit array name."""
        arr = np.random.rand(50, 5).astype(np.float32)
        npy_path = tmp_path / "data.npy"
        npk_path = tmp_path / "output.npk"
        
        np.save(npy_path, arr)
        
        from_numpy(npy_path, npk_path, array_name="my_array", drop_if_exists=True)
        
        with NumPack(npk_path) as npk:
            loaded = npk.load("my_array")
            np.testing.assert_array_almost_equal(arr, loaded)
    
    def test_from_npz(self, tmp_path):
        """Test importing from a .npz file."""
        arr1 = np.random.rand(100, 10).astype(np.float64)
        arr2 = np.random.randint(0, 100, (50, 20)).astype(np.int32)
        
        npz_path = tmp_path / "test.npz"
        npk_path = tmp_path / "test.npk"
        
        np.savez(npz_path, array1=arr1, array2=arr2)
        
        from_numpy(npz_path, npk_path, drop_if_exists=True)
        
        with NumPack(npk_path) as npk:
            members = npk.get_member_list()
            assert "array1" in members
            assert "array2" in members
            
            loaded1 = npk.load("array1")
            loaded2 = npk.load("array2")
            
            np.testing.assert_array_almost_equal(arr1, loaded1)
            np.testing.assert_array_equal(arr2, loaded2)
    
    def test_to_npy(self, tmp_path):
        """Test exporting to a .npy file."""
        arr = np.random.rand(100, 10).astype(np.float64)
        npk_path = tmp_path / "test.npk"
        npy_path = tmp_path / "output.npy"
        
        # Create NumPack file
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({"data": arr})
        
        # Export
        to_numpy(npk_path, npy_path, array_names=["data"])
        
        # Verify
        loaded = np.load(npy_path)
        np.testing.assert_array_almost_equal(arr, loaded)
    
    def test_to_npz(self, tmp_path):
        """Test exporting to a .npz file."""
        arr1 = np.random.rand(100, 10).astype(np.float64)
        arr2 = np.random.randint(0, 100, (50, 20)).astype(np.int32)
        
        npk_path = tmp_path / "test.npk"
        npz_path = tmp_path / "output.npz"
        
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({"array1": arr1, "array2": arr2})
        
        to_numpy(npk_path, npz_path)
        
        with np.load(npz_path) as data:
            np.testing.assert_array_almost_equal(arr1, data["array1"])
            np.testing.assert_array_equal(arr2, data["array2"])
    
    def test_roundtrip_npy(self, tmp_path):
        """Test round-trip conversion for .npy."""
        original = np.random.rand(200, 50).astype(np.float32)
        
        npy1 = tmp_path / "original.npy"
        npk_path = tmp_path / "intermediate.npk"
        npy2 = tmp_path / "final.npy"
        
        np.save(npy1, original)
        from_numpy(npy1, npk_path, drop_if_exists=True)
        to_numpy(npk_path, npy2, array_names=["original"])
        
        final = np.load(npy2)
        np.testing.assert_array_almost_equal(original, final)


class TestCsvConversion:
    """Tests for CSV format conversion."""
    
    def test_from_csv_small(self, tmp_path):
        """Test importing from a small CSV file."""
        # Create test CSV
        arr = np.random.rand(100, 5).astype(np.float64)
        csv_path = tmp_path / "test.csv"
        npk_path = tmp_path / "test.npk"
        
        np.savetxt(csv_path, arr, delimiter=',')
        
        # pandas treats the first line as a header by default, so pass header=None
        from_csv(csv_path, npk_path, drop_if_exists=True, header=None)
        
        with NumPack(npk_path) as npk:
            loaded = npk.load("test")
            np.testing.assert_array_almost_equal(arr, loaded)
    
    def test_from_csv_with_delimiter(self, tmp_path):
        """Test importing with a custom delimiter."""
        arr = np.random.rand(50, 3).astype(np.float64)
        csv_path = tmp_path / "test.csv"
        npk_path = tmp_path / "test.npk"
        
        np.savetxt(csv_path, arr, delimiter=';')
        
        from_csv(csv_path, npk_path, delimiter=';', drop_if_exists=True)
        
        with NumPack(npk_path) as npk:
            loaded = npk.load("test")
            np.testing.assert_array_almost_equal(arr, loaded)
    
    def test_to_csv(self, tmp_path):
        """Test exporting to CSV."""
        arr = np.random.rand(100, 5).astype(np.float64)
        npk_path = tmp_path / "test.npk"
        csv_path = tmp_path / "output.csv"
        
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({"data": arr})
        
        to_csv(npk_path, csv_path, array_name="data", fmt='%.10e')
        
        loaded = np.loadtxt(csv_path, delimiter=',')
        np.testing.assert_array_almost_equal(arr, loaded, decimal=8)
    
    def test_roundtrip_csv(self, tmp_path):
        """Test round-trip conversion for CSV."""
        original = np.random.rand(50, 4).astype(np.float64)
        
        csv1 = tmp_path / "original.csv"
        npk_path = tmp_path / "intermediate.npk"
        csv2 = tmp_path / "final.csv"
        
        np.savetxt(csv1, original, delimiter=',', fmt='%.15e')
        from_csv(csv1, npk_path, drop_if_exists=True)
        to_csv(npk_path, csv2, fmt='%.15e')
        
        final = np.loadtxt(csv2, delimiter=',')
        np.testing.assert_array_almost_equal(original, final, decimal=10)


class TestTxtConversion:
    """Tests for whitespace-delimited TXT conversion."""
    
    def test_from_txt(self, tmp_path):
        """Test importing from a text file."""
        arr = np.random.rand(50, 3).astype(np.float64)
        txt_path = tmp_path / "test.txt"
        npk_path = tmp_path / "test.npk"
        
        np.savetxt(txt_path, arr, delimiter=' ')
        
        from_txt(txt_path, npk_path, drop_if_exists=True)
        
        with NumPack(npk_path) as npk:
            loaded = npk.load("test")
            np.testing.assert_array_almost_equal(arr, loaded)
    
    def test_to_txt(self, tmp_path):
        """Test exporting to a text file."""
        arr = np.random.rand(50, 3).astype(np.float64)
        npk_path = tmp_path / "test.npk"
        txt_path = tmp_path / "output.txt"
        
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({"data": arr})
        
        to_txt(npk_path, txt_path, array_name="data", fmt='%.10e')
        
        loaded = np.loadtxt(txt_path)
        np.testing.assert_array_almost_equal(arr, loaded, decimal=8)


class TestDifferentDtypes:
    """Tests for different dtypes."""
    
    @pytest.mark.parametrize("dtype", [
        np.float32, np.float64, 
        np.int32, np.int64,
        np.uint8, np.uint16, np.uint32,
    ])
    def test_numpy_roundtrip_dtypes(self, tmp_path, dtype):
        """Test round-trip conversion for different dtypes."""
        if np.issubdtype(dtype, np.integer):
            arr = np.random.randint(0, 100, (50, 10)).astype(dtype)
        else:
            arr = np.random.rand(50, 10).astype(dtype)
        
        npy_path = tmp_path / "test.npy"
        npk_path = tmp_path / "test.npk"
        npy_out = tmp_path / "output.npy"
        
        np.save(npy_path, arr)
        from_numpy(npy_path, npk_path, drop_if_exists=True)
        to_numpy(npk_path, npy_out, array_names=["test"])
        
        loaded = np.load(npy_out)
        assert loaded.dtype == arr.dtype
        np.testing.assert_array_equal(arr, loaded)


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_file_not_found(self, tmp_path):
        """Test FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            from_numpy(tmp_path / "nonexistent.npy", tmp_path / "out.npk")
    
    def test_invalid_format(self, tmp_path):
        """Test unsupported format."""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("test")
        
        with pytest.raises(ValueError):
            from_numpy(test_file, tmp_path / "out.npk")
    
    def test_npy_multiple_arrays_error(self, tmp_path):
        """Test error when exporting multiple arrays to .npy."""
        arr1 = np.random.rand(10, 5)
        arr2 = np.random.rand(20, 3)
        
        npk_path = tmp_path / "test.npk"
        npy_path = tmp_path / "output.npy"
        
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({"arr1": arr1, "arr2": arr2})
        
        with pytest.raises(ValueError, match=r"can only store one array"):
            to_numpy(npk_path, npy_path)


class TestOptionalDependencies:
    """Tests for optional dependency handling."""
    
    def test_dependency_error_message(self):
        """Test DependencyError message."""
        error = DependencyError("test message")
        assert "test message" in str(error)


# Optional dependency tests (run only when the corresponding library is installed)

class TestHdf5Conversion:
    """Tests for HDF5 conversion (requires h5py)."""
    
    @pytest.fixture(autouse=True)
    def check_h5py(self):
        """Check whether h5py is installed."""
        pytest.importorskip("h5py")
    
    def test_from_hdf5(self, tmp_path):
        """Test importing from HDF5."""
        import h5py
        from numpack.io import from_hdf5
        
        arr = np.random.rand(100, 10).astype(np.float64)
        h5_path = tmp_path / "test.h5"
        npk_path = tmp_path / "test.npk"
        
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('data', data=arr)
        
        from_hdf5(h5_path, npk_path, drop_if_exists=True)
        
        with NumPack(npk_path) as npk:
            loaded = npk.load("data")
            np.testing.assert_array_almost_equal(arr, loaded)
    
    def test_to_hdf5(self, tmp_path):
        """Test exporting to HDF5."""
        import h5py
        from numpack.io import to_hdf5
        
        arr = np.random.rand(100, 10).astype(np.float64)
        npk_path = tmp_path / "test.npk"
        h5_path = tmp_path / "output.h5"
        
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({"data": arr})
        
        to_hdf5(npk_path, h5_path)
        
        with h5py.File(h5_path, 'r') as f:
            loaded = f['data'][...]
            np.testing.assert_array_almost_equal(arr, loaded)


class TestPandasConversion:
    """Tests for pandas conversion (requires pandas)."""
    
    @pytest.fixture(autouse=True)
    def check_pandas(self):
        """Check whether pandas is installed."""
        pytest.importorskip("pandas")
    
    def test_from_pandas(self, tmp_path):
        """Test importing from a DataFrame."""
        import pandas as pd
        from numpack.io import from_pandas
        
        df = pd.DataFrame({
            'a': np.random.rand(100),
            'b': np.random.rand(100),
            'c': np.random.rand(100)
        })
        npk_path = tmp_path / "test.npk"
        
        from_pandas(df, npk_path, drop_if_exists=True)
        
        with NumPack(npk_path) as npk:
            loaded = npk.load("data")
            np.testing.assert_array_almost_equal(df.values, loaded)
    
    def test_to_pandas(self, tmp_path):
        """Test exporting to a DataFrame."""
        import pandas as pd
        from numpack.io import to_pandas
        
        arr = np.random.rand(100, 3).astype(np.float64)
        npk_path = tmp_path / "test.npk"
        
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({"data": arr})
        
        df = to_pandas(npk_path)
        np.testing.assert_array_almost_equal(arr, df.values)


class TestParquetConversion:
    """Tests for Parquet conversion (requires pyarrow)."""
    
    @pytest.fixture(autouse=True)
    def check_pyarrow(self):
        """Check whether pyarrow is installed."""
        pytest.importorskip("pyarrow")
    
    def test_from_parquet(self, tmp_path):
        """Test importing from Parquet."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from numpack.io import from_parquet
        
        arr = np.random.rand(100, 5).astype(np.float64)
        pq_path = tmp_path / "test.parquet"
        npk_path = tmp_path / "test.npk"
        
        table = pa.table({f'col{i}': arr[:, i] for i in range(arr.shape[1])})
        pq.write_table(table, pq_path)
        
        from_parquet(pq_path, npk_path, drop_if_exists=True)
        
        with NumPack(npk_path) as npk:
            loaded = npk.load("test")
            np.testing.assert_array_almost_equal(arr, loaded)
    
    def test_to_parquet(self, tmp_path):
        """Test exporting to Parquet."""
        import pyarrow.parquet as pq
        from numpack.io import to_parquet
        
        arr = np.random.rand(100, 5).astype(np.float64)
        npk_path = tmp_path / "test.npk"
        pq_path = tmp_path / "output.parquet"
        
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({"data": arr})
        
        to_parquet(npk_path, pq_path)
        
        table = pq.read_table(pq_path)
        loaded = table.to_pandas().values
        np.testing.assert_array_almost_equal(arr, loaded)


class TestZarrConversion:
    """Tests for Zarr conversion (requires zarr)."""
    
    @pytest.fixture(autouse=True)
    def check_zarr(self):
        """Check whether zarr is installed."""
        pytest.importorskip("zarr")
    
    def test_from_zarr(self, tmp_path):
        """Test importing from Zarr."""
        import zarr
        from numpack.io import from_zarr
        
        arr = np.random.rand(100, 10).astype(np.float64)
        zarr_path = tmp_path / "test.zarr"
        npk_path = tmp_path / "test.npk"
        
        store = zarr.open(str(zarr_path), mode='w')
        if hasattr(store, 'create_array'):
            zarr_arr = store.create_array('data', data=arr, overwrite=True)
        else:
            zarr_arr = store.create_dataset('data', data=arr)
        
        from_zarr(zarr_path, npk_path, drop_if_exists=True)
        
        with NumPack(npk_path) as npk:
            loaded = npk.load("data")
            np.testing.assert_array_almost_equal(arr, loaded)
    
    def test_to_zarr(self, tmp_path):
        """Test exporting to Zarr."""
        import zarr
        from numpack.io import to_zarr
        
        arr = np.random.rand(100, 10).astype(np.float64)
        npk_path = tmp_path / "test.npk"
        zarr_path = tmp_path / "output.zarr"
        
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({"data": arr})
        
        to_zarr(npk_path, zarr_path)
        
        store = zarr.open(str(zarr_path), mode='r')
        loaded = store['data'][...]
        np.testing.assert_array_almost_equal(arr, loaded)


class TestPytorchConversion:
    """Tests for PyTorch conversion (requires torch)."""
    
    @pytest.fixture(autouse=True)
    def check_torch(self):
        """Check whether torch is installed."""
        pytest.importorskip("torch")
    
    def test_from_pytorch(self, tmp_path):
        """Test importing from PyTorch."""
        import torch
        from numpack.io import from_pytorch
        
        tensor = torch.rand(100, 10, dtype=torch.float32)
        pt_path = tmp_path / "test.pt"
        npk_path = tmp_path / "test.npk"
        
        torch.save(tensor, pt_path)
        
        from_pytorch(pt_path, npk_path, drop_if_exists=True)
        
        with NumPack(npk_path) as npk:
            loaded = npk.load("test")
            np.testing.assert_array_almost_equal(tensor.numpy(), loaded)
    
    def test_from_pytorch_dict(self, tmp_path):
        """Test importing from a PyTorch dict."""
        import torch
        from numpack.io import from_pytorch
        
        tensors = {
            'features': torch.rand(100, 10),
            'labels': torch.randint(0, 10, (100,))
        }
        pt_path = tmp_path / "test.pt"
        npk_path = tmp_path / "test.npk"
        
        torch.save(tensors, pt_path)
        
        from_pytorch(pt_path, npk_path, drop_if_exists=True)
        
        with NumPack(npk_path) as npk:
            members = npk.get_member_list()
            assert 'features' in members
            assert 'labels' in members
    
    def test_to_pytorch(self, tmp_path):
        """Test exporting to PyTorch."""
        import torch
        from numpack.io import to_pytorch
        
        arr = np.random.rand(100, 10).astype(np.float32)
        npk_path = tmp_path / "test.npk"
        pt_path = tmp_path / "output.pt"
        
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({"data": arr})
        
        to_pytorch(npk_path, pt_path)
        
        loaded = torch.load(pt_path, weights_only=False)
        assert isinstance(loaded, dict)
        np.testing.assert_array_almost_equal(arr, loaded['data'].numpy())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
