"""Comprehensive tests for return_npk_obj parameter in all IO functions.

This file tests that the return_npk_obj parameter works correctly across
all import functions in the numpack.io module.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from numpack import NumPack


class TestReturnNpkObjNumpy:
    """Tests for return_npk_obj in NumPy conversion functions."""

    def test_from_numpy_npy_return_none_by_default(self, tmp_path):
        """Test that from_numpy returns None by default."""
        from numpack.io import from_numpy

        arr = np.random.rand(50, 10).astype(np.float64)
        npy_path = tmp_path / "test.npy"
        npk_path = tmp_path / "test.npk"

        np.save(npy_path, arr)
        result = from_numpy(npy_path, npk_path, drop_if_exists=True)

        assert result is None

    def test_from_numpy_npy_return_npk_obj_false(self, tmp_path):
        """Test that from_numpy returns None when return_npk_obj=False."""
        from numpack.io import from_numpy

        arr = np.random.rand(50, 10).astype(np.float64)
        npy_path = tmp_path / "test.npy"
        npk_path = tmp_path / "test.npk"

        np.save(npy_path, arr)
        result = from_numpy(npy_path, npk_path, drop_if_exists=True, return_npk_obj=False)

        assert result is None

    def test_from_numpy_npy_return_npk_obj_true(self, tmp_path):
        """Test that from_numpy returns a valid NumPack object when return_npk_obj=True."""
        from numpack.io import from_numpy

        arr = np.random.rand(50, 10).astype(np.float64)
        npy_path = tmp_path / "test.npy"
        npk_path = tmp_path / "test.npk"

        np.save(npy_path, arr)
        result = from_numpy(npy_path, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert result is not None
            assert isinstance(result, NumPack)
            # Verify the NumPack object is opened and usable
            members = result.get_member_list()
            assert "test" in members
            loaded = result.load("test")
            np.testing.assert_array_almost_equal(arr, loaded)
        finally:
            result.close()

    def test_from_numpy_npz_return_npk_obj_true(self, tmp_path):
        """Test from_numpy with .npz file and return_npk_obj=True."""
        from numpack.io import from_numpy

        arr1 = np.random.rand(30, 5).astype(np.float64)
        arr2 = np.random.randint(0, 100, (20, 10)).astype(np.int32)
        npz_path = tmp_path / "test.npz"
        npk_path = tmp_path / "test.npk"

        np.savez(npz_path, array1=arr1, array2=arr2)
        result = from_numpy(npz_path, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            members = result.get_member_list()
            assert "array1" in members
            assert "array2" in members
            np.testing.assert_array_almost_equal(arr1, result.load("array1"))
            np.testing.assert_array_equal(arr2, result.load("array2"))
        finally:
            result.close()


class TestReturnNpkObjCsv:
    """Tests for return_npk_obj in CSV/TXT conversion functions."""

    def test_from_csv_return_none_by_default(self, tmp_path):
        """Test that from_csv returns None by default."""
        from numpack.io import from_csv

        arr = np.random.rand(50, 5).astype(np.float64)
        csv_path = tmp_path / "test.csv"
        npk_path = tmp_path / "test.npk"

        np.savetxt(csv_path, arr, delimiter=',')
        result = from_csv(csv_path, npk_path, drop_if_exists=True, header=None)

        assert result is None

    def test_from_csv_return_npk_obj_true(self, tmp_path):
        """Test that from_csv returns a valid NumPack object when return_npk_obj=True."""
        from numpack.io import from_csv

        arr = np.random.rand(50, 5).astype(np.float64)
        csv_path = tmp_path / "test.csv"
        npk_path = tmp_path / "test.npk"

        np.savetxt(csv_path, arr, delimiter=',')
        result = from_csv(csv_path, npk_path, drop_if_exists=True, header=None, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            loaded = result.load("test")
            np.testing.assert_array_almost_equal(arr, loaded)
        finally:
            result.close()

    def test_from_txt_return_none_by_default(self, tmp_path):
        """Test that from_txt returns None by default."""
        from numpack.io import from_txt

        arr = np.random.rand(50, 3).astype(np.float64)
        txt_path = tmp_path / "test.txt"
        npk_path = tmp_path / "test.npk"

        np.savetxt(txt_path, arr, delimiter=' ')
        result = from_txt(txt_path, npk_path, drop_if_exists=True)

        assert result is None

    def test_from_txt_return_npk_obj_true(self, tmp_path):
        """Test that from_txt returns a valid NumPack object when return_npk_obj=True."""
        from numpack.io import from_txt

        arr = np.random.rand(50, 3).astype(np.float64)
        txt_path = tmp_path / "test.txt"
        npk_path = tmp_path / "test.npk"

        np.savetxt(txt_path, arr, delimiter=' ')
        result = from_txt(txt_path, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            loaded = result.load("test")
            np.testing.assert_array_almost_equal(arr, loaded)
        finally:
            result.close()


class TestReturnNpkObjHdf5:
    """Tests for return_npk_obj in HDF5 conversion functions."""

    @pytest.fixture(autouse=True)
    def check_h5py(self):
        """Check whether h5py is installed."""
        pytest.importorskip("h5py")

    def test_from_hdf5_return_none_by_default(self, tmp_path):
        """Test that from_hdf5 returns None by default."""
        import h5py
        from numpack.io import from_hdf5

        arr = np.random.rand(50, 10).astype(np.float64)
        h5_path = tmp_path / "test.h5"
        npk_path = tmp_path / "test.npk"

        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('data', data=arr)

        result = from_hdf5(h5_path, npk_path, drop_if_exists=True)
        assert result is None

    def test_from_hdf5_return_npk_obj_true(self, tmp_path):
        """Test that from_hdf5 returns a valid NumPack object when return_npk_obj=True."""
        import h5py
        from numpack.io import from_hdf5

        arr = np.random.rand(50, 10).astype(np.float64)
        h5_path = tmp_path / "test.h5"
        npk_path = tmp_path / "test.npk"

        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('data', data=arr)

        result = from_hdf5(h5_path, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            loaded = result.load("data")
            np.testing.assert_array_almost_equal(arr, loaded)
        finally:
            result.close()

    def test_from_hdf5_multiple_datasets_return_npk_obj(self, tmp_path):
        """Test from_hdf5 with multiple datasets and return_npk_obj=True."""
        import h5py
        from numpack.io import from_hdf5

        arr1 = np.random.rand(30, 5).astype(np.float64)
        arr2 = np.random.randint(0, 100, (20, 8)).astype(np.int32)
        h5_path = tmp_path / "test.h5"
        npk_path = tmp_path / "test.npk"

        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('features', data=arr1)
            f.create_dataset('labels', data=arr2)

        result = from_hdf5(h5_path, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            members = result.get_member_list()
            assert "features" in members
            assert "labels" in members
            np.testing.assert_array_almost_equal(arr1, result.load("features"))
            np.testing.assert_array_equal(arr2, result.load("labels"))
        finally:
            result.close()


class TestReturnNpkObjZarr:
    """Tests for return_npk_obj in Zarr conversion functions."""

    @pytest.fixture(autouse=True)
    def check_zarr(self):
        """Check whether zarr is installed."""
        pytest.importorskip("zarr")

    def test_from_zarr_return_none_by_default(self, tmp_path):
        """Test that from_zarr returns None by default."""
        import zarr
        from numpack.io import from_zarr

        arr = np.random.rand(50, 10).astype(np.float64)
        zarr_path = tmp_path / "test.zarr"
        npk_path = tmp_path / "test.npk"

        store = zarr.open(str(zarr_path), mode='w')
        if hasattr(store, 'create_array'):
            store.create_array('data', data=arr, overwrite=True)
        else:
            store.create_dataset('data', data=arr)

        result = from_zarr(zarr_path, npk_path, drop_if_exists=True)
        assert result is None

    def test_from_zarr_return_npk_obj_true(self, tmp_path):
        """Test that from_zarr returns a valid NumPack object when return_npk_obj=True."""
        import zarr
        from numpack.io import from_zarr

        arr = np.random.rand(50, 10).astype(np.float64)
        zarr_path = tmp_path / "test.zarr"
        npk_path = tmp_path / "test.npk"

        store = zarr.open(str(zarr_path), mode='w')
        if hasattr(store, 'create_array'):
            store.create_array('data', data=arr, overwrite=True)
        else:
            store.create_dataset('data', data=arr)

        result = from_zarr(zarr_path, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            loaded = result.load("data")
            np.testing.assert_array_almost_equal(arr, loaded)
        finally:
            result.close()


class TestReturnNpkObjPandas:
    """Tests for return_npk_obj in Pandas conversion functions."""

    @pytest.fixture(autouse=True)
    def check_pandas(self):
        """Check whether pandas is installed."""
        pytest.importorskip("pandas")

    def test_from_pandas_return_none_by_default(self, tmp_path):
        """Test that from_pandas returns None by default."""
        import pandas as pd
        from numpack.io import from_pandas

        df = pd.DataFrame({
            'a': np.random.rand(50),
            'b': np.random.rand(50),
        })
        npk_path = tmp_path / "test.npk"

        result = from_pandas(df, npk_path, drop_if_exists=True)
        assert result is None

    def test_from_pandas_return_npk_obj_true(self, tmp_path):
        """Test that from_pandas returns a valid NumPack object when return_npk_obj=True."""
        import pandas as pd
        from numpack.io import from_pandas

        df = pd.DataFrame({
            'a': np.random.rand(50),
            'b': np.random.rand(50),
        })
        npk_path = tmp_path / "test.npk"

        result = from_pandas(df, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            loaded = result.load("data")
            np.testing.assert_array_almost_equal(df.values, loaded)
        finally:
            result.close()


class TestReturnNpkObjParquet:
    """Tests for return_npk_obj in Parquet conversion functions."""

    @pytest.fixture(autouse=True)
    def check_pyarrow(self):
        """Check whether pyarrow is installed."""
        pytest.importorskip("pyarrow")

    def test_from_parquet_file_return_none_by_default(self, tmp_path):
        """Test that from_parquet_file returns None by default."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from numpack.io import from_parquet_file

        arr = np.random.rand(50, 5).astype(np.float64)
        pq_path = tmp_path / "test.parquet"
        npk_path = tmp_path / "test.npk"

        table = pa.table({f'col{i}': arr[:, i] for i in range(arr.shape[1])})
        pq.write_table(table, pq_path)

        result = from_parquet_file(pq_path, npk_path, drop_if_exists=True)
        assert result is None

    def test_from_parquet_file_return_npk_obj_true(self, tmp_path):
        """Test that from_parquet_file returns a valid NumPack object when return_npk_obj=True."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from numpack.io import from_parquet_file

        arr = np.random.rand(50, 5).astype(np.float64)
        pq_path = tmp_path / "test.parquet"
        npk_path = tmp_path / "test.npk"

        table = pa.table({f'col{i}': arr[:, i] for i in range(arr.shape[1])})
        pq.write_table(table, pq_path)

        result = from_parquet_file(pq_path, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            loaded = result.load("test")
            np.testing.assert_array_almost_equal(arr, loaded)
        finally:
            result.close()

    def test_from_parquet_legacy_alias_return_npk_obj_true(self, tmp_path):
        """Test the legacy from_parquet alias with return_npk_obj=True."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from numpack.io import from_parquet

        arr = np.random.rand(30, 4).astype(np.float64)
        pq_path = tmp_path / "test.parquet"
        npk_path = tmp_path / "test.npk"

        table = pa.table({f'col{i}': arr[:, i] for i in range(arr.shape[1])})
        pq.write_table(table, pq_path)

        result = from_parquet(pq_path, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            loaded = result.load("test")
            np.testing.assert_array_almost_equal(arr, loaded)
        finally:
            result.close()

    def test_from_parquet_table_return_npk_obj_true(self, tmp_path):
        """Test that from_parquet_table returns a valid NumPack object when return_npk_obj=True."""
        import pyarrow as pa
        from numpack.io import from_parquet_table

        arr = np.random.rand(50, 5).astype(np.float64)
        npk_path = tmp_path / "test.npk"

        table = pa.table({f'col{i}': arr[:, i] for i in range(arr.shape[1])})

        result = from_parquet_table(table, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            loaded = result.load("data")
            np.testing.assert_array_almost_equal(arr, loaded)
        finally:
            result.close()


class TestReturnNpkObjFeather:
    """Tests for return_npk_obj in Feather/Arrow conversion functions."""

    @pytest.fixture(autouse=True)
    def check_pyarrow(self):
        """Check whether pyarrow is installed."""
        pytest.importorskip("pyarrow")

    def test_from_feather_file_return_none_by_default(self, tmp_path):
        """Test that from_feather_file returns None by default."""
        import pyarrow as pa
        import pyarrow.feather as feather
        from numpack.io import from_feather_file

        arr = np.random.rand(50, 5).astype(np.float64)
        feather_path = tmp_path / "test.feather"
        npk_path = tmp_path / "test.npk"

        table = pa.table({f'col{i}': arr[:, i] for i in range(arr.shape[1])})
        feather.write_feather(table, feather_path)

        result = from_feather_file(feather_path, npk_path, drop_if_exists=True)
        assert result is None

    def test_from_feather_file_return_npk_obj_true(self, tmp_path):
        """Test that from_feather_file returns a valid NumPack object when return_npk_obj=True."""
        import pyarrow as pa
        import pyarrow.feather as feather
        from numpack.io import from_feather_file

        arr = np.random.rand(50, 5).astype(np.float64)
        feather_path = tmp_path / "test.feather"
        npk_path = tmp_path / "test.npk"

        table = pa.table({f'col{i}': arr[:, i] for i in range(arr.shape[1])})
        feather.write_feather(table, feather_path)

        result = from_feather_file(feather_path, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            loaded = result.load("test")
            np.testing.assert_array_almost_equal(arr, loaded)
        finally:
            result.close()

    def test_from_feather_legacy_alias_return_npk_obj_true(self, tmp_path):
        """Test the legacy from_feather alias with return_npk_obj=True."""
        import pyarrow as pa
        import pyarrow.feather as feather
        from numpack.io import from_feather

        arr = np.random.rand(30, 4).astype(np.float64)
        feather_path = tmp_path / "test.feather"
        npk_path = tmp_path / "test.npk"

        table = pa.table({f'col{i}': arr[:, i] for i in range(arr.shape[1])})
        feather.write_feather(table, feather_path)

        result = from_feather(feather_path, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            loaded = result.load("test")
            np.testing.assert_array_almost_equal(arr, loaded)
        finally:
            result.close()

    def test_from_arrow_return_npk_obj_true(self, tmp_path):
        """Test that from_arrow returns a valid NumPack object when return_npk_obj=True."""
        import pyarrow as pa
        from numpack.io import from_arrow

        arr = np.random.rand(50, 5).astype(np.float64)
        npk_path = tmp_path / "test.npk"

        table = pa.table({f'col{i}': arr[:, i] for i in range(arr.shape[1])})

        result = from_arrow(table, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            loaded = result.load("data")
            np.testing.assert_array_almost_equal(arr, loaded)
        finally:
            result.close()


class TestReturnNpkObjPytorch:
    """Tests for return_npk_obj in PyTorch conversion functions."""

    @pytest.fixture(autouse=True)
    def check_torch(self):
        """Check whether torch is installed."""
        pytest.importorskip("torch")

    def test_from_torch_return_none_by_default(self, tmp_path):
        """Test that from_torch returns None by default."""
        import torch
        from numpack.io import from_torch

        tensor = torch.rand(50, 10, dtype=torch.float32)
        npk_path = tmp_path / "test.npk"

        result = from_torch(tensor, npk_path, drop_if_exists=True)
        assert result is None

    def test_from_torch_return_npk_obj_true(self, tmp_path):
        """Test that from_torch returns a valid NumPack object when return_npk_obj=True."""
        import torch
        from numpack.io import from_torch

        tensor = torch.rand(50, 10, dtype=torch.float32)
        npk_path = tmp_path / "test.npk"

        result = from_torch(tensor, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            loaded = result.load("data")
            np.testing.assert_array_almost_equal(tensor.numpy(), loaded)
        finally:
            result.close()

    def test_from_torch_file_return_none_by_default(self, tmp_path):
        """Test that from_torch_file returns None by default."""
        import torch
        from numpack.io import from_torch_file

        tensor = torch.rand(50, 10, dtype=torch.float32)
        pt_path = tmp_path / "test.pt"
        npk_path = tmp_path / "test.npk"

        torch.save(tensor, pt_path)
        result = from_torch_file(pt_path, npk_path, drop_if_exists=True)
        assert result is None

    def test_from_torch_file_return_npk_obj_true(self, tmp_path):
        """Test that from_torch_file returns a valid NumPack object when return_npk_obj=True."""
        import torch
        from numpack.io import from_torch_file

        tensor = torch.rand(50, 10, dtype=torch.float32)
        pt_path = tmp_path / "test.pt"
        npk_path = tmp_path / "test.npk"

        torch.save(tensor, pt_path)
        result = from_torch_file(pt_path, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            loaded = result.load("test")
            np.testing.assert_array_almost_equal(tensor.numpy(), loaded)
        finally:
            result.close()

    def test_from_pytorch_legacy_alias_return_npk_obj_true(self, tmp_path):
        """Test the legacy from_pytorch alias with return_npk_obj=True."""
        import torch
        from numpack.io import from_pytorch

        tensor = torch.rand(30, 8, dtype=torch.float32)
        pt_path = tmp_path / "test.pt"
        npk_path = tmp_path / "test.npk"

        torch.save(tensor, pt_path)
        result = from_pytorch(pt_path, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            loaded = result.load("test")
            np.testing.assert_array_almost_equal(tensor.numpy(), loaded)
        finally:
            result.close()

    def test_from_torch_file_dict_return_npk_obj_true(self, tmp_path):
        """Test from_torch_file with dict input and return_npk_obj=True."""
        import torch
        from numpack.io import from_torch_file

        tensors = {
            'features': torch.rand(50, 10, dtype=torch.float32),
            'labels': torch.randint(0, 10, (50,)),
        }
        pt_path = tmp_path / "test.pt"
        npk_path = tmp_path / "test.npk"

        torch.save(tensors, pt_path)
        result = from_torch_file(pt_path, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            members = result.get_member_list()
            assert "features" in members
            assert "labels" in members
            np.testing.assert_array_almost_equal(
                tensors['features'].numpy(), result.load("features")
            )
        finally:
            result.close()


class TestReturnNpkObjSafetensors:
    """Tests for return_npk_obj in SafeTensors conversion functions."""

    @pytest.fixture(autouse=True)
    def check_safetensors(self):
        """Check whether safetensors is installed."""
        pytest.importorskip("safetensors")

    def test_from_safetensors_return_none_by_default(self, tmp_path):
        """Test that from_safetensors returns None by default."""
        from numpack.io import from_safetensors

        tensors = {
            'weights': np.random.rand(50, 10).astype(np.float32),
            'bias': np.random.rand(10).astype(np.float32),
        }
        npk_path = tmp_path / "test.npk"

        result = from_safetensors(tensors, npk_path, drop_if_exists=True)
        assert result is None

    def test_from_safetensors_return_npk_obj_true(self, tmp_path):
        """Test that from_safetensors returns a valid NumPack object when return_npk_obj=True."""
        from numpack.io import from_safetensors

        tensors = {
            'weights': np.random.rand(50, 10).astype(np.float32),
            'bias': np.random.rand(10).astype(np.float32),
        }
        npk_path = tmp_path / "test.npk"

        result = from_safetensors(tensors, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            members = result.get_member_list()
            assert "weights" in members
            assert "bias" in members
            np.testing.assert_array_almost_equal(tensors['weights'], result.load("weights"))
            np.testing.assert_array_almost_equal(tensors['bias'], result.load("bias"))
        finally:
            result.close()

    def test_from_safetensors_file_return_none_by_default(self, tmp_path):
        """Test that from_safetensors_file returns None by default."""
        from safetensors.numpy import save_file
        from numpack.io import from_safetensors_file

        tensors = {
            'weights': np.random.rand(50, 10).astype(np.float32),
            'bias': np.random.rand(10).astype(np.float32),
        }
        st_path = tmp_path / "test.safetensors"
        npk_path = tmp_path / "test.npk"

        save_file(tensors, str(st_path))
        result = from_safetensors_file(st_path, npk_path, drop_if_exists=True)
        assert result is None

    def test_from_safetensors_file_return_npk_obj_true(self, tmp_path):
        """Test that from_safetensors_file returns a valid NumPack object when return_npk_obj=True."""
        from safetensors.numpy import save_file
        from numpack.io import from_safetensors_file

        tensors = {
            'weights': np.random.rand(50, 10).astype(np.float32),
            'bias': np.random.rand(10).astype(np.float32),
        }
        st_path = tmp_path / "test.safetensors"
        npk_path = tmp_path / "test.npk"

        save_file(tensors, str(st_path))
        result = from_safetensors_file(st_path, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            members = result.get_member_list()
            assert "weights" in members
            assert "bias" in members
            np.testing.assert_array_almost_equal(tensors['weights'], result.load("weights"))
            np.testing.assert_array_almost_equal(tensors['bias'], result.load("bias"))
        finally:
            result.close()


class TestReturnNpkObjDataIntegrity:
    """Tests to verify data integrity when using return_npk_obj."""

    @pytest.mark.parametrize("dtype", [
        np.float32, np.float64,
        np.int32, np.int64,
        np.uint8, np.uint16, np.uint32,
    ])
    def test_from_numpy_various_dtypes(self, tmp_path, dtype):
        """Test return_npk_obj with various dtypes."""
        from numpack.io import from_numpy

        if np.issubdtype(dtype, np.integer):
            arr = np.random.randint(0, 100, (50, 10)).astype(dtype)
        else:
            arr = np.random.rand(50, 10).astype(dtype)

        npy_path = tmp_path / "test.npy"
        npk_path = tmp_path / "test.npk"

        np.save(npy_path, arr)
        result = from_numpy(npy_path, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            loaded = result.load("test")
            assert loaded.dtype == arr.dtype
            np.testing.assert_array_equal(arr, loaded)
        finally:
            result.close()

    def test_return_npk_obj_multiple_operations(self, tmp_path):
        """Test that the returned NumPack object supports multiple operations."""
        from numpack.io import from_numpy

        arr1 = np.random.rand(50, 10).astype(np.float64)
        arr2 = np.random.rand(30, 10).astype(np.float64)  # Same column count as arr1

        npy_path = tmp_path / "test.npy"
        npk_path = tmp_path / "test.npk"

        np.save(npy_path, arr1)
        result = from_numpy(npy_path, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            # Test various operations on the returned NumPack object
            assert isinstance(result, NumPack)

            # 1. Get member list
            members = result.get_member_list()
            assert "test" in members

            # 2. Get shape
            shape = result.get_shape("test")
            assert shape == (50, 10)

            # 3. Load array
            loaded = result.load("test")
            np.testing.assert_array_almost_equal(arr1, loaded)

            # 4. Append data (same column count)
            result.append({"test": arr2[:10, :]})

            # 5. Verify updated shape
            new_shape = result.get_shape("test")
            assert new_shape == (60, 10)
        finally:
            result.close()


class TestReturnNpkObjEdgeCases:
    """Edge case tests for return_npk_obj parameter."""

    def test_empty_array(self, tmp_path):
        """Test return_npk_obj with an empty array."""
        from numpack.io import from_numpy

        arr = np.array([]).reshape(0, 5)
        npy_path = tmp_path / "test.npy"
        npk_path = tmp_path / "test.npk"

        np.save(npy_path, arr)
        result = from_numpy(npy_path, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            loaded = result.load("test")
            assert loaded.shape == (0, 5)
        finally:
            result.close()

    def test_single_element_array(self, tmp_path):
        """Test return_npk_obj with a single-element array."""
        from numpack.io import from_numpy

        arr = np.array([[42.0]])
        npy_path = tmp_path / "test.npy"
        npk_path = tmp_path / "test.npk"

        np.save(npy_path, arr)
        result = from_numpy(npy_path, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            loaded = result.load("test")
            np.testing.assert_array_equal(arr, loaded)
        finally:
            result.close()

    def test_1d_array(self, tmp_path):
        """Test return_npk_obj with a 1D array."""
        from numpack.io import from_numpy

        arr = np.random.rand(100).astype(np.float64)
        npy_path = tmp_path / "test.npy"
        npk_path = tmp_path / "test.npk"

        np.save(npy_path, arr)
        result = from_numpy(npy_path, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            loaded = result.load("test")
            np.testing.assert_array_almost_equal(arr, loaded)
        finally:
            result.close()

    def test_3d_array(self, tmp_path):
        """Test return_npk_obj with a 3D array."""
        from numpack.io import from_numpy

        arr = np.random.rand(10, 20, 30).astype(np.float32)
        npy_path = tmp_path / "test.npy"
        npk_path = tmp_path / "test.npk"

        np.save(npy_path, arr)
        result = from_numpy(npy_path, npk_path, drop_if_exists=True, return_npk_obj=True)

        try:
            assert isinstance(result, NumPack)
            loaded = result.load("test")
            np.testing.assert_array_almost_equal(arr, loaded)
        finally:
            result.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

