"""
Tests for NumPack pack/unpack functionality.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
from pathlib import Path

from numpack import NumPack, pack, unpack, get_package_info


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    # Cleanup
    if os.path.exists(temp):
        shutil.rmtree(temp)


@pytest.fixture
def sample_numpack(temp_dir):
    """Create a sample NumPack with test data."""
    npk_path = temp_dir / "sample.npk"
    
    with NumPack(npk_path) as npk:
        # Save various array types
        npk.save({
            'float_array': np.random.randn(100, 10).astype(np.float32),
            'int_array': np.arange(1000).reshape(100, 10).astype(np.int64),
            'bool_array': np.random.rand(50, 20) > 0.5,
        })
    
    return npk_path


class TestPack:
    """Tests for pack function."""
    
    def test_pack_basic(self, sample_numpack, temp_dir):
        """Test basic pack operation."""
        npkg_path = pack(sample_numpack)
        
        assert npkg_path.exists()
        assert npkg_path.suffix == '.npkg'
        assert npkg_path.stat().st_size > 0
    
    def test_pack_custom_target(self, sample_numpack, temp_dir):
        """Test pack with custom target path."""
        custom_target = temp_dir / "backup" / "custom.npkg"
        npkg_path = pack(sample_numpack, target=custom_target)
        
        assert npkg_path == custom_target
        assert npkg_path.exists()
    
    def test_pack_no_compression(self, sample_numpack, temp_dir):
        """Test pack without compression."""
        compressed = pack(sample_numpack, target=temp_dir / "compressed.npkg", compression=True)
        uncompressed = pack(sample_numpack, target=temp_dir / "uncompressed.npkg", compression=False)
        
        # Compressed should generally be smaller
        assert compressed.stat().st_size <= uncompressed.stat().st_size
    
    def test_pack_overwrite(self, sample_numpack, temp_dir):
        """Test pack overwrite behavior."""
        target = temp_dir / "test.npkg"
        
        # First pack
        pack(sample_numpack, target=target)
        
        # Should raise without overwrite
        with pytest.raises(FileExistsError):
            pack(sample_numpack, target=target)
        
        # Should succeed with overwrite
        pack(sample_numpack, target=target, overwrite=True)
    
    def test_pack_invalid_source(self, temp_dir):
        """Test pack with invalid source."""
        with pytest.raises(ValueError):
            pack(temp_dir / "nonexistent")
        
        # Create a file (not a directory)
        file_path = temp_dir / "not_a_dir.txt"
        file_path.write_text("test")
        with pytest.raises(ValueError):
            pack(file_path)
    
    def test_pack_invalid_numpack(self, temp_dir):
        """Test pack with non-NumPack directory."""
        # Create empty directory
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        with pytest.raises(ValueError, match="metadata.npkm"):
            pack(empty_dir)


class TestUnpack:
    """Tests for unpack function."""
    
    def test_unpack_basic(self, sample_numpack, temp_dir):
        """Test basic unpack operation."""
        # Pack first
        npkg_path = pack(sample_numpack)
        
        # Remove original
        shutil.rmtree(sample_numpack)
        
        # Unpack
        restored_path = unpack(npkg_path)
        
        assert restored_path.exists()
        assert (restored_path / "metadata.npkm").exists()
    
    def test_unpack_data_integrity(self, sample_numpack, temp_dir):
        """Test that unpacked data matches original."""
        # Load original data
        with NumPack(sample_numpack) as npk:
            original_float = npk.load('float_array')
            original_int = npk.load('int_array')
            original_bool = npk.load('bool_array')
        
        # Pack and unpack
        npkg_path = pack(sample_numpack)
        restored_path = temp_dir / "restored.npk"
        unpack(npkg_path, target=restored_path)
        
        # Verify data
        with NumPack(restored_path) as npk:
            np.testing.assert_array_equal(npk.load('float_array'), original_float)
            np.testing.assert_array_equal(npk.load('int_array'), original_int)
            np.testing.assert_array_equal(npk.load('bool_array'), original_bool)
    
    def test_unpack_custom_target(self, sample_numpack, temp_dir):
        """Test unpack with custom target path."""
        npkg_path = pack(sample_numpack)
        custom_target = temp_dir / "custom_restore"
        
        restored_path = unpack(npkg_path, target=custom_target)
        
        assert restored_path == custom_target
        assert restored_path.exists()
    
    def test_unpack_overwrite(self, sample_numpack, temp_dir):
        """Test unpack overwrite behavior."""
        npkg_path = pack(sample_numpack)
        target = temp_dir / "restored.npk"
        
        # First unpack
        unpack(npkg_path, target=target)
        
        # Should raise without overwrite
        with pytest.raises(FileExistsError):
            unpack(npkg_path, target=target)
        
        # Should succeed with overwrite
        unpack(npkg_path, target=target, overwrite=True)
    
    def test_unpack_invalid_source(self, temp_dir):
        """Test unpack with invalid source."""
        with pytest.raises(ValueError):
            unpack(temp_dir / "nonexistent.npkg")
        
        # Create invalid file
        invalid = temp_dir / "invalid.npkg"
        invalid.write_bytes(b"not a valid npkg file")
        with pytest.raises(ValueError, match="Invalid"):
            unpack(invalid)


class TestGetPackageInfo:
    """Tests for get_package_info function."""
    
    def test_get_info_basic(self, sample_numpack, temp_dir):
        """Test basic package info retrieval."""
        npkg_path = pack(sample_numpack)
        info = get_package_info(npkg_path)
        
        assert info['version'] >= 1  # Accept v1 or v2
        assert info['file_count'] >= 1  # At least metadata
        assert 'files' in info
        assert info['total_original_size'] > 0
    
    def test_get_info_file_details(self, sample_numpack, temp_dir):
        """Test file details in package info."""
        npkg_path = pack(sample_numpack)
        info = get_package_info(npkg_path)
        
        file_names = [f['name'] for f in info['files']]
        assert 'metadata.npkm' in file_names
        
        # Should have data files for each array
        data_files = [f for f in file_names if f.startswith('data_')]
        assert len(data_files) == 3  # float_array, int_array, bool_array
    
    def test_get_info_compression_ratio(self, sample_numpack, temp_dir):
        """Test compression ratio in package info."""
        # With compression
        compressed_pkg = pack(sample_numpack, target=temp_dir / "compressed.npkg", compression=True)
        compressed_info = get_package_info(compressed_pkg)
        
        # Without compression
        uncompressed_pkg = pack(sample_numpack, target=temp_dir / "uncompressed.npkg", compression=False)
        uncompressed_info = get_package_info(uncompressed_pkg)
        
        # Compressed should have ratio <= 1
        assert compressed_info['compression_ratio'] <= 1.0
        # Uncompressed should have ratio == 1
        assert uncompressed_info['compression_ratio'] == 1.0


class TestPackUnpackWithDeletions:
    """Tests for pack/unpack with deletion bitmaps."""
    
    def test_pack_with_deletions(self, temp_dir):
        """Test that deletion bitmaps are preserved."""
        npk_path = temp_dir / "with_deletions.npk"
        
        # Create NumPack with deletions
        with NumPack(npk_path) as npk:
            npk.save({'data': np.arange(100).reshape(10, 10)})
            npk.drop('data', indexes=[0, 5, 9])  # Delete some rows
        
        # Verify deletion bitmap exists
        deletion_files = list(npk_path.glob("deleted_*.npkb"))
        assert len(deletion_files) > 0
        
        # Pack and unpack
        npkg_path = pack(npk_path)
        restored_path = temp_dir / "restored.npk"
        unpack(npkg_path, target=restored_path)
        
        # Verify deletion bitmap is preserved
        restored_deletion_files = list(restored_path.glob("deleted_*.npkb"))
        assert len(restored_deletion_files) == len(deletion_files)
        
        # Verify data shape (should reflect deletions)
        with NumPack(restored_path) as npk:
            shape = npk.get_shape('data')
            assert shape[0] == 7  # 10 - 3 deleted rows


class TestRoundTrip:
    """End-to-end round-trip tests."""
    
    def test_multiple_pack_unpack_cycles(self, temp_dir):
        """Test multiple pack/unpack cycles maintain integrity."""
        npk_path = temp_dir / "original.npk"
        original_data = np.random.randn(50, 20).astype(np.float64)
        
        # Create original
        with NumPack(npk_path) as npk:
            npk.save({'data': original_data})
        
        # Multiple cycles
        current_path = npk_path
        for i in range(3):
            npkg_path = temp_dir / f"cycle_{i}.npkg"
            pack(current_path, target=npkg_path)
            
            next_path = temp_dir / f"restored_{i}.npk"
            unpack(npkg_path, target=next_path)
            current_path = next_path
        
        # Verify final data
        with NumPack(current_path) as npk:
            final_data = npk.load('data')
            np.testing.assert_array_equal(final_data, original_data)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
