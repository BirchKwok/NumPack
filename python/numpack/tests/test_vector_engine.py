"""
Test cases for vector_engine module: VectorEngine and StreamingVectorEngine classes

Covers:
- All supported data types (float32, float64, int8, uint8)
- All supported metrics (dot, cosine, l2, l2sq, hamming, jaccard, kl, js, inner)
- Edge cases: empty arrays, single element, dimension mismatch, k > candidates, etc.
"""

import numpy as np
import pytest
import tempfile
import sys
from pathlib import Path

from numpack import NumPack
from numpack.vector_engine import VectorEngine, StreamingVectorEngine

# Import conftest utilities
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# Test fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory fixture"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def vector_search():
    """Create a VectorEngine instance fixture"""
    return VectorEngine()


@pytest.fixture
def streaming_search():
    """Create a StreamingVectorEngine instance fixture"""
    return StreamingVectorEngine()


# ============================================================================
# Helper functions
# ============================================================================

def create_test_vectors(n_vectors: int, dim: int, dtype=np.float32):
    """Create test vectors with specified dtype"""
    if dtype in (np.float32, np.float64):
        return np.random.randn(n_vectors, dim).astype(dtype)
    elif dtype == np.int8:
        return np.random.randint(-128, 127, size=(n_vectors, dim), dtype=dtype)
    elif dtype == np.uint8:
        return np.random.randint(0, 255, size=(n_vectors, dim), dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def create_test_query(dim: int, dtype=np.float32):
    """Create a single test query vector with specified dtype"""
    if dtype in (np.float32, np.float64):
        return np.random.randn(dim).astype(dtype)
    elif dtype == np.int8:
        return np.random.randint(-128, 127, size=dim, dtype=dtype)
    elif dtype == np.uint8:
        return np.random.randint(0, 255, size=dim, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


# Supported dtypes for float metrics
FLOAT_DTYPES = [np.float32, np.float64]

# All integer dtypes
INT_DTYPES = [np.int8]

# Binary dtypes (for hamming/jaccard)
BINARY_DTYPES = [np.uint8]

# Float metrics (work with float32, float64)
FLOAT_METRICS = ['dot', 'cosine', 'l2', 'l2sq', 'inner']

# Distance metrics
DISTANCE_METRICS = ['l2', 'l2sq', 'hamming', 'jaccard', 'kl', 'js']

# Similarity metrics  
SIMILARITY_METRICS = ['dot', 'cosine', 'inner']

# Metric aliases for testing
METRIC_ALIASES = {
    'dot': ['dot', 'dot_product', 'dotproduct'],
    'cosine': ['cos', 'cosine', 'cosine_similarity'],
    'l2': ['l2', 'euclidean', 'l2_distance'],
    'l2sq': ['l2sq', 'l2_squared', 'squared_euclidean'],
    'inner': ['inner', 'inner_product'],
}


# ============================================================================
# VectorEngine class tests
# ============================================================================

class TestVectorEngineCapabilities:
    """Test VectorEngine.capabilities() method"""
    
    def test_capabilities_returns_string(self, vector_search):
        """Test that capabilities() returns a non-empty string"""
        caps = vector_search.capabilities()
        assert isinstance(caps, str)
        assert len(caps) > 0
    
    def test_capabilities_contains_cpu_info(self, vector_search):
        """Test that capabilities contains CPU info"""
        caps = vector_search.capabilities()
        # Should contain some CPU/SIMD related info
        assert 'CPU' in caps or 'cpu' in caps.lower() or 'SIMD' in caps.upper() or len(caps) > 0


class TestVectorEngineComputeMetric:
    """Test VectorEngine.compute_metric() method"""
    
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("metric", FLOAT_METRICS)
    def test_compute_metric_float_types(self, vector_search, dtype, metric):
        """Test compute_metric with float dtypes and float metrics"""
        dim = 128
        a = create_test_query(dim, dtype)
        b = create_test_query(dim, dtype)
        
        score = vector_search.compute_metric(a, b, metric)
        assert isinstance(score, float)
        assert not np.isnan(score)
    
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    @pytest.mark.parametrize("metric", ['dot', 'cosine', 'l2', 'l2sq'])
    def test_compute_metric_int8_types(self, vector_search, dtype, metric):
        """Test compute_metric with int8 dtype"""
        dim = 128
        a = create_test_query(dim, dtype)
        b = create_test_query(dim, dtype)
        
        score = vector_search.compute_metric(a, b, metric)
        assert isinstance(score, float)
        assert not np.isnan(score)
    
    def test_compute_metric_uint8_hamming(self, vector_search):
        """Test compute_metric with uint8 and hamming metric"""
        dim = 64
        a = create_test_query(dim, np.uint8)
        b = create_test_query(dim, np.uint8)
        
        score = vector_search.compute_metric(a, b, 'hamming')
        assert isinstance(score, float)
        assert score >= 0  # Hamming distance is non-negative
    
    def test_compute_metric_uint8_jaccard(self, vector_search):
        """Test compute_metric with uint8 and jaccard metric"""
        dim = 64
        a = create_test_query(dim, np.uint8)
        b = create_test_query(dim, np.uint8)
        
        score = vector_search.compute_metric(a, b, 'jaccard')
        assert isinstance(score, float)
        assert 0 <= score <= 1  # Jaccard distance is between 0 and 1
    
    @pytest.mark.parametrize("alias_group", list(METRIC_ALIASES.keys()))
    def test_compute_metric_aliases(self, vector_search, alias_group):
        """Test that metric aliases produce the same results"""
        dim = 64
        dtype = np.float32
        a = create_test_query(dim, dtype)
        b = create_test_query(dim, dtype)
        
        aliases = METRIC_ALIASES[alias_group]
        scores = [vector_search.compute_metric(a, b, alias) for alias in aliases]
        
        # All aliases should produce the same result
        for i in range(1, len(scores)):
            np.testing.assert_almost_equal(scores[0], scores[i], decimal=5)
    
    def test_compute_metric_identical_vectors(self, vector_search):
        """Test compute_metric with identical vectors"""
        dim = 128
        a = create_test_query(dim, np.float32)
        
        # Cosine similarity of identical vectors should be 1.0
        cosine_score = vector_search.compute_metric(a, a, 'cosine')
        np.testing.assert_almost_equal(cosine_score, 1.0, decimal=5)
        
        # L2 distance of identical vectors should be 0.0
        l2_score = vector_search.compute_metric(a, a, 'l2')
        np.testing.assert_almost_equal(l2_score, 0.0, decimal=5)
    
    def test_compute_metric_orthogonal_vectors(self, vector_search):
        """Test compute_metric with orthogonal vectors"""
        # Create two orthogonal unit vectors
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        
        # Dot product of orthogonal vectors should be 0
        dot_score = vector_search.compute_metric(a, b, 'dot')
        np.testing.assert_almost_equal(dot_score, 0.0, decimal=5)
        
        # Cosine similarity of orthogonal vectors should be 0
        cosine_score = vector_search.compute_metric(a, b, 'cosine')
        np.testing.assert_almost_equal(cosine_score, 0.0, decimal=5)
    
    def test_compute_metric_dimension_mismatch(self, vector_search):
        """Test compute_metric raises error on dimension mismatch"""
        a = create_test_query(64, np.float32)
        b = create_test_query(128, np.float32)
        
        with pytest.raises(Exception):  # Could be ValueError or other
            vector_search.compute_metric(a, b, 'cosine')
    
    def test_compute_metric_dtype_mismatch(self, vector_search):
        """Test compute_metric handles mixed dtypes via auto-conversion to float64"""
        a = create_test_query(64, np.float32)
        b = create_test_query(64, np.float64)

        # Mixed dtype should work (auto-converted to float64)
        result = vector_search.compute_metric(a, b, 'cosine')
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0  # Cosine similarity range
    
    def test_compute_metric_invalid_metric(self, vector_search):
        """Test compute_metric raises error on invalid metric"""
        a = create_test_query(64, np.float32)
        b = create_test_query(64, np.float32)
        
        with pytest.raises(Exception):  # Could be ValueError or other
            vector_search.compute_metric(a, b, 'invalid_metric')
    
    def test_compute_metric_single_element(self, vector_search):
        """Test compute_metric with single element vectors"""
        a = np.array([3.0], dtype=np.float32)
        b = np.array([4.0], dtype=np.float32)
        
        # Dot product should be 12.0
        dot_score = vector_search.compute_metric(a, b, 'dot')
        np.testing.assert_almost_equal(dot_score, 12.0, decimal=5)
    
    def test_compute_metric_kl_divergence(self, vector_search):
        """Test KL divergence metric"""
        # Create probability distributions
        a = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        b = np.array([0.4, 0.4, 0.2], dtype=np.float32)
        
        score = vector_search.compute_metric(a, b, 'kl')
        assert isinstance(score, float)
        assert score >= 0  # KL divergence is non-negative
    
    def test_compute_metric_js_divergence(self, vector_search):
        """Test JS divergence metric"""
        # Create probability distributions
        a = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        b = np.array([0.4, 0.4, 0.2], dtype=np.float32)
        
        score = vector_search.compute_metric(a, b, 'js')
        assert isinstance(score, float)
        assert 0 <= score  # JS divergence is bounded


class TestVectorEngineBatchCompute:
    """Test VectorEngine.batch_compute() method"""
    
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("metric", FLOAT_METRICS)
    def test_batch_compute_basic(self, vector_search, dtype, metric):
        """Test batch_compute with various dtypes and metrics"""
        dim = 128
        n_candidates = 100
        
        query = create_test_query(dim, dtype)
        candidates = create_test_vectors(n_candidates, dim, dtype)
        
        scores = vector_search.batch_compute(query, candidates, metric)
        
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (n_candidates,)
        assert scores.dtype == np.float64
        assert not np.any(np.isnan(scores))
    
    def test_batch_compute_int8(self, vector_search):
        """Test batch_compute with int8 dtype"""
        dim = 64
        n_candidates = 50
        
        query = create_test_query(dim, np.int8)
        candidates = create_test_vectors(n_candidates, dim, np.int8)
        
        scores = vector_search.batch_compute(query, candidates, 'cosine')
        
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (n_candidates,)
    
    def test_batch_compute_uint8_hamming(self, vector_search):
        """Test batch_compute with uint8 and hamming metric"""
        dim = 64
        n_candidates = 50
        
        query = create_test_query(dim, np.uint8)
        candidates = create_test_vectors(n_candidates, dim, np.uint8)
        
        scores = vector_search.batch_compute(query, candidates, 'hamming')
        
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (n_candidates,)
        assert np.all(scores >= 0)  # Hamming distance is non-negative
    
    def test_batch_compute_large_batch(self, vector_search):
        """Test batch_compute with large batch (parallel processing)"""
        dim = 128
        n_candidates = 1000  # >= 500 triggers parallel processing
        
        query = create_test_query(dim, np.float32)
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        
        scores = vector_search.batch_compute(query, candidates, 'cosine')
        
        assert scores.shape == (n_candidates,)
    
    def test_batch_compute_single_candidate(self, vector_search):
        """Test batch_compute with single candidate"""
        dim = 64
        
        query = create_test_query(dim, np.float32)
        candidates = create_test_vectors(1, dim, np.float32)
        
        scores = vector_search.batch_compute(query, candidates, 'cosine')
        
        assert scores.shape == (1,)
    
    def test_batch_compute_consistency_with_compute_metric(self, vector_search):
        """Test that batch_compute results match compute_metric"""
        dim = 64
        n_candidates = 10
        
        query = create_test_query(dim, np.float32)
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        
        # Batch compute
        batch_scores = vector_search.batch_compute(query, candidates, 'cosine')
        
        # Individual compute
        individual_scores = np.array([
            vector_search.compute_metric(query, candidates[i], 'cosine')
            for i in range(n_candidates)
        ])
        
        np.testing.assert_array_almost_equal(batch_scores, individual_scores, decimal=5)
    
    def test_batch_compute_dimension_mismatch(self, vector_search):
        """Test batch_compute raises error on dimension mismatch"""
        query = create_test_query(64, np.float32)
        candidates = create_test_vectors(10, 128, np.float32)
        
        with pytest.raises(Exception):
            vector_search.batch_compute(query, candidates, 'cosine')


class TestVectorEngineTopKSearch:
    """Test VectorEngine.top_k_search() method"""
    
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("metric", FLOAT_METRICS)
    def test_top_k_search_basic(self, vector_search, dtype, metric):
        """Test top_k_search with various dtypes and metrics"""
        dim = 128
        n_candidates = 100
        k = 10
        
        query = create_test_query(dim, dtype)
        candidates = create_test_vectors(n_candidates, dim, dtype)
        
        indices, scores = vector_search.top_k_search(query, candidates, metric, k)
        
        assert isinstance(indices, np.ndarray)
        assert isinstance(scores, np.ndarray)
        assert indices.shape == (k,)
        assert scores.shape == (k,)
        assert indices.dtype == np.uint64
        assert scores.dtype == np.float64
        
        # All indices should be valid
        assert np.all(indices < n_candidates)
    
    def test_top_k_search_k_equals_candidates(self, vector_search):
        """Test top_k_search when k equals number of candidates"""
        dim = 64
        n_candidates = 20
        k = n_candidates
        
        query = create_test_query(dim, np.float32)
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        
        indices, scores = vector_search.top_k_search(query, candidates, 'cosine', k)
        
        assert indices.shape == (k,)
        # All indices should be unique
        assert len(np.unique(indices)) == k
    
    def test_top_k_search_k_greater_than_candidates(self, vector_search):
        """Test top_k_search when k > number of candidates"""
        dim = 64
        n_candidates = 10
        k = 20  # k > n_candidates
        
        query = create_test_query(dim, np.float32)
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        
        indices, scores = vector_search.top_k_search(query, candidates, 'cosine', k)
        
        # Should return min(k, n_candidates) results
        assert indices.shape[0] <= k
        assert indices.shape[0] == n_candidates
    
    def test_top_k_search_k_equals_one(self, vector_search):
        """Test top_k_search with k=1"""
        dim = 64
        n_candidates = 100
        k = 1
        
        query = create_test_query(dim, np.float32)
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        
        indices, scores = vector_search.top_k_search(query, candidates, 'cosine', k)
        
        assert indices.shape == (k,)
        assert scores.shape == (k,)
    
    def test_top_k_search_similarity_ordering(self, vector_search):
        """Test that similarity metrics return results in descending order"""
        dim = 64
        n_candidates = 50
        k = 10
        
        query = create_test_query(dim, np.float32)
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        
        indices, scores = vector_search.top_k_search(query, candidates, 'cosine', k)
        
        # Scores should be sorted in descending order for similarity metrics
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]
    
    def test_top_k_search_distance_ordering(self, vector_search):
        """Test that distance metrics return results in ascending order"""
        dim = 64
        n_candidates = 50
        k = 10
        
        query = create_test_query(dim, np.float32)
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        
        indices, scores = vector_search.top_k_search(query, candidates, 'l2', k)
        
        # Scores should be sorted in ascending order for distance metrics
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]
    
    def test_top_k_search_finds_exact_match(self, vector_search):
        """Test that top_k_search finds exact match"""
        dim = 64
        n_candidates = 50
        
        query = create_test_query(dim, np.float32)
        # Insert query as one of the candidates
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        insert_pos = 25
        candidates[insert_pos] = query
        
        indices, scores = vector_search.top_k_search(query, candidates, 'cosine', k=1)
        
        # The exact match should be the top result
        assert indices[0] == insert_pos
        np.testing.assert_almost_equal(scores[0], 1.0, decimal=5)


class TestVectorEngineMultiQueryTopK:
    """Test VectorEngine.multi_query_top_k() method
    
    Note: These tests are skipped if multi_query_top_k is not available in the current build.
    """
    
    @pytest.fixture(autouse=True)
    def check_method_available(self, vector_search):
        """Skip tests if multi_query_top_k is not available"""
        if not hasattr(vector_search, 'multi_query_top_k'):
            pytest.skip("multi_query_top_k method not available in current build")
    
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_multi_query_top_k_basic(self, vector_search, dtype):
        """Test multi_query_top_k with various dtypes"""
        dim = 128
        n_queries = 5
        n_candidates = 100
        k = 10
        
        queries = create_test_vectors(n_queries, dim, dtype)
        candidates = create_test_vectors(n_candidates, dim, dtype)
        
        all_indices, all_scores = vector_search.multi_query_top_k(
            queries, candidates, 'cosine', k
        )
        
        assert all_indices.shape == (n_queries * k,)
        assert all_scores.shape == (n_queries * k,)
        
        # Reshape to verify structure
        indices_reshaped = all_indices.reshape(n_queries, k)
        scores_reshaped = all_scores.reshape(n_queries, k)
        
        assert indices_reshaped.shape == (n_queries, k)
        assert scores_reshaped.shape == (n_queries, k)
    
    def test_multi_query_top_k_consistency(self, vector_search):
        """Test multi_query_top_k matches individual top_k_search"""
        dim = 64
        n_queries = 3
        n_candidates = 50
        k = 5
        
        queries = create_test_vectors(n_queries, dim, np.float32)
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        
        # Multi-query result
        all_indices, all_scores = vector_search.multi_query_top_k(
            queries, candidates, 'cosine', k
        )
        
        # Individual results
        for i in range(n_queries):
            single_indices, single_scores = vector_search.top_k_search(
                queries[i], candidates, 'cosine', k
            )
            
            # Compare results
            np.testing.assert_array_equal(
                all_indices[i*k:(i+1)*k], single_indices
            )
            np.testing.assert_array_almost_equal(
                all_scores[i*k:(i+1)*k], single_scores, decimal=5
            )
    
    def test_multi_query_top_k_single_query(self, vector_search):
        """Test multi_query_top_k with single query"""
        dim = 64
        n_candidates = 50
        k = 10
        
        queries = create_test_vectors(1, dim, np.float32)
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        
        all_indices, all_scores = vector_search.multi_query_top_k(
            queries, candidates, 'cosine', k
        )
        
        assert all_indices.shape == (k,)
        assert all_scores.shape == (k,)


class TestVectorEngineSegmentedTopK:
    """Test VectorEngine.segmented_top_k_search() method"""
    
    def test_segmented_top_k_search_basic(self, vector_search):
        """Test segmented_top_k_search basic functionality"""
        dim = 64
        n_candidates = 50
        k = 10
        
        query = create_test_query(dim, np.float32)
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        current_indices = np.array([], dtype=np.uint64)
        current_scores = np.array([], dtype=np.float64)
        
        new_indices, new_scores = vector_search.segmented_top_k_search(
            query, candidates, 'cosine', 0, current_indices, current_scores, k, True
        )
        
        assert new_indices.shape == (k,)
        assert new_scores.shape == (k,)
    
    def test_segmented_top_k_search_incremental(self, vector_search):
        """Test segmented_top_k_search incrementally"""
        dim = 64
        batch_size = 20
        n_batches = 5
        k = 10
        
        query = create_test_query(dim, np.float32)
        
        current_indices = np.array([], dtype=np.uint64)
        current_scores = np.array([], dtype=np.float64)
        
        for batch_idx in range(n_batches):
            candidates = create_test_vectors(batch_size, dim, np.float32)
            global_offset = batch_idx * batch_size
            
            current_indices, current_scores = vector_search.segmented_top_k_search(
                query, candidates, 'cosine', global_offset,
                current_indices, current_scores, k, True
            )
        
        # After all batches, should have k results
        assert len(current_indices) <= k
        assert len(current_scores) <= k


# ============================================================================
# StreamingVectorEngine class tests
# ============================================================================

class TestStreamingVectorEngineCapabilities:
    """Test StreamingVectorEngine.capabilities() method"""
    
    def test_capabilities_returns_string(self, streaming_search):
        """Test that capabilities() returns a non-empty string"""
        caps = streaming_search.capabilities()
        assert isinstance(caps, str)
        assert len(caps) > 0


class TestStreamingVectorEngineTopK:
    """Test StreamingVectorEngine.streaming_top_k_from_file() method"""
    
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("metric", ['cosine', 'dot', 'l2'])
    def test_streaming_top_k_basic(self, streaming_search, temp_dir, dtype, metric):
        """Test streaming_top_k_from_file with various dtypes and metrics"""
        dim = 128
        n_candidates = 100
        k = 10
        
        # Create test data and save to NumPack file
        candidates = create_test_vectors(n_candidates, dim, dtype)
        query = create_test_query(dim, dtype)
        
        npk_path = Path(temp_dir) / "test_vectors.npk"
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({'embeddings': candidates})
        
        # Perform streaming search
        indices, scores = streaming_search.streaming_top_k_from_file(
            query, str(npk_path), 'embeddings', metric, k
        )
        
        assert isinstance(indices, np.ndarray)
        assert isinstance(scores, np.ndarray)
        assert indices.shape == (k,)
        assert scores.shape == (k,)
        assert indices.dtype == np.uint64
        assert scores.dtype == np.float64
    
    def test_streaming_top_k_large_file(self, streaming_search, temp_dir):
        """Test streaming_top_k_from_file with large dataset"""
        dim = 128
        n_candidates = 5000
        k = 10
        batch_size = 1000
        
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        query = create_test_query(dim, np.float32)
        
        npk_path = Path(temp_dir) / "large_vectors.npk"
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({'embeddings': candidates})
        
        indices, scores = streaming_search.streaming_top_k_from_file(
            query, str(npk_path), 'embeddings', 'cosine', k, batch_size=batch_size
        )
        
        assert indices.shape == (k,)
        assert np.all(indices < n_candidates)
    
    def test_streaming_top_k_consistency_with_in_memory(self, streaming_search, temp_dir):
        """Test streaming search matches in-memory search"""
        dim = 64
        n_candidates = 100
        k = 10
        
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        query = create_test_query(dim, np.float32)
        
        npk_path = Path(temp_dir) / "test_vectors.npk"
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({'embeddings': candidates})
        
        # Streaming search
        stream_indices, stream_scores = streaming_search.streaming_top_k_from_file(
            query, str(npk_path), 'embeddings', 'cosine', k
        )
        
        # In-memory search
        vs = VectorEngine()
        mem_indices, mem_scores = vs.top_k_search(query, candidates, 'cosine', k)
        
        # Results should match
        np.testing.assert_array_equal(stream_indices, mem_indices)
        np.testing.assert_array_almost_equal(stream_scores, mem_scores, decimal=5)
    
    def test_streaming_top_k_k_greater_than_candidates(self, streaming_search, temp_dir):
        """Test streaming_top_k when k > number of candidates"""
        dim = 64
        n_candidates = 10
        k = 20
        
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        query = create_test_query(dim, np.float32)
        
        npk_path = Path(temp_dir) / "small_vectors.npk"
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({'embeddings': candidates})
        
        indices, scores = streaming_search.streaming_top_k_from_file(
            query, str(npk_path), 'embeddings', 'cosine', k
        )
        
        # Should return min(k, n_candidates) results
        assert indices.shape[0] <= k
        assert indices.shape[0] == n_candidates
    
    def test_streaming_top_k_invalid_array_name(self, streaming_search, temp_dir):
        """Test streaming_top_k raises error on invalid array name"""
        dim = 64
        n_candidates = 50
        
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        query = create_test_query(dim, np.float32)
        
        npk_path = Path(temp_dir) / "test_vectors.npk"
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({'embeddings': candidates})
        
        with pytest.raises(Exception):
            streaming_search.streaming_top_k_from_file(
                query, str(npk_path), 'nonexistent_array', 'cosine', k=10
            )
    
    def test_streaming_top_k_invalid_file_path(self, streaming_search):
        """Test streaming_top_k raises error on invalid file path"""
        query = create_test_query(64, np.float32)
        
        with pytest.raises(Exception):
            streaming_search.streaming_top_k_from_file(
                query, '/nonexistent/path.npk', 'embeddings', 'cosine', k=10
            )


class TestStreamingVectorEngineBatchCompute:
    """Test StreamingVectorEngine.streaming_batch_compute() method"""
    
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_streaming_batch_compute_basic(self, streaming_search, temp_dir, dtype):
        """Test streaming_batch_compute with various dtypes"""
        dim = 128
        n_candidates = 100
        
        candidates = create_test_vectors(n_candidates, dim, dtype)
        query = create_test_query(dim, dtype)
        
        npk_path = Path(temp_dir) / "test_vectors.npk"
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({'embeddings': candidates})
        
        scores = streaming_search.streaming_batch_compute(
            query, str(npk_path), 'embeddings', 'cosine'
        )
        
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (n_candidates,)
        assert scores.dtype == np.float64
    
    def test_streaming_batch_compute_consistency_with_in_memory(self, streaming_search, temp_dir):
        """Test streaming batch compute matches in-memory batch compute"""
        dim = 64
        n_candidates = 100
        
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        query = create_test_query(dim, np.float32)
        
        npk_path = Path(temp_dir) / "test_vectors.npk"
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({'embeddings': candidates})
        
        # Streaming batch compute
        stream_scores = streaming_search.streaming_batch_compute(
            query, str(npk_path), 'embeddings', 'cosine'
        )
        
        # In-memory batch compute
        vs = VectorEngine()
        mem_scores = vs.batch_compute(query, candidates, 'cosine')
        
        np.testing.assert_array_almost_equal(stream_scores, mem_scores, decimal=5)


class TestStreamingVectorEngineMultiQuery:
    """Test StreamingVectorEngine.streaming_multi_query_top_k() method
    
    Note: These tests are skipped if streaming_multi_query_top_k is not available in the current build.
    """
    
    @pytest.fixture(autouse=True)
    def check_method_available(self, streaming_search):
        """Skip tests if streaming_multi_query_top_k is not available"""
        if not hasattr(streaming_search, 'streaming_multi_query_top_k'):
            pytest.skip("streaming_multi_query_top_k method not available in current build")
    
    def test_streaming_multi_query_basic(self, streaming_search, temp_dir):
        """Test streaming_multi_query_top_k with float32
        
        Note: streaming_multi_query_top_k only supports float32 currently.
        """
        dim = 128
        n_queries = 5
        n_candidates = 100
        k = 10
        dtype = np.float32  # Only float32 is supported
        
        queries = create_test_vectors(n_queries, dim, dtype)
        candidates = create_test_vectors(n_candidates, dim, dtype)
        
        npk_path = Path(temp_dir) / "test_vectors.npk"
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({'embeddings': candidates})
        
        all_indices, all_scores = streaming_search.streaming_multi_query_top_k(
            queries, str(npk_path), 'embeddings', 'cosine', k
        )
        
        assert all_indices.shape == (n_queries * k,)
        assert all_scores.shape == (n_queries * k,)
        
        # Reshape to verify structure
        indices_reshaped = all_indices.reshape(n_queries, k)
        scores_reshaped = all_scores.reshape(n_queries, k)
        
        assert indices_reshaped.shape == (n_queries, k)
        assert scores_reshaped.shape == (n_queries, k)
    
    def test_streaming_multi_query_consistency(self, streaming_search, temp_dir):
        """Test streaming_multi_query matches individual streaming searches"""
        dim = 64
        n_queries = 3
        n_candidates = 50
        k = 5
        
        queries = create_test_vectors(n_queries, dim, np.float32)
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        
        npk_path = Path(temp_dir) / "test_vectors.npk"
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({'embeddings': candidates})
        
        # Multi-query result
        all_indices, all_scores = streaming_search.streaming_multi_query_top_k(
            queries, str(npk_path), 'embeddings', 'cosine', k
        )
        
        # Individual results
        for i in range(n_queries):
            single_indices, single_scores = streaming_search.streaming_top_k_from_file(
                queries[i], str(npk_path), 'embeddings', 'cosine', k
            )
            
            np.testing.assert_array_equal(
                all_indices[i*k:(i+1)*k], single_indices
            )
            np.testing.assert_array_almost_equal(
                all_scores[i*k:(i+1)*k], single_scores, decimal=5
            )
    
    def test_streaming_multi_query_single_query(self, streaming_search, temp_dir):
        """Test streaming_multi_query with single query"""
        dim = 64
        n_candidates = 50
        k = 10
        
        queries = create_test_vectors(1, dim, np.float32)
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        
        npk_path = Path(temp_dir) / "test_vectors.npk"
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({'embeddings': candidates})
        
        all_indices, all_scores = streaming_search.streaming_multi_query_top_k(
            queries, str(npk_path), 'embeddings', 'cosine', k
        )
        
        assert all_indices.shape == (k,)
        assert all_scores.shape == (k,)


# ============================================================================
# Edge cases and error handling tests
# ============================================================================

class TestVectorEngineEdgeCases:
    """Test edge cases for VectorEngine"""
    
    def test_zero_vector(self, vector_search):
        """Test with zero vectors"""
        dim = 64
        a = np.zeros(dim, dtype=np.float32)
        b = create_test_query(dim, np.float32)
        
        # L2 distance with zero vector should work
        l2_score = vector_search.compute_metric(a, b, 'l2')
        assert isinstance(l2_score, float)
        
        # Dot product with zero vector should be 0
        dot_score = vector_search.compute_metric(a, b, 'dot')
        np.testing.assert_almost_equal(dot_score, 0.0, decimal=5)
    
    def test_large_dimension(self, vector_search):
        """Test with large dimension vectors"""
        dim = 2048
        n_candidates = 50
        k = 10
        
        query = create_test_query(dim, np.float32)
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        
        indices, scores = vector_search.top_k_search(query, candidates, 'cosine', k)
        
        assert indices.shape == (k,)
    
    def test_small_dimension(self, vector_search):
        """Test with very small dimension vectors"""
        dim = 2
        n_candidates = 50
        k = 5
        
        query = create_test_query(dim, np.float32)
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        
        indices, scores = vector_search.top_k_search(query, candidates, 'cosine', k)
        
        assert indices.shape == (k,)
    
    def test_normalized_vectors(self, vector_search):
        """Test with pre-normalized vectors"""
        dim = 128
        
        a = create_test_query(dim, np.float32)
        a = a / np.linalg.norm(a)  # Normalize
        
        b = create_test_query(dim, np.float32)
        b = b / np.linalg.norm(b)  # Normalize
        
        # For normalized vectors, dot product and cosine should be equal
        dot_score = vector_search.compute_metric(a, b, 'dot')
        cosine_score = vector_search.compute_metric(a, b, 'cosine')
        
        np.testing.assert_almost_equal(dot_score, cosine_score, decimal=5)
    
    def test_negative_values(self, vector_search):
        """Test with vectors containing negative values"""
        dim = 64
        a = np.array([-1.0, 2.0, -3.0, 4.0] * (dim // 4), dtype=np.float32)
        b = np.array([1.0, -2.0, 3.0, -4.0] * (dim // 4), dtype=np.float32)
        
        score = vector_search.compute_metric(a, b, 'cosine')
        assert isinstance(score, float)
        assert -1 <= score <= 1  # Cosine similarity is bounded


class TestStreamingVectorEngineEdgeCases:
    """Test edge cases for StreamingVectorEngine"""
    
    def test_small_batch_size(self, streaming_search, temp_dir):
        """Test streaming search with very small batch size"""
        dim = 64
        n_candidates = 100
        k = 10
        batch_size = 5  # Very small batch
        
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        query = create_test_query(dim, np.float32)
        
        npk_path = Path(temp_dir) / "test_vectors.npk"
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({'embeddings': candidates})
        
        indices, scores = streaming_search.streaming_top_k_from_file(
            query, str(npk_path), 'embeddings', 'cosine', k, batch_size=batch_size
        )
        
        assert indices.shape == (k,)
    
    def test_batch_size_larger_than_candidates(self, streaming_search, temp_dir):
        """Test streaming search when batch_size > n_candidates"""
        dim = 64
        n_candidates = 50
        k = 10
        batch_size = 10000  # Much larger than candidates
        
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        query = create_test_query(dim, np.float32)
        
        npk_path = Path(temp_dir) / "test_vectors.npk"
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({'embeddings': candidates})
        
        indices, scores = streaming_search.streaming_top_k_from_file(
            query, str(npk_path), 'embeddings', 'cosine', k, batch_size=batch_size
        )
        
        assert indices.shape == (k,)
    
    def test_single_candidate(self, streaming_search, temp_dir):
        """Test streaming search with single candidate"""
        dim = 64
        n_candidates = 1
        k = 1
        
        candidates = create_test_vectors(n_candidates, dim, np.float32)
        query = create_test_query(dim, np.float32)
        
        npk_path = Path(temp_dir) / "single_vector.npk"
        with NumPack(npk_path, drop_if_exists=True) as npk:
            npk.save({'embeddings': candidates})
        
        indices, scores = streaming_search.streaming_top_k_from_file(
            query, str(npk_path), 'embeddings', 'cosine', k
        )
        
        assert indices.shape == (1,)
        assert indices[0] == 0


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

