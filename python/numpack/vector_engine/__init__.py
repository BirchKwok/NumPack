"""Vector Engine module for high-performance vector similarity computation.

This module provides two main classes:
- VectorSearch: Pure in-memory vector similarity computation with SIMD acceleration
- StreamingVectorSearch: Memory-efficient streaming vector search from NumPack files

Example:
    >>> from numpack.vector_engine import VectorSearch, StreamingVectorSearch
    >>> import numpy as np
    >>>
    >>> # In-memory search
    >>> engine = VectorSearch()
    >>> query = np.random.randn(128).astype(np.float32)
    >>> candidates = np.random.randn(10000, 128).astype(np.float32)
    >>> indices, scores = engine.top_k_search(query, candidates, 'cosine', k=10)
    >>>
    >>> # Streaming search from file (memory-efficient)
    >>> streaming = StreamingVectorSearch()
    >>> with NumPack('vectors.npk') as npk:
    ...     indices, scores = streaming.streaming_top_k_from_file(
    ...         query, str(npk._filename), 'embeddings', 'cosine', k=10
    ...     )
"""

from typing import Tuple, Union, Literal, overload
import numpy as np
from numpy.typing import NDArray

# Import from Rust backend
try:
    from numpack._lib_numpack import VectorSearch as _VectorSearch
    from numpack._lib_numpack import StreamingVectorSearch as _StreamingVectorSearch
except ImportError:
    # Fallback for type checking
    _VectorSearch = None
    _StreamingVectorSearch = None


# Type aliases
MetricType = Literal[
    'dot', 'dot_product', 'dotproduct',
    'cos', 'cosine', 'cosine_similarity',
    'l2', 'euclidean', 'l2_distance',
    'l2sq', 'l2_squared', 'squared_euclidean',
    'hamming', 'jaccard',
    'kl', 'kl_divergence',
    'js', 'js_divergence',
    'inner', 'inner_product'
]


class VectorSearch:
    """High-performance in-memory vector similarity computation.
    
    VectorSearch provides SIMD-accelerated (AVX2, AVX-512, NEON, SVE) vector 
    similarity computation for datasets that fit in memory.
    
    Supported data types:
        - float64 (f64): Double precision floating point
        - float32 (f32): Single precision floating point
        - int8 (i8): 8-bit signed integers
        - uint8 (u8): Binary vectors (for hamming/jaccard metrics)
    
    Supported metrics:
        - Similarity (higher is better):
            - 'dot', 'dot_product', 'dotproduct': Dot product
            - 'cos', 'cosine', 'cosine_similarity': Cosine similarity
            - 'inner', 'inner_product': Inner product
        - Distance (lower is better):
            - 'l2', 'euclidean', 'l2_distance': L2/Euclidean distance
            - 'l2sq', 'l2_squared', 'squared_euclidean': Squared L2 distance
            - 'hamming': Hamming distance (uint8 only)
            - 'jaccard': Jaccard distance (uint8 only)
            - 'kl', 'kl_divergence': Kullback-Leibler divergence
            - 'js', 'js_divergence': Jensen-Shannon divergence
    
    Example:
        >>> engine = VectorSearch()
        >>> print(engine.capabilities())  # Show SIMD features
        >>>
        >>> # Single pair computation
        >>> a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        >>> b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        >>> score = engine.compute_metric(a, b, 'cosine')
        >>>
        >>> # Batch computation
        >>> query = np.random.randn(128).astype(np.float32)
        >>> candidates = np.random.randn(10000, 128).astype(np.float32)
        >>> scores = engine.batch_compute(query, candidates, 'cosine')
        >>>
        >>> # Top-K search
        >>> indices, scores = engine.top_k_search(query, candidates, 'cosine', k=10)
    
    Note:
        For large datasets that don't fit in memory, use StreamingVectorSearch instead.
    """
    
    def __new__(cls) -> 'VectorSearch':
        if _VectorSearch is None:
            raise ImportError("Rust backend not available")
        return _VectorSearch()
    
    def capabilities(self) -> str:
        """Get SIMD capabilities information.
        
        Returns:
            str: A string describing detected SIMD features (e.g., "CPU: AVX2, AVX-512")
        """
        ...
    
    def compute_metric(
        self,
        a: NDArray[np.floating],
        b: NDArray[np.floating],
        metric: MetricType
    ) -> float:
        """Compute the metric value between two vectors.
        
        Args:
            a: First vector (1D numpy array)
            b: Second vector (1D numpy array, same dtype and length as a)
            metric: Metric type string
        
        Returns:
            float: The computed metric value
        
        Raises:
            TypeError: If dtype is not supported or a/b dtypes don't match
            ValueError: If the metric is unknown or dimensions don't match
        """
        ...
    
    def batch_compute(
        self,
        query: NDArray[np.floating],
        candidates: NDArray[np.floating],
        metric: MetricType
    ) -> NDArray[np.float64]:
        """Batch compute metrics between a query vector and multiple candidates.
        
        Uses parallel computation for large batches (>= 500 candidates).
        
        Args:
            query: Query vector (1D numpy array)
            candidates: Candidate vectors matrix (2D numpy array, shape: [N, D])
            metric: Metric type string
        
        Returns:
            numpy.ndarray: Metric values array (1D, shape: [N], dtype: float64)
        
        Raises:
            TypeError: If dtype is not supported or dtypes don't match
            ValueError: If metric is unknown or dimensions don't match
        """
        ...
    
    def top_k_search(
        self,
        query: NDArray[np.floating],
        candidates: NDArray[np.floating],
        metric: MetricType,
        k: int
    ) -> Tuple[NDArray[np.uint64], NDArray[np.float64]]:
        """Top-K search: Find the k most similar/closest vectors.
        
        Args:
            query: Query vector (1D numpy array)
            candidates: Candidate vectors matrix (2D numpy array)
            metric: Metric type string
            k: Number of results to return
        
        Returns:
            tuple: (indices, scores)
                - indices: Array of candidate indices (uint64, shape: [k])
                - scores: Array of metric scores (float64, shape: [k])
                
                For similarity metrics: returns k highest scores
                For distance metrics: returns k lowest scores
        
        Raises:
            TypeError: If dtype is not supported or dtypes don't match
            ValueError: If metric is unknown or dimensions don't match
        """
        ...
    
    def multi_query_top_k(
        self,
        queries: NDArray[np.floating],
        candidates: NDArray[np.floating],
        metric: MetricType,
        k: int
    ) -> Tuple[NDArray[np.uint64], NDArray[np.float64]]:
        """Batch multi-query Top-K search (optimized for multiple queries).
        
        This method processes multiple queries in a single FFI call, significantly
        reducing Python-Rust boundary overhead compared to calling top_k_search repeatedly.
        
        Performance: ~30-50% faster than calling top_k_search in a loop.
        
        Args:
            queries: Multiple query vectors (2D numpy array, shape: [N, D])
            candidates: Candidate vectors matrix (2D numpy array, shape: [M, D])
            metric: Metric type string ('cosine', 'dot', 'l2', etc.)
            k: Number of top results to return per query
        
        Returns:
            tuple: (all_indices, all_scores)
                - all_indices: 1D array of shape [N*k], can reshape to [N, k]
                - all_scores: 1D array of shape [N*k], can reshape to [N, k]
        
        Example:
            >>> indices, scores = engine.multi_query_top_k(queries, candidates, 'cosine', k=10)
            >>> indices = indices.reshape(len(queries), k)
            >>> scores = scores.reshape(len(queries), k)
        """
        ...
    
    def merge_top_k(
        self,
        batch_scores: NDArray[np.float64],
        global_offset: int,
        current_indices: NDArray[np.uint64],
        current_scores: NDArray[np.float64],
        k: int,
        is_similarity: bool
    ) -> Tuple[NDArray[np.uint64], NDArray[np.float64]]:
        """Merge batch scores into existing top-k results.
        
        This is a helper method for implementing custom streaming searches.
        It efficiently merges new batch results with current top-k.
        
        Args:
            batch_scores: Scores for current batch (float64)
            global_offset: Starting global index for this batch
            current_indices: Current top-k indices (uint64, can be empty)
            current_scores: Current top-k scores (float64, can be empty)
            k: Number of top results to maintain
            is_similarity: True = higher is better, False = lower is better
        
        Returns:
            tuple: (new_indices, new_scores) - Updated top-k results
        """
        ...
    
    def batch_compute_and_merge_top_k(
        self,
        query: NDArray[np.floating],
        candidates: NDArray[np.floating],
        metric: MetricType,
        global_offset: int,
        current_indices: NDArray[np.uint64],
        current_scores: NDArray[np.float64],
        k: int,
        is_similarity: bool
    ) -> Tuple[NDArray[np.uint64], NDArray[np.float64]]:
        """Compute batch scores AND merge into top-k in a single FFI call.
        
        This is an optimized method that combines batch_compute + merge_top_k,
        eliminating intermediate Python array allocation and reducing FFI overhead.
        
        Args:
            query: Query vector (1D numpy array)
            candidates: Candidate vectors batch (2D numpy array)
            metric: Metric type string
            global_offset: Starting global index for this batch
            current_indices: Current top-k indices (can be empty)
            current_scores: Current top-k scores (can be empty)
            k: Number of top results to maintain
            is_similarity: True = higher is better, False = lower is better
        
        Returns:
            tuple: (new_indices, new_scores) - Updated top-k results
        """
        ...


class StreamingVectorSearch:
    """Memory-efficient streaming vector search from NumPack files.
    
    StreamingVectorSearch is designed for large datasets that don't fit in memory.
    It reads data directly from NumPack files using memory mapping, processing
    data in batches with a single FFI call for the entire search operation.
    
    Performance characteristics:
        - ~10-30x faster than Python-based streaming
        - Zero-copy mmap file access
        - No intermediate Python array allocation per batch
        - Single FFI call for entire search
    
    Supported data types:
        - float32 (f32): Single precision floating point
        - float64 (f64): Double precision floating point
    
    Example:
        >>> from numpack import NumPack
        >>> from numpack.vector_engine import StreamingVectorSearch
        >>> import numpy as np
        >>>
        >>> streaming = StreamingVectorSearch()
        >>> query = np.random.randn(128).astype(np.float32)
        >>>
        >>> with NumPack('vectors.npk') as npk:
        ...     # Top-K search from file
        ...     indices, scores = streaming.streaming_top_k_from_file(
        ...         query, str(npk._filename), 'embeddings', 'cosine', k=10, batch_size=10000
        ...     )
        ...     
        ...     # Compute all scores from file
        ...     all_scores = streaming.streaming_batch_compute(
        ...         query, str(npk._filename), 'embeddings', 'cosine', batch_size=10000
        ...     )
    
    Note:
        For in-memory operations where all data fits in memory, use VectorSearch instead.
    """
    
    def __new__(cls) -> 'StreamingVectorSearch':
        if _StreamingVectorSearch is None:
            raise ImportError("Rust backend not available")
        return _StreamingVectorSearch()
    
    def capabilities(self) -> str:
        """Get SIMD capabilities information.
        
        Returns:
            str: A string describing detected SIMD features (e.g., "CPU: AVX2, AVX-512")
        """
        ...
    
    def streaming_top_k_from_file(
        self,
        query: NDArray[np.floating],
        npk_dir: str,
        array_name: str,
        metric: MetricType,
        k: int,
        batch_size: int = 10000
    ) -> Tuple[NDArray[np.uint64], NDArray[np.float64]]:
        """Streaming Top-K search directly from NumPack file.
        
        This is a highly optimized streaming search that:
        1. Reads candidates directly from NumPack file using memory mapping
        2. Processes data in batches entirely within Rust
        3. Maintains global Top-K using efficient partial sort
        
        Args:
            query: Query vector (1D numpy array, float32 or float64)
            npk_dir: NumPack directory path (string)
            array_name: Name of candidates array in NumPack file
            metric: Metric type string ('cosine', 'dot', 'l2', etc.)
            k: Number of top results to return
            batch_size: Number of rows to process per batch (default: 10000)
        
        Returns:
            tuple: (indices, scores)
                - indices: Global indices of top-k candidates (uint64)
                - scores: Corresponding metric scores (float64)
        
        Raises:
            TypeError: If query dtype is not float32 or float64
            ValueError: If array not found or dimensions don't match
        """
        ...
    
    def streaming_batch_compute(
        self,
        query: NDArray[np.floating],
        npk_dir: str,
        array_name: str,
        metric: MetricType,
        batch_size: int = 10000
    ) -> NDArray[np.float64]:
        """Streaming batch compute directly from NumPack file.
        
        Computes metric values between query and all candidates without loading
        all data into Python memory.
        
        Args:
            query: Query vector (1D numpy array, float32 or float64)
            npk_dir: NumPack directory path (string)
            array_name: Name of candidates array in NumPack file
            metric: Metric type string ('cosine', 'dot', 'l2', etc.)
            batch_size: Number of rows to process per batch (default: 10000)
        
        Returns:
            numpy.ndarray: All computed metric values (1D array of float64)
        
        Raises:
            TypeError: If query dtype is not float32 or float64
            ValueError: If array not found or dimensions don't match
        """
        ...
    
    def streaming_multi_query_top_k(
        self,
        queries: NDArray[np.floating],
        npk_dir: str,
        array_name: str,
        metric: MetricType,
        k: int,
        batch_size: int = 10000
    ) -> Tuple[NDArray[np.uint64], NDArray[np.float64]]:
        """Batch multi-query Top-K search from file (optimized for multiple queries).
        
        This method opens the file once and executes multiple queries, amortizing
        the file open and metadata loading overhead across all queries.
        
        Performance: ~2x faster than calling streaming_top_k_from_file repeatedly.
        
        Args:
            queries: Multiple query vectors (2D numpy array, shape: [N, D], float32)
            npk_dir: NumPack directory path (string)
            array_name: Name of candidates array in NumPack file
            metric: Metric type string ('cosine', 'dot', 'l2', etc.)
            k: Number of top results to return per query
            batch_size: Number of rows to process per batch (default: 10000)
        
        Returns:
            tuple: (all_indices, all_scores)
                - all_indices: 1D array of shape [N*k], can reshape to [N, k]
                - all_scores: 1D array of shape [N*k], can reshape to [N, k]
        
        Raises:
            TypeError: If queries dtype is not float32
            ValueError: If array not found or dimensions don't match
        
        Example:
            >>> indices, scores = streaming.streaming_multi_query_top_k(
            ...     queries, str(npk_path), 'candidates', 'cosine', k=10
            ... )
            >>> indices = indices.reshape(len(queries), k)
            >>> scores = scores.reshape(len(queries), k)
        """
        ...


__all__ = [
    'VectorSearch',
    'StreamingVectorSearch',
    'MetricType',
]
