"""Vector similarity engines backed by the Rust SIMD implementation.

This module exposes two public classes:

- `VectorEngine`: In-memory vector similarity / distance computation.
- `StreamingVectorEngine`: Streaming Top-K search directly from a NumPack file.

Notes
-----
The implementations are provided by the Rust extension module and are exposed in
Python as thin wrappers.

Examples
--------
In-memory Top-K search:

>>> from numpack.vector_engine import VectorEngine
>>> import numpy as np
>>> engine = VectorEngine()
>>> query = np.random.randn(128).astype(np.float32)
>>> candidates = np.random.randn(10000, 128).astype(np.float32)
>>> indices, scores = engine.top_k_search(query, candidates, 'cosine', k=10)

Streaming Top-K search from a NumPack file:

>>> from numpack import NumPack
>>> from numpack.vector_engine import StreamingVectorEngine
>>> streaming = StreamingVectorEngine()
>>> query = np.random.randn(128).astype(np.float32)
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
    from numpack._lib_numpack import VectorEngine as _VectorEngine
    from numpack._lib_numpack import StreamingVectorEngine as _StreamingVectorEngine
except ImportError:
    # Fallback for type checking
    _VectorEngine = None
    _StreamingVectorEngine = None

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


# Dot product metric aliases
_DOT_METRICS = {'dot', 'dot_product', 'dotproduct', 'inner', 'inner_product'}


class VectorEngine:
    """In-memory vector similarity / distance computation.

    This class is backed by the Rust SIMD implementation (AVX2/AVX-512/NEON/SVE
    depending on the CPU). Use it when the candidate matrix fits in memory.

    Notes
    -----
    - **Mixed dtypes**: if `query` and `candidates` dtypes differ, both are
      converted to `float64` for computation.
    - **Metric direction**:

      - Similarity metrics: higher is better (e.g. cosine, dot).
      - Distance metrics: lower is better (e.g. L2).
    - **NumPy fast path**: For multi-query dot product operations, NumPy's
      optimized matrix multiplication is used automatically for best performance.

    Examples
    --------
    >>> import numpy as np
    >>> from numpack.vector_engine import VectorEngine
    >>> engine = VectorEngine()
    >>> query = np.random.randn(128).astype(np.float32)
    >>> candidates = np.random.randn(10000, 128).astype(np.float32)
    >>> indices, scores = engine.top_k_search(query, candidates, 'cosine', k=10)
    """

    def __init__(self) -> None:
        if _VectorEngine is None:
            raise ImportError("Rust backend not available")
        self._backend = _VectorEngine()

    def capabilities(self) -> str:
        """Return detected SIMD capabilities.

        Returns
        -------
        str
            Human-readable SIMD feature summary (for example, ``"CPU: AVX2"").
        """
        return self._backend.capabilities()

    def compute_metric(
        self,
        a: NDArray,
        b: NDArray,
        metric: MetricType
    ) -> float:
        """Compute the metric value between two vectors.

        Parameters
        ----------
        a : numpy.ndarray
            First vector (1D).
        b : numpy.ndarray
            Second vector (1D). Must have the same length as `a`.
        metric : MetricType
            Metric identifier (for example, ``"cosine"``, ``"dot"``, ``"l2"``).

        Returns
        -------
        float
            The computed metric value.

        Raises
        ------
        TypeError
            If dtypes are not supported.
        ValueError
            If `metric` is unknown or the dimensions do not match.
        """
        return self._backend.compute_metric(a, b, metric)

    def batch_compute(
        self,
        query: NDArray,
        candidates: NDArray,
        metric: MetricType
    ) -> NDArray[np.float64]:
        """Compute metric values between query vector(s) and many candidates.

        Parameters
        ----------
        query : numpy.ndarray
            Query vector(s). Can be:
            - 1D array (shape ``(dim,)``): Single query vector
            - 2D array (shape ``(n_queries, dim)``): Multiple query vectors
        candidates : numpy.ndarray
            Candidate matrix (2D, shape ``(n_candidates, dim)``).
        metric : MetricType
            Metric identifier.

        Returns
        -------
        numpy.ndarray
            Metric values array, dtype ``float64``:
            - If query is 1D: returns 1D array of shape ``(n_candidates,)``
            - If query is 2D: returns 2D array of shape ``(n_queries, n_candidates)``

        Raises
        ------
        TypeError
            If dtypes are not supported or query has more than 2 dimensions.
        ValueError
            If `metric` is unknown or the dimensions do not match.

        Examples
        --------
        Single query:
        >>> scores = engine.batch_compute(query, candidates, 'cosine')  # shape: (n_candidates,)

        Multiple queries (batch mode):
        >>> queries = np.random.randn(5, 128).astype(np.float32)
        >>> scores = engine.batch_compute(queries, candidates, 'cosine')  # shape: (5, n_candidates)
        """
        # NumPy fast path for multi-query dot product
        # NumPy's direct BLAS call has less overhead than Rust FFI
        if query.ndim == 2 and metric in _DOT_METRICS:
            return query @ candidates.T
        
        # Use Rust SIMD backend for other metrics (cosine, L2, etc.)
        return self._backend.batch_compute(query, candidates, metric)

    def top_k_search(
        self,
        query: NDArray,
        candidates: NDArray,
        metric: MetricType,
        k: int
    ) -> Tuple[NDArray[np.uint64], NDArray[np.float64]]:
        """Return the Top-K candidates for a single query.

        Parameters
        ----------
        query : numpy.ndarray
            Query vector (1D).
        candidates : numpy.ndarray
            Candidate matrix (2D, shape ``(n_candidates, dim)``).
        metric : MetricType
            Metric identifier.
        k : int
            Number of results to return.

        Returns
        -------
        indices : numpy.ndarray
            Indices of the selected candidates, shape ``(k,)``, dtype ``uint64``.
        scores : numpy.ndarray
            Corresponding metric values, shape ``(k,)``, dtype ``float64``.

        Notes
        -----
        - For similarity metrics, this returns the **largest** `k` scores.
        - For distance metrics, this returns the **smallest** `k` scores.

        Raises
        ------
        TypeError
            If dtypes are not supported.
        ValueError
            If `metric` is unknown or the dimensions do not match.
        """
        return self._backend.top_k_search(query, candidates, metric, k)

    def multi_query_top_k(
        self,
        queries: NDArray,
        candidates: NDArray,
        metric: MetricType,
        k: int
    ) -> Tuple[NDArray[np.uint64], NDArray[np.float64]]:
        """Top-K search for multiple queries in one call.

        Compared to calling `top_k_search` in a Python loop, this method reduces
        Python-to-Rust overhead by performing the computation in a single FFI call.

        Parameters
        ----------
        queries : numpy.ndarray
            Query matrix (2D, shape ``(n_queries, dim)``).
        candidates : numpy.ndarray
            Candidate matrix (2D, shape ``(n_candidates, dim)``).
        metric : MetricType
            Metric identifier.
        k : int
            Number of results to return per query.

        Returns
        -------
        indices : numpy.ndarray
            Flattened indices, shape ``(n_queries * k,)``, dtype ``uint64``.
        scores : numpy.ndarray
            Flattened scores, shape ``(n_queries * k,)``, dtype ``float64``.

        Examples
        --------
        >>> indices, scores = engine.multi_query_top_k(queries, candidates, 'cosine', k=10)
        >>> indices = indices.reshape(len(queries), 10)
        >>> scores = scores.reshape(len(queries), 10)
        """
        return self._backend.multi_query_top_k(queries, candidates, metric, k)

    def segmented_top_k_search(
        self,
        query: NDArray,
        candidates: NDArray,
        metric: MetricType,
        global_offset: int,
        current_indices: NDArray[np.uint64],
        current_scores: NDArray[np.float64],
        k: int,
        is_similarity: bool
    ) -> Tuple[NDArray[np.uint64], NDArray[np.float64]]:
        """Incrementally update Top-K results for a batch of candidates.

        This method is intended for segmented processing where you stream candidate
        vectors in batches and want to merge each batch into a running Top-K.

        Parameters
        ----------
        query : numpy.ndarray
            Query vector (1D).
        candidates : numpy.ndarray
            Candidate batch (2D, shape ``(batch, dim)``).
        metric : MetricType
            Metric identifier.
        global_offset : int
            Global starting index corresponding to row 0 of `candidates`.
        current_indices : numpy.ndarray
            Current Top-K indices (can be empty), dtype ``uint64``.
        current_scores : numpy.ndarray
            Current Top-K scores (can be empty), dtype ``float64``.
        k : int
            Number of results to keep.
        is_similarity : bool
            If True, larger scores are better; if False, smaller scores are better.

        Returns
        -------
        indices : numpy.ndarray
            Updated indices, shape ``(k,)``, dtype ``uint64``.
        scores : numpy.ndarray
            Updated scores, shape ``(k,)``, dtype ``float64``.

        Examples
        --------
        >>> engine = VectorEngine()
        >>> query = np.random.randn(128).astype(np.float32)
        >>> current_indices = np.array([], dtype=np.uint64)
        >>> current_scores = np.array([], dtype=np.float64)
        >>> for batch_idx, candidates_batch in enumerate(data_batches):
        ...     current_indices, current_scores = engine.segmented_top_k_search(
        ...         query,
        ...         candidates_batch,
        ...         'cosine',
        ...         global_offset=batch_idx * batch_size,
        ...         current_indices=current_indices,
        ...         current_scores=current_scores,
        ...         k=10,
        ...         is_similarity=True,
        ...     )
        """
        return self._backend.segmented_top_k_search(
            query, candidates, metric, global_offset,
            current_indices, current_scores, k, is_similarity
        )


class StreamingVectorEngine:
    """Streaming vector search directly from NumPack files.

    This engine is intended for large candidate sets that do not fit in memory.
    Candidates are read from the NumPack data file via memory mapping and processed
    in batches inside Rust.

    Notes
    -----
    - Query vectors are validated on the Python side, then passed into the Rust
      implementation.
    - For in-memory data, prefer `VectorEngine`.

    Examples
    --------
    >>> import numpy as np
    >>> from numpack import NumPack
    >>> from numpack.vector_engine import StreamingVectorEngine
    >>> streaming = StreamingVectorEngine()
    >>> query = np.random.randn(128).astype(np.float32)
    >>> with NumPack('vectors.npk') as npk:
    ...     indices, scores = streaming.streaming_top_k_from_file(
    ...         query, str(npk._filename), 'embeddings', 'cosine', k=10
    ...     )
    """

    def __new__(cls) -> 'StreamingVectorEngine':
        if _StreamingVectorEngine is None:
            raise ImportError("Rust backend not available")
        return _StreamingVectorEngine()

    def capabilities(self) -> str:
        """Return detected SIMD capabilities.

        Returns
        -------
        str
            Human-readable SIMD feature summary.
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
        """Streaming Top-K search from a NumPack file.

        Parameters
        ----------
        query : numpy.ndarray
            Query vector (1D). Must be a floating dtype supported by the backend.
        npk_dir : str
            Path to the NumPack directory.
        array_name : str
            Name of the candidate array stored in the file.
        metric : MetricType
            Metric identifier.
        k : int
            Number of results to return.
        batch_size : int, optional
            Number of rows processed per batch.

        Returns
        -------
        indices : numpy.ndarray
            Global indices of the Top-K candidates, dtype ``uint64``.
        scores : numpy.ndarray
            Corresponding metric values, dtype ``float64``.

        Raises
        ------
        TypeError
            If `query` dtype is not supported.
        ValueError
            If the array does not exist or the dimensions do not match.
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
        """Compute metric values against all candidates in a NumPack file.

        Parameters
        ----------
        query : numpy.ndarray
            Query vector(s). Can be:
            - 1D array (shape ``(dim,)``): Single query vector
            - 2D array (shape ``(n_queries, dim)``): Multiple query vectors
        npk_dir : str
            Path to the NumPack directory.
        array_name : str
            Name of the candidate array stored in the file.
        metric : MetricType
            Metric identifier.
        batch_size : int, optional
            Number of rows processed per batch.

        Returns
        -------
        numpy.ndarray
            Metric values array, dtype ``float64``:
            - If query is 1D: returns 1D array of shape ``(n_candidates,)``
            - If query is 2D: returns 2D array of shape ``(n_queries, n_candidates)``

        Raises
        ------
        TypeError
            If `query` dtype is not supported or query has more than 2 dimensions.
        ValueError
            If the array does not exist or the dimensions do not match.

        Examples
        --------
        Single query:
        >>> scores = streaming.streaming_batch_compute(query, npk_dir, 'embeddings', 'cosine')

        Multiple queries (batch mode):
        >>> queries = np.random.randn(5, 128).astype(np.float32)
        >>> scores = streaming.streaming_batch_compute(queries, npk_dir, 'embeddings', 'cosine')
        >>> # scores.shape == (5, n_candidates)
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
        """Top-K search for multiple queries against a NumPack file.

        The file is opened once and reused for all queries.

        Parameters
        ----------
        queries : numpy.ndarray
            Query matrix (2D, shape ``(n_queries, dim)``).
        npk_dir : str
            Path to the NumPack directory.
        array_name : str
            Name of the candidate array stored in the file.
        metric : MetricType
            Metric identifier.
        k : int
            Number of results to return per query.
        batch_size : int, optional
            Number of rows processed per batch.

        Returns
        -------
        indices : numpy.ndarray
            Flattened indices, shape ``(n_queries * k,)``, dtype ``uint64``.
        scores : numpy.ndarray
            Flattened scores, shape ``(n_queries * k,)``, dtype ``float64``.

        Raises
        ------
        TypeError
            If `queries` dtype is not supported.
        ValueError
            If the array does not exist or the dimensions do not match.

        Examples
        --------
        >>> indices, scores = streaming.streaming_multi_query_top_k(
        ...     queries, str(npk_path), 'candidates', 'cosine', k=10
        ... )
        >>> indices = indices.reshape(len(queries), 10)
        >>> scores = scores.reshape(len(queries), 10)
        """
        ...


__all__ = [
    'VectorEngine',
    'StreamingVectorEngine',
    'MetricType',
]
