## 2024-05-24 - Vectorize Cosine Similarity for Performance
**Learning:** When calculating vector similarities in hot loops, pure Python math functions introduce significant overhead compared to `numpy`. However, `numpy` scalar results (like `np.float64` from `np.dot`) can break JSON serialization later in the API pipeline if not carefully handled. Also, passing standard lists into a vectorized function inside a nested loop causes implicit type-casting overhead on every iteration.
**Action:** Always pre-cast variables to `np.array` *outside* of hot loops, and explicitly cast the final scalar result back to a native Python `float()` before returning it from the similarity function. Ensure type hints use `Union[List[float], np.ndarray]` to prevent linter/type errors.

## 2024-05-18 - Vectorized MMR Reranking Optimization
**Learning:** Pre-casting Python lists to NumPy arrays before a nested loop is good, but performing `O(N^2)` pairwise vector similarity computations in a Python loop (like `_cosine`) creates a major CPU bottleneck for RAG reranking.
**Action:** Replace iterative dot product loops with fully vectorized L2 normalization and a single full similarity matrix pre-calculation (`np.dot(doc_mat, doc_mat.T)`). This converts O(n^2) Python iterations into highly optimized O(1) NumPy matrix multiplications.
