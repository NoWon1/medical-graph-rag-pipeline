## 2026-07-23 - Optimise vector similarity calculations

**Learning:** When dealing with intensive cosine similarity or distance metric loops on vector embeddings, pure Python loops over lists are excessively slow. Using native Python `sum` and `math.sqrt` causes high overhead in hot paths like `mmr_rerank` or `_retrieve_image_chunks`. Pre-casting arrays to `numpy.ndarray` outside of the loops and vectorising the computation dramatically reduces evaluation time while keeping the same output structure (as long as it casts back to `float` for JSON serialization safety).

**Action:** Whenever a hot path iterates over vector comparisons, pre-cast inputs to numpy arrays before the loop. Use `np.dot` and `np.linalg.norm` and avoid mixed Python/NumPy loop computations. Always cast the final result to a standard Python `float()` to prevent downstream type errors.
