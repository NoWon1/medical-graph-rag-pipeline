## 2024-05-24 - Numpy Vectorization

**Learning:** Pure Python nested loops for mathematical operations like calculating cosine similarity or executing Maximal Marginal Relevance (MMR) algorithms are significant performance bottlenecks, especially during dense vector retrieval.
**Action:** When working with vector embeddings, use `numpy` array operations (e.g., `np.dot`, `np.linalg.norm`) for vectorized calculations instead of Python generators or sum functions, pre-casting data lists to `numpy` arrays before running similarity comparisons over iterative arrays.
