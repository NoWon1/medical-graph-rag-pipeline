## 2024-05-17 - [Redundant Query Embedding Inference]
**Learning:** In multi-stage RAG pipelines, invoking `embed_query` on the user query redundantly at each stage (e.g., dense retrieval, image BM25 reranking, general MMR reranking) wastes significant CPU cycles per request.
**Action:** Use `functools.lru_cache` to memoize the `embed_query` method on the thread-safe singleton model instance so the embedding for identical queries is computed exactly once.

## 2024-05-18 - [BLAS GEMV Dot Product Optimization]
**Learning:** Implicit NumPy upcasting of dense embeddings (e.g., mixing float32 and float64) silently disables fast BLAS GEMV dot products, causing suboptimal performance for cosine similarity operations.
**Action:** Always wrap query and document embedding matrices in `np.ascontiguousarray(..., dtype=np.float32)` prior to calculating similarities to ensure C-contiguous memory alignment and peak vectorization performance.
