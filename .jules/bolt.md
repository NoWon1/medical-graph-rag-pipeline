## 2024-05-17 - [Redundant Query Embedding Inference]
**Learning:** In multi-stage RAG pipelines, invoking `embed_query` on the user query redundantly at each stage (e.g., dense retrieval, image BM25 reranking, general MMR reranking) wastes significant CPU cycles per request.
**Action:** Use `functools.lru_cache` to memoize the `embed_query` method on the thread-safe singleton model instance so the embedding for identical queries is computed exactly once.

## 2024-10-25 - [Pre-Normalization & BLAS Dtype Alignment]
**Learning:** Implicit NumPy upcasting of dense embeddings to float64 silently disables fast BLAS GEMV dot products.
**Action:** Always wrap query and document embeddings in np.ascontiguousarray(..., dtype=np.float32) prior to calculating similarities to ensure peak performance.
