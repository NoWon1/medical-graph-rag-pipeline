## 2024-05-17 - [Redundant Query Embedding Inference]
**Learning:** In multi-stage RAG pipelines, invoking `embed_query` on the user query redundantly at each stage (e.g., dense retrieval, image BM25 reranking, general MMR reranking) wastes significant CPU cycles per request.
**Action:** Use `functools.lru_cache` to memoize the `embed_query` method on the thread-safe singleton model instance so the embedding for identical queries is computed exactly once.
