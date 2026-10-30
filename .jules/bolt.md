## 2024-05-17 - [Redundant Query Embedding Inference]
**Learning:** In multi-stage RAG pipelines, invoking `embed_query` on the user query redundantly at each stage (e.g., dense retrieval, image BM25 reranking, general MMR reranking) wastes significant CPU cycles per request.
**Action:** Use `functools.lru_cache` to memoize the `embed_query` method on the thread-safe singleton model instance so the embedding for identical queries is computed exactly once.

## 2024-05-18 - [BLAS GEMV Dot Product Optimization]
**Learning:** Implicit NumPy upcasting of dense embeddings (e.g., mixing float32 and float64) silently disables fast BLAS GEMV dot products, causing suboptimal performance for cosine similarity operations.
**Action:** Always wrap query and document embedding matrices in `np.ascontiguousarray(..., dtype=np.float32)` prior to calculating similarities to ensure C-contiguous memory alignment and peak vectorization performance.
## 2024-05-19 - [Redundant Query Embedding Inference - Pydantic Bypass]
**Learning:** In multi-stage RAG pipelines, invoking `embed_query` on the user query redundantly at each stage (e.g., dense retrieval, image BM25 reranking, general MMR reranking) wastes significant CPU cycles per request. However, when the model is a Pydantic `BaseModel` (like `HuggingFaceEmbeddings`), using standard attribute assignment (e.g., `model.embed_query = lru_cache(...)`) fails because Pydantic intercepts `__setattr__`.
**Action:** Use `functools.lru_cache` to memoize the `embed_query` method on the thread-safe singleton model instance, but bypass Pydantic's validation by using `object.__setattr__(instance, 'embed_query', memoized_method)` instead of standard assignment.

## 2024-05-20 - [Redundant Regex Compilation in Document Iteration]
**Learning:** When using regular expressions in tight loops (e.g., iterating over retrieved vector chunks and image chunks), passing raw string patterns to `re.sub()` causes redundant compilation and cache-lookup overhead on every iteration.
**Action:** Always extract and pre-compile the patterns as module-level constants (e.g., `_PATTERN_RE = re.compile(...)`) to eliminate this overhead.
