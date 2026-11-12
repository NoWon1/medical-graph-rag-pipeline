## 2024-05-17 - [Redundant Query Embedding Inference]
**Learning:** In multi-stage RAG pipelines, invoking `embed_query` on the user query redundantly at each stage (e.g., dense retrieval, image BM25 reranking, general MMR reranking) wastes significant CPU cycles per request.
**Action:** Use `functools.lru_cache` to memoize the `embed_query` method on the thread-safe singleton model instance so the embedding for identical queries is computed exactly once.

## 2024-05-18 - [BLAS GEMV Dot Product Optimization]
**Learning:** Implicit NumPy upcasting of dense embeddings (e.g., mixing float32 and float64) silently disables fast BLAS GEMV dot products, causing suboptimal performance for cosine similarity operations.
**Action:** Always wrap query and document embedding matrices in `np.ascontiguousarray(..., dtype=np.float32)` prior to calculating similarities to ensure C-contiguous memory alignment and peak vectorization performance.

## 2024-05-19 - [Redundant Query Embedding Inference - Pydantic Bypass]
**Learning:** In multi-stage RAG pipelines, invoking `embed_query` on the user query redundantly at each stage (e.g., dense retrieval, image BM25 reranking, general MMR reranking) wastes significant CPU cycles per request. However, when the model is a Pydantic `BaseModel` (like `HuggingFaceEmbeddings`), using standard attribute assignment (e.g., `model.embed_query = lru_cache(...)`) fails because Pydantic intercepts `__setattr__`.
**Action:** Use `functools.lru_cache` to memoize the `embed_query` method on the thread-safe singleton model instance, but bypass Pydantic's validation by using `object.__setattr__(instance, 'embed_query', memoized_method)` instead of standard assignment.

## 2024-05-20 - [Redundant Computation in Streamlit UI Loop]
**Learning:** In Streamlit, the entire script executes top-to-bottom on every user event (e.g., chat submission). Running expensive string operations like `hashlib.md5` on large text (like a patient report) repeatedly across interactive turns scales poorly and creates unneeded CPU overhead in the main application loop.
**Action:** Always wrap deterministic, computationally heavy pure functions (like hashing large strings) with `@st.cache_data(show_spinner=False)` to memoize the result and ensure constant-time O(1) lookups on subsequent Streamlit reruns.

## 2024-05-21 - [Redundant Document Embedding Inference in Rerankers]
**Learning:** During MMR and image semantic reranking, `embed_documents` is called repeatedly on static, frequently-retrieved chunks. Since the embeddings are not cached on the retrieved chunks, this forces redundant CPU-bound model inference, creating a severe bottleneck when processing the same top-k BM25 or Dense search results.
**Action:** Implement an LRU cache (`OrderedDict`) to memoize the `embed_documents` method on the thread-safe singleton model instance, bypassing Pydantic's `__setattr__`, to ensure static text chunks are only embedded once across sequential turns or concurrent requests.
