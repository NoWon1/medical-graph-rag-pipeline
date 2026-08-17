## 2026-08-17 - Avoid redundant disk I/O when parsing previously loaded chunks
**Learning:** Found redundant initialization logic where the same JSON files were read from disk multiple times during startup by separate retrievers.
**Action:** Reused the documents array already parsed and stored in memory by `get_bm25_retriever().docs` inside `get_image_bm25_retriever()` instead of making multiple `open()` and `json.load()` calls. Demonstrated a ~8.7x latency improvement on dummy data.
