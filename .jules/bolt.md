## 2024-08-14 - Optimize JSON Chunk Loading in Retrieval
**Learning:** Initializing multiple retrievers (e.g., standard BM25 and image BM25) caused redundant disk reads of chunk JSON files.
**Action:** Cached the raw chunk dictionaries in memory using `_load_all_chunks()` and `_cached_raw_chunks` to prevent N+1 file operations when instantiating multiple retrievers.
