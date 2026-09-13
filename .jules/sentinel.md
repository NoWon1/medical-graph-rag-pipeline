## 2024-09-13 - Code Health Improvement in `cancer_ingestion.py`
**Vulnerability:** The `clean_text` function in `cancer_ingestion.py` was initializing large dictionaries and compiling regexes on every call.
**Learning:** This approach causes performance overhead and makes the code harder to read. Extracting dictionaries and pre-compiling regexes at the module level solves this. Ensure exact patterns are preserved (e.g., `_DOCLING_RE = re.compile(r'', flags=re.IGNORECASE)` which matches the previously empty pattern `r''`).
**Prevention:** Always define constants and compile regex expressions outside of frequently called functions to optimize execution time and memory.
