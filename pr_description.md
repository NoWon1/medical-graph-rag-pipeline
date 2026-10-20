💡 What: Pre-compiled the regex used to clean up chunk IDs (`r'_cap_\d+$|_\d{4}$'`) by defining it as a module-level constant (`_CHUNK_ID_RE`) in `cancer_retrieval.py` and using `_CHUNK_ID_RE.sub()` instead of `re.sub()`. Added a comment explaining the optimization. Appended `__pycache__/` to `.gitignore`.

🎯 Why: The original implementation called `re.sub(r'...', ...)` inside loops (`build_context` and `_build_sources`), which forced Python to parse and compile the regex on every iteration, leading to redundant compilation overhead in hot retrieval paths.

📊 Impact: Reduces retrieval latency by eliminating the overhead of regex compilation during the context-building phase, leading to a faster response time.

🔬 Measurement: Review the response times of the retrieval pipeline in a benchmarking environment before and after this change. Run `python -m pytest test_cancer_app.py` to verify functionality.
