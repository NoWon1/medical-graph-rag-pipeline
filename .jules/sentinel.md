## 2025-02-09 - Path Traversal in Image Rendering
**Vulnerability:** Local File Inclusion / Path Traversal vulnerability in Streamlit app where `filename` from LLM tags (`[IMAGE: filename]`) was directly appended to `IMAGE_DIR` without sanitization.
**Learning:** LLM outputs must be treated as untrusted user input. A hallucinated or maliciously crafted tag could lead the application to read arbitrary files from the filesystem.
**Prevention:** Always extract the safe basename using `Path(filename).name` and strictly verify the resolved path remains relative to the intended base directory using `is_relative_to()`. Use `.is_file()` instead of `.exists()` to prevent directory access.
## 2025-02-12 - Exception Leakage in Streamlit App
**Vulnerability:** Information Disclosure / Exception Leakage. The `cancer_app.py` directly caught generic Exceptions and exposed the raw stacktrace/exception object to the user interface via `st.exception(e)`.
**Learning:** Returning raw exception strings (`str(e)`) or using `st.exception()` exposes internal stack traces and implementation details which can be leveraged by attackers to better understand the system architecture and its weaknesses.
**Prevention:** Avoid using `st.exception()` or returning raw exceptions to the UI or API. Always log detailed errors securely server-side using `logging.error(..., exc_info=True)` and use `st.error()` or similar to display generic, secure fallback messages to the user.
