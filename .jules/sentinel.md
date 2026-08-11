## 2025-02-09 - Path Traversal in Image Rendering
**Vulnerability:** Local File Inclusion / Path Traversal vulnerability in Streamlit app where `filename` from LLM tags (`[IMAGE: filename]`) was directly appended to `IMAGE_DIR` without sanitization.
**Learning:** LLM outputs must be treated as untrusted user input. A hallucinated or maliciously crafted tag could lead the application to read arbitrary files from the filesystem.
**Prevention:** Always extract the safe basename using `Path(filename).name` and strictly verify the resolved path remains relative to the intended base directory using `is_relative_to()`. Use `.is_file()` instead of `.exists()` to prevent directory access.
## 2025-02-09 - Information Leakage via Stack Traces in Streamlit UI
**Vulnerability:** Application uses `st.exception(e)` and returns raw exception strings (`str(e)`) in `st.error()`, leaking internal stack traces and implementation details to the end user.
**Learning:** In Streamlit applications, exposing raw exception strings or stack traces to the UI or API leaks internal information.
**Prevention:** Avoid using `st.exception()` or raw exception strings in the UI. Instead, log detailed errors safely to the server console using `logging.error(..., exc_info=True)` and use `st.error()` to display generic, secure fallback messages to the user.
