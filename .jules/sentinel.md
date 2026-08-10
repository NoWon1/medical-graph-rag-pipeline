## 2025-02-09 - Path Traversal in Image Rendering
**Vulnerability:** Local File Inclusion / Path Traversal vulnerability in Streamlit app where `filename` from LLM tags (`[IMAGE: filename]`) was directly appended to `IMAGE_DIR` without sanitization.
**Learning:** LLM outputs must be treated as untrusted user input. A hallucinated or maliciously crafted tag could lead the application to read arbitrary files from the filesystem.
**Prevention:** Always extract the safe basename using `Path(filename).name` and strictly verify the resolved path remains relative to the intended base directory using `is_relative_to()`. Use `.is_file()` instead of `.exists()` to prevent directory access.
## 2025-02-10 - Information Leakage via Streamlit Exception Handling
**Vulnerability:** Leaking internal stack traces and implementation details to the UI by using `st.exception(e)` and raw exception strings in error messages.
**Learning:** Stack traces can expose sensitive architecture details to end users.
**Prevention:** Always log exceptions safely to the server console using `logging.error(..., exc_info=True)` and use `st.error()` with a generic, secure fallback message.
