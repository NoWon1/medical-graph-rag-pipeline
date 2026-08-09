## 2025-02-09 - Path Traversal in Image Rendering
**Vulnerability:** Local File Inclusion / Path Traversal vulnerability in Streamlit app where `filename` from LLM tags (`[IMAGE: filename]`) was directly appended to `IMAGE_DIR` without sanitization.
**Learning:** LLM outputs must be treated as untrusted user input. A hallucinated or maliciously crafted tag could lead the application to read arbitrary files from the filesystem.
**Prevention:** Always extract the safe basename using `Path(filename).name` and strictly verify the resolved path remains relative to the intended base directory using `is_relative_to()`. Use `.is_file()` instead of `.exists()` to prevent directory access.

## 2025-02-12 - Information Leakage via Stack Traces in Streamlit
**Vulnerability:** Streamlit app was directly printing stack traces via `st.exception(e)` and raw exception strings in `st.error(f"Error: {e}")` when unhandled exceptions occurred in the pipeline or report analysis.
**Learning:** Returning raw stack traces and exception strings to the UI can expose sensitive internal application logic, file paths, dependency versions, or data schema details to an attacker.
**Prevention:** Avoid `st.exception()` for production apps. Use `logging.error(..., exc_info=True)` to log full details safely on the server backend, and present only generic, safe error messages to the user via `st.error()`.
