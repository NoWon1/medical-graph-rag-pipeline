## 2025-02-13 - [Path Traversal in Streamlit Image Rendering]
**Vulnerability:** User-provided filename in `[IMAGE: ...]` tags was directly appended to the `IMAGE_DIR` path and rendered, allowing arbitrary local file read (path traversal).
**Learning:** `pathlib.Path(dir) / filename` does not inherently sanitize `filename` against `../` or absolute paths.
**Prevention:** Always extract just the final file name using `.name` (e.g. `Path(filename).name`) before appending to a base directory when constructing paths from untrusted input, and verify with `.is_file()`.

## 2025-02-13 - [Information Disclosure in Error Handling]
**Vulnerability:** Using `st.exception(e)` exposed the full Python stack trace and potential environment specifics to the end-user on the Streamlit frontend.
**Learning:** Streamlit's `st.exception()` is designed for debugging and development, not for production error handling where users can see it.
**Prevention:** Use `st.error()` with generic, user-friendly messages for frontend presentation. Log the actual exception details securely on the backend (e.g. using `logging` or `print`).
