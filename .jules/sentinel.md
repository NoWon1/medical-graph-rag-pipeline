## 2025-02-09 - Path Traversal in Image Rendering
**Vulnerability:** Local File Inclusion / Path Traversal vulnerability in Streamlit app where `filename` from LLM tags (`[IMAGE: filename]`) was directly appended to `IMAGE_DIR` without sanitization.
**Learning:** LLM outputs must be treated as untrusted user input. A hallucinated or maliciously crafted tag could lead the application to read arbitrary files from the filesystem.
**Prevention:** Always extract the safe basename using `Path(filename).name` and strictly verify the resolved path remains relative to the intended base directory using `is_relative_to()`. Use `.is_file()` instead of `.exists()` to prevent directory access.

## 2025-02-09 - Information Exposure in Exception Handling
**Vulnerability:** Information Exposure / Information Leakage vulnerability in Streamlit app where `st.exception(e)` and `st.error(f"Error: {e}")` were used, exposing internal stack traces and error details to the end user.
**Learning:** Returning raw exception strings or using debugging UI elements like `st.exception()` in production applications leaks implementation details, which can be exploited by attackers to understand the system architecture.
**Prevention:** Catch exceptions, log detailed errors safely to the server console using `logging.error(..., exc_info=True)`, and use generic fallback messages (e.g., `st.error("An error occurred")`) for user-facing UI.
