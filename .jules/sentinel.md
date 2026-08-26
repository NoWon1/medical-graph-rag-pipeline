## 2025-02-09 - Path Traversal in Image Rendering
**Vulnerability:** Local File Inclusion / Path Traversal vulnerability in Streamlit app where `filename` from LLM tags (`[IMAGE: filename]`) was directly appended to `IMAGE_DIR` without sanitization.
**Learning:** LLM outputs must be treated as untrusted user input. A hallucinated or maliciously crafted tag could lead the application to read arbitrary files from the filesystem.
**Prevention:** Always extract the safe basename using `Path(filename).name` and strictly verify the resolved path remains relative to the intended base directory using `is_relative_to()`. Use `.is_file()` instead of `.exists()` to prevent directory access.
## 2024-05-18 - Fix stack trace exposure in error handling
**Vulnerability:** The application was exposing detailed stack traces to the user interface via `st.exception(e)` and `st.error(f"... {e}")` in `cancer_app.py`.
**Learning:** Returning raw exception strings or stack traces directly to the UI leaks internal implementation details, which could be exploited by an attacker to understand the system architecture or identify vulnerable components.
**Prevention:** Always catch exceptions and return generic, safe error messages to the user (e.g., "An error occurred. Please check server logs."). Log the detailed exception, including the stack trace, securely to the server console or a logging system using `logging.error(..., exc_info=True)`.
## 2024-05-18 - Fix stack trace exposure in streaming endpoints
**Vulnerability:** The application was exposing internal exceptions directly to the user interface via generator streams in `generate_answer_stream` and `generate_answer` functions in `cancer_retrieval.py`.
**Learning:** Generator functions in backends can bypass frontend exception handling. Returning stringified raw exceptions (`str(e)`) leaks internal implementation details.
**Prevention:** Always catch exceptions in streaming or retrieval pipelines and return/yield generic, safe error messages to the user. Log the detailed exception securely to the server console using `logging.error(..., exc_info=True)`.
