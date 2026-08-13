## 2025-02-09 - Path Traversal in Image Rendering
**Vulnerability:** Local File Inclusion / Path Traversal vulnerability in Streamlit app where `filename` from LLM tags (`[IMAGE: filename]`) was directly appended to `IMAGE_DIR` without sanitization.
**Learning:** LLM outputs must be treated as untrusted user input. A hallucinated or maliciously crafted tag could lead the application to read arbitrary files from the filesystem.
**Prevention:** Always extract the safe basename using `Path(filename).name` and strictly verify the resolved path remains relative to the intended base directory using `is_relative_to()`. Use `.is_file()` instead of `.exists()` to prevent directory access.
## 2025-02-09 - Information Exposure via Unhandled Exceptions
**Vulnerability:** Information leakage through unhandled or explicitly rendered exceptions (`st.exception(e)` and raw `str(e)`) in Streamlit application.
**Learning:** Directly exposing internal error details, stack traces, or exception messages to the user interface provides attackers with valuable insights into the underlying system architecture, file paths, and potential entry points.
**Prevention:** Avoid rendering raw exceptions to the frontend. Implement robust error boundaries, log detailed error information safely on the server side using `logging.error(..., exc_info=True)`, and display generic, user-friendly error messages to the client.
