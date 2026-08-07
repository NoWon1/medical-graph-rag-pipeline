## 2025-02-09 - Path Traversal in Image Rendering
**Vulnerability:** Local File Inclusion / Path Traversal vulnerability in Streamlit app where `filename` from LLM tags (`[IMAGE: filename]`) was directly appended to `IMAGE_DIR` without sanitization.
**Learning:** LLM outputs must be treated as untrusted user input. A hallucinated or maliciously crafted tag could lead the application to read arbitrary files from the filesystem.
**Prevention:** Always extract the safe basename using `Path(filename).name` and strictly verify the resolved path remains relative to the intended base directory using `is_relative_to()`. Use `.is_file()` instead of `.exists()` to prevent directory access.
## 2025-02-09 - Information Exposure via Exception Handling
**Vulnerability:** Information leakage / stack trace exposure in Streamlit UI and LLM API response due to the use of `st.exception(e)` and raw error strings like `str(e)`.
**Learning:** Returning full exception objects or raw exception strings to users can expose sensitive internal configurations, file paths, and library versions, aiding attackers in reconnaissance.
**Prevention:** Fail securely. Log internal error details to standard out or a secure logging facility, and return generic, safe error messages to the client using `st.error()` or standard string returns.
