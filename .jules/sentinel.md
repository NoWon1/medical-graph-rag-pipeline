## 2025-02-09 - Path Traversal in Image Rendering
**Vulnerability:** Local File Inclusion / Path Traversal vulnerability in Streamlit app where `filename` from LLM tags (`[IMAGE: filename]`) was directly appended to `IMAGE_DIR` without sanitization.
**Learning:** LLM outputs must be treated as untrusted user input. A hallucinated or maliciously crafted tag could lead the application to read arbitrary files from the filesystem.
**Prevention:** Always extract the safe basename using `Path(filename).name` and strictly verify the resolved path remains relative to the intended base directory using `is_relative_to()`. Use `.is_file()` instead of `.exists()` to prevent directory access.
## 2024-05-18 - Fix stack trace exposure in error handling
**Vulnerability:** The application was exposing detailed stack traces to the user interface via `st.exception(e)` and `st.error(f"... {e}")` in `cancer_app.py`.
**Learning:** Returning raw exception strings or stack traces directly to the UI leaks internal implementation details, which could be exploited by an attacker to understand the system architecture or identify vulnerable components.
**Prevention:** Always catch exceptions and return generic, safe error messages to the user (e.g., "An error occurred. Please check server logs."). Log the detailed exception, including the stack trace, securely to the server console or a logging system using `logging.error(..., exc_info=True)`.
## 2024-05-18 - Prevented Information Leakage in PDF Parsing
**Vulnerability:** PDF parsing errors returned raw exception strings (`str(e)`) which could be rendered by the LLM and exposed to the user, leaking internal application state and stack trace details.
**Learning:** Even if an error message is not directly rendered via `st.error()`, returning raw exceptions in data pipelines that feed into the UI or LLM prompts still constitutes an information leakage risk.
**Prevention:** Always log detailed exceptions to the server console using `logging.error(..., exc_info=True)` and return sanitized, generic error messages to any downstream function that interacts with the UI or LLM.
## 2025-02-09 - Missing input length limits on UI inputs
**Vulnerability:** Streamlit `st.text_area` and `st.chat_input` lacked character limits (`max_chars`), exposing the application to potential Denial of Service (DoS) attacks and excessive backend API costs if users pasted massive amounts of text.
**Learning:** Default Streamlit text components do not restrict input size natively. Any UI component that feeds into an LLM or database must have explicit length bounds to prevent resource exhaustion and unbounded API charges.
**Prevention:** Always explicitly define the `max_chars` parameter on Streamlit text inputs (e.g., `st.text_area`, `st.text_input`, `st.chat_input`) when building user-facing interfaces.
## 2025-02-09 - Missing length limits on file uploads
**Vulnerability:** File uploads in Streamlit lacked explicit size or character limits when read into memory (`load_report_from_upload`), exposing the application to Denial of Service (DoS) attacks and backend token exhaustion.
**Learning:** Even if manual text inputs are constrained, file upload mechanisms that feed into the same processing pipeline can bypass those constraints, allowing massive payloads to crash the server or run up API costs.
**Prevention:** Always enforce strict length limits (e.g., slicing text to a maximum character count like `text[:10000]`) on any data extracted from user-uploaded files before passing it to LLMs or downstream logic.

## 2026-09-10 - Silent Clinical Data Truncation
**Vulnerability:** Medical applications silently truncating patient reports to prevent DoS without alerting the user.
**Learning:** Silently truncating medical data can cause vital context to be lost, which may lead to incorrect answers or misdiagnosis. This represents a clinical safety risk disguised as a performance optimization.
**Prevention:** Always explicitly warn the user if clinical data is truncated or reject the input entirely, providing clear UI feedback.
## 2025-02-09 - Indirect Prompt Injection via Clinical Data
**Vulnerability:** Application parsed untrusted external text (DuckDuckGo results) and user-supplied medical reports directly into the LLM context without clear boundaries, allowing for indirect prompt injection.
**Learning:** An attacker or malicious file could contain instructions like "Ignore previous instructions" which the LLM might execute because it cannot distinguish between system instructions and user data.
**Prevention:** Always implement clear instruction boundaries (e.g., using XML tags like `<clinical_report>` and `</clinical_report>`) in the system prompt and explicitly instruct the LLM to treat anything inside those boundaries strictly as data to be analyzed, not as instructions to be executed.
## 2024-05-18 - Use SHA256 instead of MD5
**Vulnerability:** Weak MD5 hash used for generating image hashes.
**Learning:** Using MD5 is generally insecure and is flagged by linters like bandit. SHA256 should be preferred, especially for hashes that might be used for validation.
**Prevention:** Always use SHA256 or a stronger hashing algorithm instead of MD5 when hashing any arbitrary data, even if it's currently only used for internal caching.
