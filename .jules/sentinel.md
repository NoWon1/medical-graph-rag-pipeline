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
## 2024-05-18 - XSS via Unsafe HTML Rendering
**Vulnerability:** Use of `st.markdown(..., unsafe_allow_html=True)` without sanitizing dynamic content allows for Cross-Site Scripting (XSS).
**Learning:** Never pass raw, untrusted user or LLM data into functions that render raw HTML, as it can be used to execute malicious JavaScript.
**Prevention:** If `unsafe_allow_html=True` is required, strictly hardcode the HTML template and only interpolate pre-sanitized or known-safe values.

## 2024-05-18 - Missing Semantic Attributes in Custom HTML
**Vulnerability:** Streamlit components overridden with raw HTML (`unsafe_allow_html=True`) lacked proper ARIA attributes, causing accessibility issues.
**Learning:** When bypassing Streamlit's built-in components to use custom HTML, semantic meaning is lost. This can break screen readers and other assistive technologies.
**Prevention:** Always manually add explicit semantic `role` and `aria-*` attributes (e.g., `role="heading" aria-level="1"`) to any custom structural or dynamic HTML blocks rendered in Streamlit.
## 2025-02-09 - Accessibility vs Security Enhancements
**Learning:** Accessibility (a11y) improvements, such as adding ARIA attributes to custom Streamlit HTML (`role`, `aria-live`), do not qualify as security enhancements and will be rejected in strict security-focused code reviews.
**Prevention:** When instructed to provide a security fix or enhancement, focus exclusively on mitigations against vulnerabilities like Prompt Injections, Path Traversals, or Insecure Configurations.
## 2025-02-09 - Missing input length limits on UI inputs
**Vulnerability:** Streamlit `st.text_area` lacked character limits (`max_chars`), exposing the application to potential Denial of Service (DoS) attacks and excessive backend API costs if users pasted massive amounts of text.
**Learning:** Default Streamlit text components do not restrict input size natively. Any UI component that feeds into an LLM or database must have explicit length bounds to prevent resource exhaustion and unbounded API charges.
**Prevention:** Always explicitly define the `max_chars` parameter on Streamlit text inputs (e.g., `st.text_area`, `st.text_input`, `st.chat_input`) when building user-facing interfaces.
## 2024-05-18 - PHI/PII Leakage to External LLMs
**Vulnerability:** The application was passing untrusted, raw user input (such as `patient_report` and `query`) directly to external API endpoints (like DuckDuckGo search or Groq LLM API), exposing sensitive PHI/PII data.
**Learning:** To prevent PHI/PII leakage (HIPAA/GDPR violations) when sending user inputs to external LLM providers, the pipeline must implement a local Data Loss Prevention (DLP) layer to mask identifiers before prompts leave the network boundary.
**Prevention:** Strictly use lightweight, standard-library regex implementations (e.g., Python's `re` module) rather than heavy NLP dependencies to redact sensitive patterns (like MRN, DOB, Phone, SSN, and Email) via semantic placeholders (e.g. `[REDACTED_MRN]`) before transmission. Ensure clinical metrics are not inadvertently captured by the regex.
## 2025-02-09 - Missing Timeout Configurations on LLM APIs
**Vulnerability:** External LLM clients (like `Groq`) were instantiated without request timeouts, exposing the application to thread starvation and Denial of Service (DoS, CWE-400) if the external API hangs.
**Learning:** Default API client instantiations typically lack timeouts or rely on excessively long defaults, making backend services fragile when upstream endpoints experience latency or outages.
**Prevention:** Always explicitly set request timeouts (e.g., `timeout=15.0`) when initializing external HTTP or LLM clients in synchronous environments like Streamlit to maintain responsiveness and prevent exhaustion.
## 2025-02-09 - Image Decompression Bombs
**Vulnerability:** The application was vulnerable to Image Decompression Bombs (CWE-409) due to loading untrusted images using Pillow without explicit dimension limits.
**Learning:** Default Pillow behavior processes large images, which can lead to Out-Of-Memory (OOM) exceptions and Denial of Service (DoS) when untrusted files are uploaded or fetched.
**Prevention:** Always mitigate Image Decompression Bombs by explicitly setting `Image.MAX_IMAGE_PIXELS` (e.g., to a 16 MP limit like 4096 * 4096) before executing `Image.open()` on untrusted images.
## 2025-02-09 - Unescaped Dynamic Interpolation in Raw HTML
**Vulnerability:** Although inputs like `query_mode` originated from controlled UI states (like radio buttons or configuration dictionaries), directly interpolating them into `st.markdown(..., unsafe_allow_html=True)` without escaping violates secure coding practices and risks XSS if upstream sources are ever polluted or altered.
**Learning:** Security linters and reviewers will flag any dynamic interpolation inside `unsafe_allow_html=True` that lacks explicit escaping, regardless of where the data originates. Relying on the origin of the data being "safe" is an anti-pattern.
**Prevention:** Always apply `html.escape()` to any dynamic variable interpolated into raw HTML strings within Streamlit, even if the variable seems to come from a controlled or internal configuration map.
