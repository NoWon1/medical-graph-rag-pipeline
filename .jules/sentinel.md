## 2024-05-24 - [Path Traversal in LLM Image Rendering]
**Vulnerability:** The application was vulnerable to path traversal because it directly used the filename extracted from the LLM-generated string `[IMAGE: filename]` to construct a file path `IMAGE_DIR / filename`.
**Learning:** Never trust input from LLMs, especially when using that input to access host system resources. An LLM might be manipulated or simply hallucinate paths like `../../../etc/passwd` or absolute paths.
**Prevention:** Always sanitize and validate file paths derived from external sources, including LLM outputs. Extract only the filename component (e.g., using `Path(filename).name`) and verify that the final resolved path strictly resides within the intended directory.
