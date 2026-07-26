## 2024-05-24 - Prevent Path Traversal via LLM Image Tags
**Vulnerability:** The application blindly appended a filename extracted from LLM output directly to a base directory (`img_path = IMAGE_DIR / filename`) and checked for its existence (`img_path.exists()`), allowing path traversal if the LLM generated malicious input like `../../etc/passwd`.
**Learning:** LLM outputs must always be treated as untrusted user input, even when instructed to follow specific formats.
**Prevention:** Sanitize the filename by extracting just the file component (`Path(filename).name`), explicitly resolve the path, verify it resides within the intended base directory using `is_relative_to(BASE_DIR.resolve())`, and ensure it's a file using `.is_file()`.
