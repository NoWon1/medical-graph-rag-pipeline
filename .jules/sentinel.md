## 2024-07-23 - Path Traversal in Image Rendering
**Vulnerability:** Found a Path Traversal / LFI vulnerability in `cancer_app.py` where untrusted LLM-generated filenames (`[IMAGE: ...]`) were directly concatenated with `IMAGE_DIR` without sanitization.
**Learning:** LLM outputs must be treated as untrusted user input, especially when used in file path operations. Direct concatenation allows path traversal.
**Prevention:** Always use `pathlib.Path` to resolve the absolute path and verify it `is_relative_to()` the intended base directory. Use `is_file()` instead of `exists()` to prevent directory access.
