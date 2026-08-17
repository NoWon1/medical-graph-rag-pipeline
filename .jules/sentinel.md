## 2024-05-18 - Path Traversal in Image Retrieval
**Vulnerability:** Path Traversal via `Path.relative_to`
**Learning:** `Path.relative_to` fails with ValueError when a path is not relative to the base directory, causing the application to fallback to using the absolute path string. This exposes absolute system paths or enables directory traversal (e.g., `../../etc/passwd`).
**Prevention:** Resolve both the file path and base directory using `.resolve()`. Validate that the resolved file path strictly belongs to the resolved base directory using `.is_relative_to()`. If an invalid path or traversal is detected, gracefully fail securely by falling back to the filename component using `.name`.
