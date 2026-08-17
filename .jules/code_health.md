## 2024-08-17 - Too Many Arguments Refactoring in `_build_prompt`
**Learning:** Refactoring functions with too many arguments into using dataclasses helps reduce code smell, clean up method signatures, and provides a clear context structure that is more scalable.
**Action:** Created `PromptContext` dataclass for `_build_prompt` and refactored call sites to use this context object instead of multiple inline arguments.
