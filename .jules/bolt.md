## 2024-03-24 - Pre-compile Regex Outside of Loops
**Learning:** Instantiating regex inside hot loops introduces unnecessary overhead, especially if the loop is iterating thousands of times, even for small datasets.
**Action:** When using regex inside a loop, always use `re.compile(PATTERN)` before the loop and use `.search()` or `.match()` on the pre-compiled regex object inside the loop.
