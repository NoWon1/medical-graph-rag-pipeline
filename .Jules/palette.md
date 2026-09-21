## 2024-03-24 - Semantic Markdown Lists for Expanders
**Learning:** Rendering individual markdown elements within a loop for lists (like sources) can create disconnected, non-semantic DOM elements that screen readers struggle to navigate properly.
**Action:** Aggregate list items into a single semantic markdown string (using bullet points) and render them with a single `st.markdown()` call to ensure proper list structure and accessibility.
