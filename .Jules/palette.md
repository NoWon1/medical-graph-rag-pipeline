## 2024-03-24 - Semantic Markdown Lists for Expanders
**Learning:** Rendering individual markdown elements within a loop for lists (like sources) can create disconnected, non-semantic DOM elements that screen readers struggle to navigate properly.
**Action:** Aggregate list items into a single semantic markdown string (using bullet points) and render them with a single `st.markdown()` call to ensure proper list structure and accessibility.
## 2024-05-18 - Hidden Essential Actions in Collapsed Sidebars
**Learning:** Essential and frequently used actions like 'Clear Chat' shouldn't be hidden inside default-collapsed sidebars, as this severely impacts discoverability and causes frustration.
**Action:** Always provide explicit access to core actions directly in the main view, especially in conversational UIs.
