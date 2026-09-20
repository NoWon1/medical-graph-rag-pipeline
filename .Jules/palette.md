## 2024-05-17 - Streamlit Semantic List Rendering
**Learning:** Using loops to render individual list items with `st.markdown()` in Streamlit creates disconnected, non-semantic DOM elements that screen readers struggle to navigate properly.
**Action:** When rendering lists of items (like sources or links), aggregate the items into a single semantic markdown string using bullet points (`- `) and render them with a single `st.markdown()` call.
