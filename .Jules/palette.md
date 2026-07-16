## 2024-07-16 - Cross-Tab Async Feedback in Streamlit
**Learning:** Streamlit's tab structure can hide active background tasks from the user if an action in one tab (e.g. uploading a file) triggers an operation whose result is rendered in another tab (e.g. chat analysis). Users might not realise they need to switch tabs to see the response.
**Action:** Always provide explicit inline feedback (like `st.info`) guiding the user to the correct tab when an action crosses tab boundaries.

## 2024-07-16 - Semantic HTML with st.markdown
**Learning:** Using generic `<div>` tags for visual layout within Streamlit's `st.markdown(..., unsafe_allow_html=True)` can impair screen reader experience.
**Action:** When injecting custom HTML for stylistic reasons in Streamlit, use semantic tags (e.g. `<h1>`, `<p>`) paired with classes to achieve the same visual outcome but with improved accessibility.
