## 2024-05-24 - Fix XSS Vulnerability in Streamlit

**Vulnerability:** Cross-Site Scripting (XSS) via `unsafe_allow_html=True` in `st.markdown`. Dynamic user-influenced input was being interpolated directly into an HTML string.
**Learning:** `unsafe_allow_html=True` allows raw HTML to be rendered in Streamlit, which opens the door for XSS if untrusted or dynamic data is included.
**Prevention:** Always escape dynamic inputs using `html.escape()` before interpolating them into HTML strings when using `unsafe_allow_html=True`.
