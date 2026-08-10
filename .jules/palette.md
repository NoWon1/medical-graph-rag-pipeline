## 2024-07-22 - Avoid Alarming Colors
**Learning:** Using red for standard interactions (like focus rings) creates a false sense of urgency in medical UIs.
**Action:** Use brand colors (e.g., green #2fa36b) for safe, non-critical UI states.
## 2024-10-24 - Streamlit Contextual Guidance Pattern
**Learning:** Streamlit applications often lack built-in contextual guidance for complex inputs, which can make file uploaders and text areas less intuitive for users. Relying exclusively on labels can clutter the interface or leave out important details.
**Action:** Consistently utilize the `help` (tooltip) and `placeholder` parameters across all interactive Streamlit components (e.g., `st.file_uploader`, `st.text_area`, `st.button`) to provide clean, accessible contextual hints. Additionally, always define `:focus-visible` states using brand colors to ensure safe, expected keyboard navigation feedback.
