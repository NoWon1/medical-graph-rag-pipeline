## 2024-07-22 - Avoid Alarming Colors
**Learning:** Using red for standard interactions (like focus rings) creates a false sense of urgency in medical UIs.
**Action:** Use brand colors (e.g., green #2fa36b) for safe, non-critical UI states.
## 2024-08-14 - Add Contextual Guidance to Streamlit Inputs
**Learning:** Providing `help` tooltips and `placeholder` text on interactive components like file uploaders and text areas significantly improves user intuition without cluttering the UI.
**Action:** Always include `help` and `placeholder` attributes on Streamlit input and button components to offer inline guidance.
## 2026-08-19 - Retain Focus-Visible States in Streamlit CSS
**Learning:** Default Streamlit custom CSS overrides often strip `outline` and `box-shadow` properties on `:focus`, harming keyboard accessibility.
**Action:** Always ensure explicit `:focus-visible` styles are retained or re-added for interactive elements (like buttons and textareas) using brand colors.
