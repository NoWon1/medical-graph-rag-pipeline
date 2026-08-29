## 2024-07-22 - Avoid Alarming Colors
**Learning:** Using red for standard interactions (like focus rings) creates a false sense of urgency in medical UIs.
**Action:** Use brand colors (e.g., green #2fa36b) for safe, non-critical UI states.
## 2024-08-14 - Add Contextual Guidance to Streamlit Inputs
**Learning:** Providing `help` tooltips and `placeholder` text on interactive components like file uploaders and text areas significantly improves user intuition without cluttering the UI.
**Action:** Always include `help` and `placeholder` attributes on Streamlit input and button components to offer inline guidance.
## 2024-05-25 - Restoring Keyboard Focus in Streamlit
**Learning:** Default Streamlit custom CSS implementations often aggressively strip `outline` and `box-shadow` properties, significantly harming keyboard accessibility.
**Action:** Always ensure explicit `:focus-visible` and `:focus` states with brand-consistent colors (like green `#2fa36b`) are defined when overriding Streamlit component styles instead of hiding them.
