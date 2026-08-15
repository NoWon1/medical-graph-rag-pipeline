## 2024-07-22 - Avoid Alarming Colors
**Learning:** Using red for standard interactions (like focus rings) creates a false sense of urgency in medical UIs.
**Action:** Use brand colors (e.g., green #2fa36b) for safe, non-critical UI states.
## 2024-08-14 - Add Contextual Guidance to Streamlit Inputs
**Learning:** Providing `help` tooltips and `placeholder` text on interactive components like file uploaders and text areas significantly improves user intuition without cluttering the UI.
**Action:** Always include `help` and `placeholder` attributes on Streamlit input and button components to offer inline guidance.
## 2024-10-24 - Retain Focus Indicators in Custom CSS
**Learning:** Custom CSS in Streamlit that removes default focus indicators (e.g., `outline: none`, `box-shadow: none`) without providing an accessible alternative severely impacts keyboard navigation.
**Action:** Always ensure interactive elements like buttons and text areas have explicit `:focus-visible` states, using brand-consistent colors for focus rings.
