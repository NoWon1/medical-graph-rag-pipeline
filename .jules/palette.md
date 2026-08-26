## 2024-07-22 - Avoid Alarming Colors
**Learning:** Using red for standard interactions (like focus rings) creates a false sense of urgency in medical UIs.
**Action:** Use brand colors (e.g., green #2fa36b) for safe, non-critical UI states.
## 2024-08-14 - Add Contextual Guidance to Streamlit Inputs
**Learning:** Providing `help` tooltips and `placeholder` text on interactive components like file uploaders and text areas significantly improves user intuition without cluttering the UI.
**Action:** Always include `help` and `placeholder` attributes on Streamlit input and button components to offer inline guidance.
## 2024-08-26 - Retain Explicit :focus-visible States
**Learning:** Default custom CSS implementations in Streamlit often aggressively strip `outline` and `box-shadow` properties (e.g., `outline: none !important;`), significantly harming keyboard accessibility by removing focus indicators.
**Action:** Always ensure explicit `:focus-visible` states are retained for interactive elements like buttons and text areas using brand-consistent colors (e.g., `outline: 2px solid #2fa36b !important; outline-offset: 2px !important;`).
