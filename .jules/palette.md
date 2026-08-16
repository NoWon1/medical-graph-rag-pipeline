## 2024-07-22 - Avoid Alarming Colors
**Learning:** Using red for standard interactions (like focus rings) creates a false sense of urgency in medical UIs.
**Action:** Use brand colors (e.g., green #2fa36b) for safe, non-critical UI states.
## 2024-08-14 - Add Contextual Guidance to Streamlit Inputs
**Learning:** Providing `help` tooltips and `placeholder` text on interactive components like file uploaders and text areas significantly improves user intuition without cluttering the UI.
**Action:** Always include `help` and `placeholder` attributes on Streamlit input and button components to offer inline guidance.
## 2024-05-24 - Restoring Keyboard Accessibility in Streamlit
**Learning:** Default custom CSS implementations in Streamlit often strip `outline` and `box-shadow` aggressively on elements like buttons and textareas, significantly breaking keyboard accessibility.
**Action:** Always ensure explicit `:focus-visible` states are manually added to interactive elements using brand-consistent colors (e.g., green, not red, to avoid false error signals) when applying custom themes.
