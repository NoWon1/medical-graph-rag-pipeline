## 2024-07-22 - Avoid Alarming Colors
**Learning:** Using red for standard interactions (like focus rings) creates a false sense of urgency in medical UIs.
**Action:** Use brand colors (e.g., green #2fa36b) for safe, non-critical UI states.
## 2024-08-14 - Add Contextual Guidance to Streamlit Inputs
**Learning:** Providing `help` tooltips and `placeholder` text on interactive components like file uploaders and text areas significantly improves user intuition without cluttering the UI.
**Action:** Always include `help` and `placeholder` attributes on Streamlit input and button components to offer inline guidance.
## 2024-05-24 - Streamlit CSS Accessibility
**Learning:** Default custom CSS in Streamlit often strips `outline` and `box-shadow` to hide default focus rings, which destroys keyboard accessibility. Adding `:focus-visible` with a non-alarming brand color is crucial for medical apps where red outlines may incorrectly signal critical errors.
**Action:** Always verify custom Streamlit styles include an explicit `:focus-visible` state utilizing safe brand colors (e.g. `#2fa36b`) for interactive elements.
