## 2024-07-22 - Avoid Alarming Colors
**Learning:** Using red for standard interactions (like focus rings) creates a false sense of urgency in medical UIs.
**Action:** Use brand colors (e.g., green #2fa36b) for safe, non-critical UI states.
## 2024-08-14 - Add Contextual Guidance to Streamlit Inputs
**Learning:** Providing `help` tooltips and `placeholder` text on interactive components like file uploaders and text areas significantly improves user intuition without cluttering the UI.
**Action:** Always include `help` and `placeholder` attributes on Streamlit input and button components to offer inline guidance.
## 2024-11-20 - Retain Explicit Focus-Visible States
**Learning:** Default Streamlit custom CSS implementations often aggressively strip `outline` and `box-shadow` properties, significantly harming keyboard accessibility.
**Action:** When styling Streamlit applications with custom CSS, always ensure explicit `:focus-visible` states are retained for interactive elements like buttons and text areas using brand-consistent colors.
## 2025-02-18 - Utilize Empty States in Tabs
**Learning:** In multi-tab interfaces, users might not know the purpose of a secondary tab. Empty states provide essential contextual guidance.
**Action:** Always provide an empty state (like `st.info`) for components that require user input (like file uploads) to explain the value of interacting with them.

## 2024-05-24 - [Smooth Interactions in Streamlit]
**Learning:** Streamlit custom CSS overrides often leave state changes (like hover, focus-visible, and active) feeling jarring because they lack default CSS transitions.
**Action:** Always include `transition: background-color 0.2s ease, box-shadow 0.2s ease, transform 0.1s ease;` and an `:active { transform: scale(0.98); }` state when overriding Streamlit component styles to maintain a polished, modern feel.
## 2025-03-01 - Add Visual Character Limits to Text Inputs
**Learning:** Streamlit text inputs lack visual bounds by default, leading to uncertainty about how much text can be pasted. Adding `max_chars` provides a helpful, auto-updating character counter in the UI.
**Action:** Always include `max_chars` on `st.text_area` and `st.chat_input` to provide clear visual constraints and improve user guidance.
## 2025-03-01 - Avoid Raw Exceptions in UI
**Learning:** Using `st.exception()` leaks internal stack traces to the UI, creating a confusing and alarming user experience during errors.
**Action:** Instead, log detailed errors safely to the server console using `logging.error(..., exc_info=True)` and provide clear, actionable fallback messages (e.g., 'Please try again') via `st.error()`.
## 2025-03-01 - Add Semantic Roles to Streamlit Custom HTML
**Learning:** Streamlit `st.markdown(..., unsafe_allow_html=True)` renders custom HTML blocks without semantic meaning, making them invisible to screen readers as headings or status updates.
**Action:** Always manually add explicit `role` and `aria-*` attributes (e.g., `role="heading" aria-level="1"`, `role="status" aria-live="polite"`) when using custom HTML for structural or dynamic UI elements.
## 2025-03-01 - Add Icebreaker Questions for Empty Chat States
**Learning:** Users often face "blank canvas paralysis" when opening a new chat interface. Providing initial suggested questions helps them understand what the system can do and lowers the barrier to interaction.
**Action:** Always provide 3-4 contextual "icebreaker" questions in the initial empty chat state to guide new users.
## 2025-03-01 - Add Cross-Tab Directional CTAs
**Learning:** When an action in one tab (like uploading a file) triggers a state change or analysis in a different tab, users often don't know where to look next without explicit guidance.
**Action:** Always provide an explicit directional CTA (e.g., using `st.info`) guiding the user to the destination tab immediately after the triggering action completes.
