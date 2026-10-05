## 2024-03-24 - Semantic Markdown Lists for Expanders
**Learning:** Rendering individual markdown elements within a loop for lists (like sources) can create disconnected, non-semantic DOM elements that screen readers struggle to navigate properly.
**Action:** Aggregate list items into a single semantic markdown string (using bullet points) and render them with a single `st.markdown()` call to ensure proper list structure and accessibility.
## 2024-05-18 - Hidden Essential Actions in Collapsed Sidebars
**Learning:** Essential and frequently used actions like 'Clear Chat' shouldn't be hidden inside default-collapsed sidebars, as this severely impacts discoverability and causes frustration.
**Action:** Always provide explicit access to core actions directly in the main view, especially in conversational UIs.
## 2024-05-19 - Top-level placement of global actions
**Learning:** Essential and frequently used actions (such as 'Clear Chat' or 'Reset') should be exposed directly in the main view to ensure high discoverability, rather than hidden inside default-collapsed sidebars. In conversational UIs, placing global actions at the top of the chat container is a better UX pattern because placing them sequentially at the bottom pushes them into the active input area as the conversation grows, disrupting the user experience.
**Action:** When adding a 'Clear Chat' action to the interface, place it at the top of the chat view instead of leaving it only in the sidebar or at the bottom.
## 2024-05-20 - Disabled states vs Disappearing UI elements
**Learning:** Hiding UI elements like a "Clear Chat" button completely when they are not applicable (e.g., when the chat is already empty) causes unnecessary layout shifts and hides the feature's existence from new users.
**Action:** Instead of conditionally hiding the element, keep it visible but disable it (`disabled=True`) and provide a dynamic tooltip explaining why it is currently unavailable to improve discoverability and provide clear feedback.
## 2024-05-21 - Mutually Exclusive Inputs
**Learning:** When multiple input methods exist for the same data (e.g., file upload vs. text paste), leaving both active while silently preferring one leads to user confusion and data loss if they spend time filling the ignored input.
**Action:** Always use disabled states (`disabled=True`) on secondary inputs when a primary input is satisfied, paired with dynamic tooltips explaining exactly how to re-enable them (e.g., "Clear the uploaded file to paste text").
## 2024-05-22 - Bidirectional Mutually Exclusive Inputs
**Learning:** In top-down frameworks like Streamlit, implementing mutually exclusive inputs (e.g., file upload vs. text paste) requires checking the session state of the second input *before* rendering the first one. Otherwise, the exclusion only works in one direction.
**Action:** When creating mutually exclusive inputs, always ensure both inputs check the other's state (using `st.session_state` keys if necessary) to disable themselves and update their help tooltips appropriately.
## 2024-05-24 - Streamlit Custom CSS and Disabled States
**Learning:** When applying custom CSS to Streamlit components (e.g., `.stButton>button`, `textarea`), explicitly including `:disabled` pseudo-class overrides is necessary. Failing to do so overrides Streamlit's native disabled visual affordances, causing disabled elements to improperly inherit active styles.
**Action:** Always explicitly include `:disabled` pseudo-class overrides when applying custom CSS to interactive components in Streamlit to preserve accessibility and proper visual cues.
## 2024-05-25 - Visibility of System Status
**Learning:** When configuration controls (like search modes) are inside a collapsible sidebar, users lose context of the active state when it is collapsed.
**Action:** Always surface the active configuration state in the main view.
## 2024-05-26 - Streamlit Chat Input CSS Specificity
**Learning:** Streamlit's chat input `textarea` requires higher specificity (`div[data-testid="stChatInput"] textarea:disabled`) to correctly style disabled states. Without this, standard `textarea:disabled` styles are overridden by Streamlit's more specific base styling for the chat input element. In addition, providing `cursor: not-allowed !important;` is crucial for accessibility.
**Action:** Always ensure high CSS specificity when styling disabled states for `stChatInput`, and remember to include the `cursor: not-allowed` indicator.
## 2024-10-02 - Visual Hierarchy for Destructive Dialog Actions
**Learning:** Applying a global custom CSS style (like `.stButton>button`) overrides Streamlit's native button type variations (like primary vs. secondary). This can result in destructive actions (like "Clear Chat") looking visually identical to safe, alternative actions (like "Cancel"), violating UX principles of visual hierarchy.
**Action:** Always map custom CSS explicitly to Streamlit's `[kind="primary"]` (destructive/core), `[kind="secondary"]` (default), and `[kind="tertiary"]` (cancel/subtle) attributes when overriding button styles, and ensure UI dialogs utilize these types appropriately (e.g., `st.button("Cancel", type="tertiary")`).
## 2024-11-20 - Visual Hierarchy for Suggested Actions
**Learning:** Using default primary/secondary button styles for contextual suggestions (like follow-up questions) creates visual noise and competes with the core interactive elements of the UI (like the chat input).
**Action:** Always style contextual suggestion buttons as `tertiary` to maintain a clean visual hierarchy while keeping them discoverable.
