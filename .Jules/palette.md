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
