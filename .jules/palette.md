## 2024-07-18 - Red Focus Borders Cause Confusion
**Learning:** Using red (`#e53935`) for standard input focus borders in forms/chat inputs creates false negative feedback, leading users to mistakenly believe they have triggered a validation error before typing anything.
**Action:** Always use brand primary colors (e.g., MedChat green `#2fa36b`) or standard neutral focus rings (with `box-shadow`) to distinguish focus states from error states.
