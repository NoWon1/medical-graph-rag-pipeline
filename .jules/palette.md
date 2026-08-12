## 2024-07-22 - Avoid Alarming Colors
**Learning:** Using red for standard interactions (like focus rings) creates a false sense of urgency in medical UIs.
**Action:** Use brand colors (e.g., green #2fa36b) for safe, non-critical UI states.
## 2024-08-12 - Prevent Junk Files in PRs
**Learning:** Running Streamlit or Python code can generate `__pycache__` and `.log` files which pollute the commit and fail code review.
**Action:** Always run `git status` and use `rm -rf` or `git checkout HEAD -- <path>` to clean up auto-generated files before finalizing a PR.
