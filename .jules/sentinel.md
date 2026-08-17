## 2024-05-18 - Fix Prompt Injection in MedChat Graph RAG
**Vulnerability:** Prompt Injection in LLM generation (`cancer_retrieval.py`).
**Learning:** Directly concatenating user queries and patient reports within the system prompt allows an attacker to inject instructions that the LLM might execute, leading to prompt injection and potential security/safety bypasses.
**Prevention:** Wrap untrusted user inputs (like `query`, `patient_report`, and `conversation_history`) in XML tags (e.g., `<user_query>`) and provide an explicit instruction in the system prompt directing the LLM to treat the content within those tags as strictly untrusted data, not executable instructions.
