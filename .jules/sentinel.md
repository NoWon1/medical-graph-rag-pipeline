## 2025-02-27 - Cypher Injection via String Interpolation in DDL
**Vulnerability:** Directly interpolating unescaped strings into Cypher queries, especially for DDL statements (like `DROP INDEX` or `DROP CONSTRAINT` which don't support parameters).
**Learning:** Even internal variables or query results can contain malicious input or special characters. DDL statements in Neo4j don't support standard query parameters, requiring manual escaping.
**Prevention:** Always use backticks to escape identifiers (e.g., `` `identifier_name` ``) when dynamically constructing Cypher queries. Double any existing backticks inside the identifier.
