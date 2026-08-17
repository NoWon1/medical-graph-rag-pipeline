## 2024-08-17 - Optimize dictionary lookups in Reciprocal Rank Fusion
**Learning:** Avoid unnecessary function calls and string allocations in tight loops (like `str(id(doc))`). Use lazy evaluation (`if did is None: did = str(id(doc))`). Also, prefer built-in functions or method references like `key=scores.get` over lambdas (`key=lambda x: scores[x]`) for sorting dictionaries by value, as it reduces lambda overhead and multiple lookups.
**Action:** Always pre-allocate or lazy-evaluate expensive computations inside loops and utilize built-in methods for performance.
