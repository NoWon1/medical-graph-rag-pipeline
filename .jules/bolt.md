## 2024-05-24 - Pre-casting NumPy Arrays in Nested Loops
**Learning:** When vectorizing Python loops using NumPy (e.g. for cosine similarity), casting variables to `np.array` directly inside the loop parameters implicitly forces re-allocation and conversion on every iteration. This degrades performance severely instead of improving it, negating the benefit of using NumPy's C-level math operations.
**Action:** Always extract the `np.array()` casting to happen *before* the hot loop starts, creating pre-cast arrays that get referenced dynamically inside the loops without hidden conversions.
