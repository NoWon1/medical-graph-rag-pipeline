## 2024-05-24 - Pure Python Math in Core Loop Bottleneck
**Learning:** Pure Python nested loops and manual mathematical operations (like `sum(a * b)` and `math.sqrt`) within core ranking algorithms like MMR create enormous computational bottlenecks.
**Action:** When implementing or optimizing algorithms that compare vector embeddings, immediately convert them to NumPy arrays to vectorize calculations. Do not rely on pure Python loops for linear algebra over many dimensions and document samples.
