## 2024-05-24 - NumPy Array Casting for Cosine Similarity
**Learning:** In hot loops doing vector distance math (like cosine similarity or MMR reranking), converting Python lists to NumPy arrays at the point of computation creates massive overhead and serialization issues when handling responses.
**Action:** When comparing vector embeddings, leverage `numpy` arrays for vectorized calculations. Pre-cast lists to NumPy arrays before entering hot loops, and explicitly cast the result to a native Python `float()` to prevent JSON serialization errors downstream in Streamlit.
