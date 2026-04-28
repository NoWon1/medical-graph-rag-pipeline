# # =============================================================================
# # cancer_retrieval.py — v4 Graph RAG Retrieval
# #
# # CHANGES FROM v3:
# #   - QdrantVectorStore replaced by Neo4jVector (single DB)
# #   - Graph enrichment added after vector retrieval
# #   - Query entity extraction via spaCy (local, zero-cost)
# #   - Local/global query mode router
# #   - generate_answer() signature UNCHANGED — app.py needs zero edits
# #
# # PIPELINE (every query):
# #   Query
# #     → spaCy entity extraction (local, <50ms)
# #     → Mode router: local vs global
# #     → Dense: Neo4jVector k=20
# #     → Sparse: BM25 k=20
# #     → RRF merge → 20 candidates
# #     → MMR rerank → 8 final chunks
# #     → Graph enrichment: traverse Neo4j from chunk entities
# #     → build_context(): vector chunks + graph context block
# #     → Groq LLM → answer
# #     → Web fallback if RAG insufficient
# #
# # RETURN SIGNATURE (unchanged):
# #   generate_answer(query, patient_report, chat_history, cancer_filter)
# #   → (answer_str, sources_list)
# # =============================================================================

# from __future__ import annotations

# import os
# import re
# import json
# import math
# import time
# from pathlib import Path
# from typing import Any, List, Optional

# from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_neo4j import Neo4jVector
# from langchain_community.retrievers import BM25Retriever
# from langchain_core.documents import Document
# from langchain_core.retrievers import BaseRetriever
# from langchain_groq import ChatGroq
# from groq import Groq
# from neo4j import GraphDatabase

# try:
#     from duckduckgo_search import DDGS
#     _DDG_AVAILABLE = True
# except ImportError:
#     _DDG_AVAILABLE = False

# # spaCy for local query entity extraction
# try:
#     import spacy
#     _nlp = spacy.load("en_core_sci_sm")
#     _SPACY_AVAILABLE = True
# except Exception:
#     try:
#         import spacy
#         _nlp = spacy.load("en_core_web_sm")   # fallback to general model
#         _SPACY_AVAILABLE = True
#         print("ℹ️  scispaCy not found — using en_core_web_sm (install scispacy for better medical NER)")
#     except Exception:
#         _SPACY_AVAILABLE = False
#         _nlp = None
#         print("ℹ️  spaCy not available — query entity extraction disabled")

# from config import (
#     CHUNK_DIR, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE,
#     NEO4J_CHUNK_INDEX, NEO4J_CHUNK_LABEL,
#     NEO4J_CHUNK_TEXT_PROP, NEO4J_CHUNK_EMBEDDING_PROP,
#     EMBEDDING_MODEL, GROQ_API_KEY, GROQ_MODEL_QUERY, GROQ_TEMP_QUERY,
#     K_DENSE, K_SPARSE, K_RRF_FINAL, K_MMR_FINAL, MMR_LAMBDA, RRF_K,
#     NO_ANSWER_PHRASES, USE_GRAPH_RAG,
#     GRAPH_HOP_DEPTH, GRAPH_TOP_K_ENTITIES,
# )

# load_dotenv()

# # =============================================================================
# # EMBEDDINGS — loaded once, reused across queries
# # =============================================================================

# _embed_model: Optional[HuggingFaceEmbeddings] = None

# def get_embeddings() -> HuggingFaceEmbeddings:
#     global _embed_model
#     if _embed_model is None:
#         print("   Loading embedding model (once)...")
#         _embed_model = HuggingFaceEmbeddings(
#             model_name=EMBEDDING_MODEL,
#             model_kwargs={"device": "cpu"},
#             encode_kwargs={"normalize_embeddings": True},
#         )
#     return _embed_model

# # =============================================================================
# # STAGE 1A — DENSE RETRIEVAL (Neo4jVector)
# # =============================================================================

# def get_dense_retriever(cancer_filter: str = "") -> BaseRetriever:
#     """
#     Neo4jVector replaces QdrantVectorStore.
#     Same .as_retriever() interface — rest of pipeline unchanged.

#     cancer_filter: if set, adds a metadata pre-filter on cancer_type.
#     Note: Neo4jVector metadata filtering uses a Cypher WHERE clause.
#     """
#     kwargs: dict = {"k": K_DENSE}

#     # Apply cancer type filter if specified
#     if cancer_filter:
#         kwargs["filter"] = {"cancer_type": cancer_filter}

#     vector_store = Neo4jVector.from_existing_index(
#         embedding=get_embeddings(),
#         url=NEO4J_URI,
#         username=NEO4J_USERNAME,
#         password=NEO4J_PASSWORD,
#         database=NEO4J_DATABASE,
#         index_name=NEO4J_CHUNK_INDEX,
#         node_label=NEO4J_CHUNK_LABEL,
#         text_node_property=NEO4J_CHUNK_TEXT_PROP,
#         embedding_node_property=NEO4J_CHUNK_EMBEDDING_PROP,
#     )
#     return vector_store.as_retriever(search_kwargs=kwargs)

# # =============================================================================
# # STAGE 1B — SPARSE RETRIEVAL (BM25)
# # BM25 reads directly from chunk JSON files — unchanged from v3
# # =============================================================================

# _bm25_retriever: Optional[BM25Retriever] = None

# def get_bm25_retriever() -> BM25Retriever:
#     global _bm25_retriever
#     if _bm25_retriever is not None:
#         return _bm25_retriever

#     print("   Building BM25 index from chunk files...")
#     documents = []
#     for json_path in sorted(CHUNK_DIR.glob("*_chunks.json")):
#         with open(json_path, "r", encoding="utf-8") as f:
#             for chunk in json.load(f):
#                 documents.append(Document(
#                     page_content=chunk.get("content", ""),
#                     metadata=chunk,
#                 ))

#     if not documents:
#         raise FileNotFoundError(
#             f"No chunk files in {CHUNK_DIR}. Run cancer_ingestion.py first."
#         )

#     _bm25_retriever   = BM25Retriever.from_documents(documents)
#     _bm25_retriever.k = K_SPARSE
#     print(f"   BM25 ready: {len(documents)} chunks indexed")
#     return _bm25_retriever

# # =============================================================================
# # STAGE 1C — RRF MERGE
# # =============================================================================

# def reciprocal_rank_fusion(
#     dense_docs:  List[Document],
#     sparse_docs: List[Document],
#     k:     int = RRF_K,
#     top_n: int = K_RRF_FINAL,
# ) -> List[Document]:
#     scores:  dict[str, float]    = {}
#     doc_map: dict[str, Document] = {}

#     for rank, doc in enumerate(dense_docs):
#         did = doc.metadata.get("chunk_id", str(id(doc)))
#         scores[did]  = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
#         doc_map[did] = doc

#     for rank, doc in enumerate(sparse_docs):
#         did = doc.metadata.get("chunk_id", str(id(doc)))
#         scores[did]  = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
#         doc_map[did] = doc

#     sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
#     return [doc_map[i] for i in sorted_ids[:top_n]]

# # =============================================================================
# # STAGE 2 — MMR RERANK
# # =============================================================================

# def _cosine(v1: List[float], v2: List[float]) -> float:
#     dot   = sum(a * b for a, b in zip(v1, v2))
#     norm1 = math.sqrt(sum(a * a for a in v1))
#     norm2 = math.sqrt(sum(b * b for b in v2))
#     return 0.0 if (norm1 == 0 or norm2 == 0) else dot / (norm1 * norm2)


# def mmr_rerank(
#     query:       str,
#     candidates:  List[Document],
#     embed_model: HuggingFaceEmbeddings,
#     k:           int   = K_MMR_FINAL,
#     lambda_mult: float = MMR_LAMBDA,
# ) -> List[Document]:
#     if not candidates or len(candidates) <= k:
#         return candidates

#     query_vec = embed_model.embed_query(query)
#     doc_vecs  = embed_model.embed_documents([d.page_content for d in candidates])
#     relevance = [_cosine(v, query_vec) for v in doc_vecs]

#     selected:  List[int] = []
#     remaining: List[int] = list(range(len(candidates)))

#     for _ in range(min(k, len(candidates))):
#         if not selected:
#             best = max(remaining, key=lambda i: relevance[i])
#         else:
#             best, best_score = -1, float("-inf")
#             for idx in remaining:
#                 max_sim = max(_cosine(doc_vecs[idx], doc_vecs[s]) for s in selected)
#                 score   = lambda_mult * relevance[idx] - (1 - lambda_mult) * max_sim
#                 if score > best_score:
#                     best_score, best = score, idx
#         selected.append(best)
#         remaining.remove(best)

#     return [candidates[i] for i in selected]

# # =============================================================================
# # STAGE 3 — GRAPH ENRICHMENT (NEW)
# # Runs after MMR, adds relational context from Neo4j
# # =============================================================================

# class QueryRouter:
#     """
#     Classifies each query as 'local' or 'global' mode.

#     local:  "What does trastuzumab do for HER2+ breast cancer?"
#             → entity-anchored, graph traversal from specific nodes

#     global: "Compare treatment approaches across all cancer types"
#             → community summaries, broad overview
#     """
#     GLOBAL_SIGNALS = [
#         r"\bcompare\b", r"\bacross\b", r"\ball cancer\b", r"\boverview\b",
#         r"\bmost common\b", r"\bwhich cancer\b", r"\bsummary\b",
#         r"\bgeneral\b", r"\bbest treatment\b", r"\blatest research\b",
#         r"\bdifference between\b", r"\bsimilarities\b", r"\bversus\b",
#     ]

#     def route(self, query: str) -> str:
#         q = query.lower()
#         if any(re.search(p, q) for p in self.GLOBAL_SIGNALS):
#             return "global"
#         return "local"


# def extract_entities_from_text(text: str) -> list[str]:
#     """
#     Extract medical named entities using spaCy.
#     Falls back to capitalised phrase extraction if spaCy unavailable.
#     """
#     if _SPACY_AVAILABLE and _nlp is not None:
#         try:
#             doc = _nlp(text[:1000])   # limit for speed
#             medical_labels = {
#                 "DISEASE", "CHEMICAL", "GENE_OR_GENE_PRODUCT",
#                 "CANCER", "DRUG", "ORGANISM", "CELL_TYPE",
#                 # en_core_web_sm fallback labels
#                 "ORG", "PRODUCT", "GPE",
#             }
#             entities = [
#                 ent.text.strip()
#                 for ent in doc.ents
#                 if ent.label_ in medical_labels and len(ent.text.strip()) > 2
#             ]
#             return list(dict.fromkeys(entities))[:10]   # deduplicated, max 10
#         except Exception:
#             pass

#     # Fallback: capitalised multi-word phrases (catches drug names, gene names)
#     matches = re.findall(
#         r'\b[A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+)*\b',
#         text
#     )
#     return list(dict.fromkeys(matches))[:8]


# class GraphEnricher:
#     """
#     Enriches retrieved vector chunks with graph context from Neo4j.

#     For local queries:
#       - Extracts entities from query + patient report
#       - Traverses :MENTIONS and :RELATION edges in Neo4j
#       - Returns structured entity-relation triples as context

#     For global queries:
#       - Fetches :Community node summaries
#       - Returns high-level thematic context

#     Designed to run after MMR — adds context, doesn't replace chunks.
#     """

#     def __init__(self) -> None:
#         self.driver = GraphDatabase.driver(
#             NEO4J_URI,
#             auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
#         )
#         self.router = QueryRouter()

#     def enrich(
#         self,
#         query:          str,
#         retrieved_docs: List[Document],
#         patient_report: str = "",
#         cancer_filter:  str = "",
#     ) -> tuple[str, str]:
#         """
#         Returns:
#           graph_context_str: formatted string for LLM prompt
#           reasoning_path:    human-readable explanation of what hops fired
#         """
#         if not USE_GRAPH_RAG:
#             return "", "Graph RAG disabled"

#         mode = self.router.route(query)

#         try:
#             if mode == "global":
#                 return self._global_enrich(query, cancer_filter)
#             else:
#                 return self._local_enrich(query, retrieved_docs, patient_report, cancer_filter)
#         except Exception as e:
#             return "", f"Graph enrichment error: {str(e)[:80]}"

#     def _local_enrich(
#         self,
#         query:          str,
#         retrieved_docs: List[Document],
#         patient_report: str,
#         cancer_filter:  str,
#     ) -> tuple[str, str]:
#         """
#         Local mode: entity → graph traversal → relational facts.

#         Priority order for seed entities:
#         1. Entities from patient report (most specific)
#         2. Entities from query text
#         3. Entity names found in retrieved chunk metadata
#         """
#         # Collect seed entities
#         seed_text = f"{patient_report[:500]} {query}"
#         query_entities = extract_entities_from_text(seed_text)

#         # Also extract entities mentioned in retrieved chunk source files
#         source_files = list({
#             d.metadata.get("source_file", "")
#             for d in retrieved_docs
#             if d.metadata.get("source_file")
#         })

#         if not query_entities and not source_files:
#             return "", "local mode — no seed entities found"

#         triples:       list[str] = []
#         entities_used: list[str] = []

#         with self.driver.session(database=NEO4J_DATABASE) as session:

#             # Path A: traverse from query entities
#             for entity in query_entities[:5]:
#                 result = session.run("""
#                     MATCH (e1:Entity)
#                     WHERE toLower(e1.name) CONTAINS toLower($entity)
#                     MATCH (e1)-[r:RELATION]-(e2:Entity)
#                     WHERE $cancer_filter = '' OR e1.cancer_type = $cancer_filter
#                        OR e2.cancer_type = $cancer_filter
#                     RETURN e1.name AS src, r.type AS rel, e2.name AS tgt,
#                            e1.type AS src_type, e2.type AS tgt_type
#                     LIMIT $top_k
#                 """,
#                     entity=entity,
#                     cancer_filter=cancer_filter or "",
#                     top_k=GRAPH_TOP_K_ENTITIES,
#                 )
#                 for record in result:
#                     triple = (
#                         f"{record['src_type']}:{record['src']} "
#                         f"--[{record['rel']}]--> "
#                         f"{record['tgt_type']}:{record['tgt']}"
#                     )
#                     triples.append(triple)
#                     entities_used.append(record['src'])

#             # Path B: entities linked to retrieved chunks (2-hop)
#             chunk_ids = [
#                 d.metadata.get("chunk_id", "")
#                 for d in retrieved_docs
#                 if d.metadata.get("chunk_id")
#             ][:5]

#             if chunk_ids:
#                 result = session.run("""
#                     MATCH (c:Chunk)-[:MENTIONS]->(e1:Entity)
#                     WHERE c.chunk_id IN $chunk_ids
#                     MATCH (e1)-[r:RELATION]-(e2:Entity)
#                     RETURN e1.name AS src, r.type AS rel, e2.name AS tgt,
#                            e1.type AS src_type, e2.type AS tgt_type
#                     LIMIT 20
#                 """, chunk_ids=chunk_ids)

#                 for record in result:
#                     triple = (
#                         f"{record['src_type']}:{record['src']} "
#                         f"--[{record['rel']}]--> "
#                         f"{record['tgt_type']}:{record['tgt']}"
#                     )
#                     if triple not in triples:
#                         triples.append(triple)

#         if not triples:
#             return "", f"local mode — entities found ({query_entities[:3]}) but no graph connections"

#         # Format for LLM prompt
#         deduped_triples = list(dict.fromkeys(triples))[:20]
#         graph_ctx = (
#             "## Graph Reasoning Context\n"
#             "The following entity relationships were found by traversing "
#             "the medical knowledge graph:\n\n"
#             + "\n".join(f"  • {t}" for t in deduped_triples)
#         )

#         reasoning_path = (
#             f"local mode | seed entities: {query_entities[:3]} | "
#             f"{len(deduped_triples)} graph triples retrieved | "
#             f"{len(chunk_ids)} chunk anchors used"
#         )

#         return graph_ctx, reasoning_path

#     def _global_enrich(self, query: str, cancer_filter: str) -> tuple[str, str]:
#         """
#         Global mode: fetch community summaries for broad thematic context.
#         """
#         summaries: list[str] = []

#         with self.driver.session(database=NEO4J_DATABASE) as session:
#             cypher = """
#                 MATCH (c:Community)
#                 WHERE $cancer_filter = '' OR $cancer_filter IN c.cancer_types
#                 RETURN c.community_id AS cid, c.summary AS summary,
#                        c.cancer_types AS types, c.entity_count AS count
#                 ORDER BY c.entity_count DESC
#                 LIMIT 8
#             """
#             result = session.run(cypher, cancer_filter=cancer_filter or "")
#             for record in result:
#                 types_str = ", ".join(record["types"] or [])
#                 summaries.append(
#                     f"[{types_str} | {record['count']} entities] {record['summary']}"
#                 )

#         if not summaries:
#             return "", "global mode — no community summaries found (run graph builder)"

#         graph_ctx = (
#             "## Graph Community Context\n"
#             "The following thematic summaries from the medical knowledge graph "
#             "are relevant to this broad query:\n\n"
#             + "\n\n".join(f"  {i+1}. {s}" for i, s in enumerate(summaries))
#         )

#         reasoning_path = f"global mode | {len(summaries)} community summaries retrieved"
#         return graph_ctx, reasoning_path

#     def close(self) -> None:
#         self.driver.close()


# # Module-level enricher — initialised once, reused
# _graph_enricher: Optional[GraphEnricher] = None

# def get_graph_enricher() -> GraphEnricher:
#     global _graph_enricher
#     if _graph_enricher is None:
#         _graph_enricher = GraphEnricher()
#     return _graph_enricher

# # =============================================================================
# # HYBRID + MMR RETRIEVER (unchanged structure from v3)
# # =============================================================================

# class HybridMMRRetriever(BaseRetriever):
#     dense_ret:    Any = None
#     sparse_ret:   Any = None
#     embed_model:  Any = None
#     cancer_filter: str = ""

#     def _get_relevant_documents(self, query: str) -> List[Document]:
#         dense_docs  = self.dense_ret.invoke(query)
#         sparse_docs = self.sparse_ret.invoke(query)
#         rrf_results = reciprocal_rank_fusion(dense_docs, sparse_docs)
#         return mmr_rerank(query, rrf_results, self.embed_model)


# def get_hybrid_mmr_retriever(cancer_filter: str = "") -> HybridMMRRetriever:
#     return HybridMMRRetriever(
#         dense_ret    = get_dense_retriever(cancer_filter),
#         sparse_ret   = get_bm25_retriever(),
#         embed_model  = get_embeddings(),
#         cancer_filter = cancer_filter,
#     )

# # =============================================================================
# # CONTEXT BUILDER — now includes graph context block
# # =============================================================================

# def build_context(docs: List[Document], graph_context: str = "") -> str:
#     """
#     Build the full context string for the LLM prompt.

#     Structure:
#       [Graph Reasoning Context]   ← new section from Neo4j traversal
#       [1] Source: ... | Cancer: ... | Section: ...
#           <chunk text>
#       [2] ...
#     """
#     parts = []

#     # Graph context comes FIRST — gives LLM the relational map
#     # before it reads the individual chunks
#     if graph_context:
#         parts.append(graph_context)
#         parts.append("")  # blank separator

#     for i, doc in enumerate(docs, 1):
#         sf = (
#             doc.metadata.get("source_file")
#             or re.sub(r'_cap_\d+$|_\d{4}$', '', doc.metadata.get("chunk_id", ""))
#             or "unknown"
#         )
#         parts.append(
#             f"[{i}] Source: {sf} | "
#             f"Cancer: {doc.metadata.get('cancer_type', 'general')} | "
#             f"Section: {doc.metadata.get('section_hierarchy', 'Body')}\n"
#             f"{doc.page_content}"
#         )

#     return "\n\n".join(parts)

# # =============================================================================
# # HELPERS
# # =============================================================================

# def _rag_has_no_answer(answer: str) -> bool:
#     lower = answer.lower()
#     return any(p in lower for p in NO_ANSWER_PHRASES)

# def _build_sources(retrieved: List[Document]) -> list[dict]:
#     sources: list[dict] = []
#     seen: set = set()
#     for doc in retrieved:
#         sf = doc.metadata.get("source_file", "").strip()
#         if not sf:
#             cid = doc.metadata.get("chunk_id", "")
#             sf  = re.sub(r'_cap_\d+$|_\d{4}$', '', cid)
#         if not sf or sf in seen:
#             continue
#         seen.add(sf)
#         sources.append({
#             "label": sf,
#             "url":   doc.metadata.get("source_url", ""),
#         })
#     return sources

# # =============================================================================
# # WEB FALLBACK — unchanged from v3
# # =============================================================================

# def _duckduckgo_search(query: str, max_results: int = 5) -> list[dict]:
#     if not _DDG_AVAILABLE:
#         return []
#     try:
#         results = []
#         with DDGS() as ddgs:
#             for r in ddgs.text(f"{query} medical oncology", max_results=max_results):
#                 results.append({
#                     "title":   r.get("title", ""),
#                     "url":     r.get("href", ""),
#                     "snippet": r.get("body", ""),
#                 })
#         return results
#     except Exception as e:
#         print(f"   ⚠️  DuckDuckGo search error: {e}")
#         return []


# def _web_search_fallback(rag_answer: str, query: str, patient_report: str) -> tuple[str, list]:
#     client      = Groq(api_key=GROQ_API_KEY)
#     web_results = _duckduckgo_search(query)
#     web_sources = [{"label": r["url"], "url": r["url"]} for r in web_results if r.get("url")]

#     if web_results:
#         web_context = "\n\n".join([
#             f"[W{i+1}] {r['title']}\nURL: {r['url']}\n{r['snippet']}"
#             for i, r in enumerate(web_results)
#         ])
#         web_prompt = (
#             f"You are a medical AI assistant.\n"
#             f"PATIENT REPORT:\n{patient_report or 'No patient report provided.'}\n\n"
#             f"WEB SEARCH RESULTS:\n{web_context}\n\n"
#             f"QUESTION: {query}\n\n"
#             f"Answer using the web results. Cite [W1], [W2] etc. "
#             f"End with a disclaimer to consult a qualified doctor."
#         )
#     else:
#         web_prompt = (
#             f"You are a medical AI assistant.\n"
#             f"PATIENT REPORT:\n{patient_report or 'No patient report provided.'}\n\n"
#             f"QUESTION: {query}\n\n"
#             f"Answer clearly. End with a disclaimer to consult a qualified doctor."
#         )
#         web_sources = [
#             {"label": "https://www.cancer.gov",                "url": "https://www.cancer.gov"},
#             {"label": "https://www.ncbi.nlm.nih.gov/pubmed/", "url": "https://www.ncbi.nlm.nih.gov/pubmed/"},
#         ]

#     try:
#         resp   = client.chat.completions.create(
#             model=GROQ_MODEL_QUERY, temperature=GROQ_TEMP_QUERY,
#             messages=[{"role": "user", "content": web_prompt}],
#         )
#         web_answer = resp.choices[0].message.content or ""
#     except Exception as e:
#         web_answer = f"Could not generate web answer: {e}"

#     combined = rag_answer.strip() + "\n\n---\n\n🌐 **Web Search Result:**\n\n" + web_answer.strip()
#     return combined, web_sources

# # =============================================================================
# # STREAMING SUPPORT
# # =============================================================================

# def generate_answer_stream(
#     query:          str,
#     patient_report: str = "",
#     chat_history:   list = None,
#     cancer_filter:  str = "",
# ):
#     """
#     Streaming version of generate_answer.
#     Yields text tokens. Stores full answer + sources + followups
#     in Streamlit session_state keys used by app.py.
#     """
#     import streamlit as st

#     chat_history = chat_history or []

#     try:
#         # Run full retrieval + graph enrichment
#         retriever = get_hybrid_mmr_retriever(cancer_filter)
#         retrieved = retriever.invoke(query)

#         if not retrieved:
#             yield "No relevant information found in the clinical context."
#             st.session_state["stream_sources"]   = []
#             st.session_state["stream_followups"] = []
#             return

#         # Graph enrichment
#         graph_ctx = ""
#         reasoning_path = ""
#         if USE_GRAPH_RAG:
#             enricher = get_graph_enricher()
#             graph_ctx, reasoning_path = enricher.enrich(
#                 query=query,
#                 retrieved_docs=retrieved,
#                 patient_report=patient_report,
#                 cancer_filter=cancer_filter,
#             )

#         context_text = build_context(retrieved, graph_ctx)
#         sources      = _build_sources(retrieved)

#         # Build chat history context (last 4 turns)
#         history_text = ""
#         if chat_history:
#             recent = chat_history[-4:]
#             history_text = "\n".join([
#                 f"{m['role'].upper()}: {m['content'][:300]}"
#                 for m in recent
#             ])

#         prompt = _build_prompt(query, patient_report, context_text, history_text, reasoning_path)

#         # Stream from Groq
#         client = Groq(api_key=GROQ_API_KEY)
#         stream = client.chat.completions.create(
#             model=GROQ_MODEL_QUERY,
#             temperature=GROQ_TEMP_QUERY,
#             messages=[{"role": "user", "content": prompt}],
#             stream=True,
#         )

#         full_answer = ""
#         for chunk in stream:
#             token = chunk.choices[0].delta.content or ""
#             full_answer += token
#             yield token

#         # Generate follow-up questions
#         followups = _generate_followups(full_answer, query)

#         st.session_state["stream_buffer"]   = full_answer
#         st.session_state["stream_sources"]  = sources
#         st.session_state["stream_followups"] = followups

#     except Exception as e:
#         error_msg = f"Stream error: {str(e)}"
#         yield error_msg
#         import traceback; traceback.print_exc()


# def _build_prompt(query: str, patient_report: str, context_text: str,
#                   history_text: str, reasoning_path: str) -> str:
#     return f"""You are an empathetic medical AI assistant helping cancer patients \
# and clinicians understand medical information.

# PATIENT REPORT:
# {patient_report if patient_report else "No patient report provided."}

# CONVERSATION HISTORY:
# {history_text if history_text else "No prior conversation."}

# CLINICAL CONTEXT (from peer-reviewed medical literature and knowledge graph):
# {context_text}

# QUESTION:
# {query}

# INSTRUCTIONS:
# - Answer using ONLY the clinical context provided above.
# - Explain clearly — avoid heavy jargon where possible.
# - Cite source numbers [1], [2] etc. when referencing specific facts.
# - If the context contains a "Graph Reasoning Context" section with entity \
# relationships relevant to the question, use those relationships in your answer \
# to show how concepts connect.
# - If the clinical context contains a "Visual Assets Database" section with image \
# references relevant to the question, include them in your answer using this \
# EXACT format: [IMAGE: filename.png]
#   For example: "As shown in [IMAGE: breast-cancer-review_picture_1.png], ..."
#   Only reference images explicitly listed in the context. Never invent filenames.
# - If the answer is not in the context, clearly state you do not have enough \
# information in the clinical context.
# - End with a disclaimer advising consultation with a qualified oncologist.

# GRAPH REASONING PATH (for transparency — do not include in answer):
# {reasoning_path}
# """


# def _generate_followups(answer: str, query: str) -> list[str]:
#     """Generate 3 follow-up questions based on the answer."""
#     try:
#         client = Groq(api_key=GROQ_API_KEY)
#         prompt = (
#             f"Based on this medical question and answer, generate exactly 3 "
#             f"short follow-up questions a patient might ask next. "
#             f"Each question should be on its own line, no numbering.\n\n"
#             f"Question: {query}\n\nAnswer excerpt: {answer[:400]}"
#         )
#         resp = client.chat.completions.create(
#             model=GROQ_MODEL_QUERY,
#             temperature=0.3,
#             messages=[{"role": "user", "content": prompt}],
#         )
#         lines = (resp.choices[0].message.content or "").strip().split("\n")
#         return [l.strip() for l in lines if l.strip() and len(l.strip()) > 10][:3]
#     except Exception:
#         return []

# # =============================================================================
# # MAIN: generate_answer — PUBLIC API (signature unchanged from v3)
# # =============================================================================

# def generate_answer(
#     query:          str,
#     patient_report: str = "",
#     chat_history:   list = None,
#     cancer_filter:  str = "",
# ) -> tuple[str, list]:
#     """
#     Full Graph RAG pipeline: query → Hybrid+MMR → Graph enrichment → Groq → answer.

#     SIGNATURE UNCHANGED from v3 — app.py requires zero modifications.

#     Returns: (answer_str, sources_list)
#     Images travel inline as [IMAGE: filename.png] tokens in answer_str.
#     """
#     chat_history = chat_history or []

#     try:
#         print(f"\n🔍 [v4 Graph RAG] Query: {query[:70]}...")

#         # ── Hybrid + MMR ──────────────────────────────────────
#         retriever = get_hybrid_mmr_retriever(cancer_filter)
#         retrieved = retriever.invoke(query)

#         if not retrieved:
#             empty_rag = (
#                 "According to the provided clinical context, "
#                 "no relevant information was found for this query."
#             )
#             return _web_search_fallback(empty_rag, query, patient_report)

#         # ── Graph enrichment ──────────────────────────────────
#         graph_ctx      = ""
#         reasoning_path = "Graph RAG disabled"

#         if USE_GRAPH_RAG:
#             enricher = get_graph_enricher()
#             graph_ctx, reasoning_path = enricher.enrich(
#                 query=query,
#                 retrieved_docs=retrieved,
#                 patient_report=patient_report,
#                 cancer_filter=cancer_filter,
#             )
#             print(f"   🕸️  Graph: {reasoning_path}")

#         # ── Build context ─────────────────────────────────────
#         context_text = build_context(retrieved, graph_ctx)

#         # ── Chat history (last 4 turns) ───────────────────────
#         history_text = ""
#         if chat_history:
#             recent = chat_history[-4:]
#             history_text = "\n".join([
#                 f"{m['role'].upper()}: {m['content'][:300]}"
#                 for m in recent
#             ])

#         # ── Build sources list ────────────────────────────────
#         sources = _build_sources(retrieved)

#         # ── Groq LLM ─────────────────────────────────────────
#         prompt = _build_prompt(query, patient_report, context_text, history_text, reasoning_path)

#         client   = Groq(api_key=GROQ_API_KEY)
#         response = client.chat.completions.create(
#             model=GROQ_MODEL_QUERY,
#             temperature=GROQ_TEMP_QUERY,
#             messages=[{"role": "user", "content": prompt}],
#         )
#         answer = response.choices[0].message.content or ""

#         # ── Web fallback if insufficient ──────────────────────
#         if _rag_has_no_answer(answer):
#             print("   ⚠️  RAG insufficient → web search fallback...")
#             return _web_search_fallback(answer, query, patient_report)

#         return answer, sources

#     except Exception as e:
#         import traceback; traceback.print_exc()
#         return f"Error in retrieval pipeline: {str(e)}", []


# # =============================================================================
# # DEMO
# # =============================================================================

# if __name__ == "__main__":
#     print("=" * 62)
#     print("  cancer_retrieval.py — v4 Graph RAG")
#     print("=" * 62)

#     tests = [
#         "What topical drug delivery systems are used in breast cancer?",
#         "What is the 5-year survival rate for osteosarcoma?",
#         "Compare treatment approaches across different cancer types.",
#         "What does the PRISMA flowchart show for systematic review?",
#     ]

#     for q in tests:
#         print(f"\n{'─'*62}\n❓ {q}")
#         answer, sources = generate_answer(q)
#         print(f"📝 {answer[:300]}...")
#         has_images = "[IMAGE:" in answer
#         print(f"🖼️  Image tags: {'YES ✅' if has_images else 'NO'}")
#         print(f"📚 Sources: {[s['label'] for s in sources]}")




# # Attempt 2

# # =============================================================================
# # cancer_retrieval.py — v5  Three-Mode Graph RAG
# #
# # CHANGES FROM v4:
# #   - Removed: LLMGraphTransformer enricher, spaCy entity extraction,
# #               QueryRouter (local/global), GraphEnricher class,
# #               USE_GRAPH_RAG flag, Community node queries
# #
# #   - Added:   Three-mode routing driven by query_mode parameter
# #               RESEARCH mode  → PDF chunks only (BM25 + dense + MMR)
# #               GRAPH mode     → Hand-crafted Cypher queries first,
# #                                small vector enrichment appended
# #               AUTO mode      → Both paths merged
# #
# #   - Added:   GraphRetriever class with 6 pre-written Cypher queries
# #               against the hand-crafted medical knowledge graph
# #
# #   - Added:   detect_query_intent() — deterministic entity matching
# #               against KNOWN_CANCERS, KNOWN_CHEMO_DRUGS etc.
# #               No ML classification, no keyword guessing
# #
# #   - Preserved: BM25 + dense + RRF + MMR pipeline (unchanged)
# #   - Preserved: [IMAGE:] tag instruction in LLM prompt (unchanged)
# #   - Preserved: Web fallback with DuckDuckGo (unchanged)
# #   - Preserved: Streaming via generate_answer_stream() (unchanged)
# #   - Preserved: Follow-up question generation (unchanged)
# #
# # PUBLIC API:
# #   generate_answer(query, patient_report, chat_history,
# #                   cancer_filter, query_mode) → (answer, sources)
# #   generate_answer_stream(query, patient_report, chat_history,
# #                          cancer_filter, query_mode) → generator
# #
# #   query_mode defaults to QUERY_MODE_AUTO — existing callers work unchanged
# # =============================================================================

# from __future__ import annotations

# import re
# import json
# import math
# from typing import Any, List, Optional

# from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_neo4j import Neo4jVector
# from langchain_community.retrievers import BM25Retriever
# from langchain_core.documents import Document
# from langchain_core.retrievers import BaseRetriever
# from groq import Groq
# from neo4j import GraphDatabase

# try:
#     from duckduckgo_search import DDGS
#     _DDG_AVAILABLE = True
# except ImportError:
#     _DDG_AVAILABLE = False

# from config import (
#     CHUNK_DIR,
#     NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE,
#     NEO4J_CHUNK_INDEX, NEO4J_CHUNK_LABEL,
#     NEO4J_CHUNK_TEXT_PROP, NEO4J_CHUNK_EMBEDDING_PROP,
#     EMBEDDING_MODEL, GROQ_API_KEY, GROQ_MODEL_QUERY, GROQ_TEMP_QUERY,
#     QUERY_MODE_RESEARCH, QUERY_MODE_GRAPH, QUERY_MODE_AUTO,
#     QUERY_MODE_DEFAULT,
#     KNOWN_CANCERS, KNOWN_CHEMO_DRUGS, KNOWN_NON_CHEMO_DRUGS,
#     KNOWN_PROTOCOLS, KNOWN_EATING_EFFECTS,
#     FOOD_KEYWORDS, INTERACTION_KEYWORDS,
#     GRAPH_TOP_K_RESULTS, GRAPH_MODE_VECTOR_ENRICHMENT,
#     RESEARCH_MODE_TOP_K,
#     INTENT_CANCER_DRUGS_EFFECTS, INTENT_DRUG_INTERACTIONS,
#     INTENT_FOOD_GUIDANCE, INTENT_PROTOCOL_DETAIL,
#     INTENT_NON_CHEMO_INTERACTION, INTENT_GENERAL_GRAPH,
#     K_DENSE, K_SPARSE, K_RRF_FINAL, K_MMR_FINAL, MMR_LAMBDA, RRF_K,
#     IMAGE_TAG_PATTERN,
#     NO_ANSWER_PHRASES,
# )

# load_dotenv()

# # =============================================================================
# # EMBEDDINGS
# # =============================================================================

# _embed_model: Optional[HuggingFaceEmbeddings] = None

# def get_embeddings() -> HuggingFaceEmbeddings:
#     global _embed_model
#     if _embed_model is None:
#         print("   🔢 Loading embedding model (once)...")
#         _embed_model = HuggingFaceEmbeddings(
#             model_name=EMBEDDING_MODEL,
#             model_kwargs={"device": "cpu"},
#             encode_kwargs={"normalize_embeddings": True},
#         )
#     return _embed_model

# # =============================================================================
# # VECTOR PIPELINE — BM25 + Dense + RRF + MMR
# # Unchanged from v4 — all three modes use this for PDF chunk retrieval
# # =============================================================================

# def get_dense_retriever(cancer_filter: str = "") -> BaseRetriever:
#     kwargs: dict = {"k": K_DENSE}
#     if cancer_filter:
#         kwargs["filter"] = {"cancer_type": cancer_filter}
#     vector_store = Neo4jVector.from_existing_index(
#         embedding=get_embeddings(),
#         url=NEO4J_URI,
#         username=NEO4J_USERNAME,
#         password=NEO4J_PASSWORD,
#         database=NEO4J_DATABASE,
#         index_name=NEO4J_CHUNK_INDEX,
#         node_label=NEO4J_CHUNK_LABEL,
#         text_node_property=NEO4J_CHUNK_TEXT_PROP,
#         embedding_node_property=NEO4J_CHUNK_EMBEDDING_PROP,
#     )
#     return vector_store.as_retriever(search_kwargs=kwargs)


# _bm25_retriever: Optional[BM25Retriever] = None

# def get_bm25_retriever() -> BM25Retriever:
#     global _bm25_retriever
#     if _bm25_retriever is not None:
#         return _bm25_retriever
#     print("   📖 Building BM25 index from chunk files...")
#     documents = []
#     for json_path in sorted(CHUNK_DIR.glob("*_chunks.json")):
#         with open(json_path, "r", encoding="utf-8") as f:
#             for chunk in json.load(f):
#                 documents.append(Document(
#                     page_content=chunk.get("content", ""),
#                     metadata=chunk,
#                 ))
#     if not documents:
#         raise FileNotFoundError(
#             f"No chunk files in {CHUNK_DIR}. Run cancer_ingestion.py first."
#         )
#     _bm25_retriever   = BM25Retriever.from_documents(documents)
#     _bm25_retriever.k = K_SPARSE
#     print(f"   ✅ BM25 ready: {len(documents)} chunks indexed")
#     return _bm25_retriever


# def reciprocal_rank_fusion(
#     dense_docs:  List[Document],
#     sparse_docs: List[Document],
#     k:     int = RRF_K,
#     top_n: int = K_RRF_FINAL,
# ) -> List[Document]:
#     scores:  dict[str, float]    = {}
#     doc_map: dict[str, Document] = {}
#     for rank, doc in enumerate(dense_docs):
#         did = doc.metadata.get("chunk_id", str(id(doc)))
#         scores[did]  = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
#         doc_map[did] = doc
#     for rank, doc in enumerate(sparse_docs):
#         did = doc.metadata.get("chunk_id", str(id(doc)))
#         scores[did]  = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
#         doc_map[did] = doc
#     sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
#     return [doc_map[i] for i in sorted_ids[:top_n]]


# def _cosine(v1: List[float], v2: List[float]) -> float:
#     dot   = sum(a * b for a, b in zip(v1, v2))
#     norm1 = math.sqrt(sum(a * a for a in v1))
#     norm2 = math.sqrt(sum(b * b for b in v2))
#     return 0.0 if (norm1 == 0 or norm2 == 0) else dot / (norm1 * norm2)


# def mmr_rerank(
#     query:       str,
#     candidates:  List[Document],
#     embed_model: HuggingFaceEmbeddings,
#     k:           int   = K_MMR_FINAL,
#     lambda_mult: float = MMR_LAMBDA,
# ) -> List[Document]:
#     if not candidates or len(candidates) <= k:
#         return candidates
#     query_vec = embed_model.embed_query(query)
#     doc_vecs  = embed_model.embed_documents([d.page_content for d in candidates])
#     relevance = [_cosine(v, query_vec) for v in doc_vecs]
#     selected:  List[int] = []
#     remaining: List[int] = list(range(len(candidates)))
#     for _ in range(min(k, len(candidates))):
#         if not selected:
#             best = max(remaining, key=lambda i: relevance[i])
#         else:
#             best, best_score = -1, float("-inf")
#             for idx in remaining:
#                 max_sim = max(_cosine(doc_vecs[idx], doc_vecs[s]) for s in selected)
#                 score   = lambda_mult * relevance[idx] - (1 - lambda_mult) * max_sim
#                 if score > best_score:
#                     best_score, best = score, idx
#         selected.append(best)
#         remaining.remove(best)
#     return [candidates[i] for i in selected]


# class HybridMMRRetriever(BaseRetriever):
#     dense_ret:   Any = None
#     sparse_ret:  Any = None
#     embed_model: Any = None

#     def _get_relevant_documents(self, query: str) -> List[Document]:
#         dense_docs  = self.dense_ret.invoke(query)
#         sparse_docs = self.sparse_ret.invoke(query)
#         rrf_results = reciprocal_rank_fusion(dense_docs, sparse_docs)
#         return mmr_rerank(query, rrf_results, self.embed_model)


# def get_hybrid_mmr_retriever(cancer_filter: str = "") -> HybridMMRRetriever:
#     return HybridMMRRetriever(
#         dense_ret   = get_dense_retriever(cancer_filter),
#         sparse_ret  = get_bm25_retriever(),
#         embed_model = get_embeddings(),
#     )


# def _vector_retrieve(
#     query: str, cancer_filter: str = "", top_k: int = K_MMR_FINAL
# ) -> List[Document]:
#     """Run BM25 + dense + RRF + MMR. Returns up to top_k chunks."""
#     retriever = get_hybrid_mmr_retriever(cancer_filter)
#     return retriever.invoke(query)[:top_k]

# # =============================================================================
# # GRAPH PIPELINE
# # =============================================================================

# # ── Intent detection ──────────────────────────────────────────────────────────

# def detect_query_intent(query: str, patient_report: str = "") -> dict:
#     """
#     Deterministic intent detection using entity matching.
#     Checks query + patient report against known entity sets from config.py.
#     No ML, no keyword guessing — just exact substring matching (lowercase).

#     Priority:
#       1. NonChemoDrug + ChemoDrug/interaction signal → interaction query
#       2. Protocol name → protocol detail query
#       3. ChemoDrug + food/eating signal → food guidance query
#       4. Cancer name + chemo/food → cancer drugs and effects query
#       5. ChemoDrug alone → food guidance (most common graph question)
#       6. Cancer name alone → cancer drugs and effects
#       7. Eating effect → general graph search
#       8. Default → general graph search
#     """
#     combined = f"{query} {patient_report[:300]}".lower()

#     cancer_name   = next((c for c in KNOWN_CANCERS        if c in combined), None)
#     chemo_drug    = next((d for d in KNOWN_CHEMO_DRUGS     if d in combined), None)
#     non_chemo     = next((d for d in KNOWN_NON_CHEMO_DRUGS if d in combined), None)
#     protocol      = next((p for p in KNOWN_PROTOCOLS       if p in combined), None)
#     eating_effect = next((e for e in KNOWN_EATING_EFFECTS  if e in combined), None)

#     has_food        = any(kw in combined for kw in FOOD_KEYWORDS)
#     has_interaction = any(kw in combined for kw in INTERACTION_KEYWORDS)

#     if non_chemo and (chemo_drug or has_interaction):
#         intent = INTENT_NON_CHEMO_INTERACTION
#     elif protocol:
#         intent = INTENT_PROTOCOL_DETAIL
#     elif chemo_drug and has_food:
#         intent = INTENT_FOOD_GUIDANCE
#     elif cancer_name and (has_food or chemo_drug):
#         intent = INTENT_CANCER_DRUGS_EFFECTS
#     elif chemo_drug:
#         intent = INTENT_FOOD_GUIDANCE
#     elif cancer_name:
#         intent = INTENT_CANCER_DRUGS_EFFECTS
#     elif eating_effect:
#         intent = INTENT_GENERAL_GRAPH
#     else:
#         intent = INTENT_GENERAL_GRAPH

#     return {
#         "intent":                 intent,
#         "cancer_name":            cancer_name,
#         "chemo_drug":             chemo_drug,
#         "non_chemo_drug":         non_chemo,
#         "protocol":               protocol,
#         "eating_effect":          eating_effect,
#         "has_food_signal":        has_food,
#         "has_interaction_signal": has_interaction,
#     }


# # ── Cypher query library ──────────────────────────────────────────────────────

# CYPHER_QUERIES: dict[str, str] = {

#     INTENT_CANCER_DRUGS_EFFECTS: """
#         MATCH (c:Cancer)-[:TREATED_WITH]->(d:ChemoDrug)
#         WHERE toLower(c.name)    CONTAINS toLower($cancer_name)
#            OR toLower(c.subtype) CONTAINS toLower($cancer_name)
#         OPTIONAL MATCH (d)-[r:CAUSES_EATING_EFFECT]->(e:EatingAdverseEffect)
#         OPTIONAL MATCH (e)-[:WORSENED_BY]->(bad:FoodItem)
#         OPTIONAL MATCH (e)-[:RELIEVED_BY]->(good:FoodItem)
#         RETURN c.name                     AS cancer,
#                d.name                     AS drug,
#                d.drug_class               AS drug_class,
#                d.notes                    AS drug_notes,
#                e.name                     AS eating_effect,
#                r.severity                 AS severity,
#                e.management_tip           AS management_tip,
#                collect(DISTINCT bad.name)  AS foods_to_avoid,
#                collect(DISTINCT good.name) AS foods_to_eat
#         ORDER BY d.name, e.name
#         LIMIT $top_k
#     """,

#     INTENT_FOOD_GUIDANCE: """
#         MATCH (d:ChemoDrug)
#         WHERE toLower(d.name) CONTAINS toLower($chemo_drug)
#         OPTIONAL MATCH (d)-[r:CAUSES_EATING_EFFECT]->(e:EatingAdverseEffect)
#         OPTIONAL MATCH (e)-[:WORSENED_BY]->(avoid:FoodItem)
#         OPTIONAL MATCH (e)-[:RELIEVED_BY]->(eat:FoodItem)
#         OPTIONAL MATCH (g:NutritionGuideline)-[:REQUIRED_FOR|MANAGES]->(d)
#         OPTIONAL MATCH (g2:NutritionGuideline)-[:MANAGES]->(e)
#         RETURN d.name                          AS drug,
#                d.drug_class                    AS drug_class,
#                d.notes                         AS drug_notes,
#                e.name                          AS eating_effect,
#                r.severity                      AS severity,
#                e.management_tip                AS management_tip,
#                collect(DISTINCT avoid.name)     AS foods_to_avoid,
#                collect(DISTINCT eat.name)       AS foods_to_eat,
#                collect(DISTINCT g.text)         AS mandatory_guidelines,
#                collect(DISTINCT g2.text)        AS effect_guidelines
#         ORDER BY e.name
#         LIMIT $top_k
#     """,

#     INTENT_NON_CHEMO_INTERACTION: """
#         MATCH (n:NonChemoDrug)
#         WHERE toLower(n.name) CONTAINS toLower($non_chemo_drug)
#         OPTIONAL MATCH (n)-[:HAS_INTERACTION_WITH]->(c:ChemoDrug)
#         OPTIONAL MATCH (n)-[:DESCRIBED_BY]->(i:DrugInteraction)
#         OPTIONAL MATCH (i)-[:COMPOUNDS_EATING_EFFECT]->(e:EatingAdverseEffect)
#         OPTIONAL MATCH (n)-[:TREATS]->(a:Ailment)
#         RETURN n.name                          AS non_chemo_drug,
#                n.drug_class                    AS non_chemo_class,
#                a.name                          AS treats_ailment,
#                c.name                          AS chemo_drug,
#                i.severity                      AS severity,
#                i.description                   AS interaction_description,
#                i.clinical_action               AS recommended_action,
#                i.eating_relevance              AS eating_relevance,
#                e.name                          AS compounded_eating_effect,
#                i.mitigation                    AS mitigation
#         ORDER BY i.severity DESC
#         LIMIT $top_k
#     """,

#     INTENT_PROTOCOL_DETAIL: """
#         MATCH (p:TreatmentProtocol)
#         WHERE toLower(p.name)   CONTAINS toLower($protocol)
#            OR toLower(p.cancer) CONTAINS toLower($protocol)
#         OPTIONAL MATCH (p)-[:INCLUDES_DRUG]->(d:ChemoDrug)
#         OPTIONAL MATCH (d)-[:CAUSES_EATING_EFFECT]->(e:EatingAdverseEffect)
#         OPTIONAL MATCH (g:NutritionGuideline)-[:REQUIRED_FOR]->(d)
#         OPTIONAL MATCH (d)-[:MAY_CAUSE]->(s:SideEffect)
#         RETURN p.name                          AS protocol,
#                p.description                   AS protocol_description,
#                p.setting                       AS setting,
#                d.name                          AS drug,
#                d.drug_class                    AS drug_class,
#                d.notes                         AS drug_notes,
#                collect(DISTINCT e.name)         AS eating_effects,
#                collect(DISTINCT g.text)         AS mandatory_guidelines,
#                collect(DISTINCT s.name)         AS side_effects
#         ORDER BY d.name
#         LIMIT $top_k
#     """,

#     INTENT_DRUG_INTERACTIONS: """
#         MATCH (n:NonChemoDrug)-[:HAS_INTERACTION_WITH]->(c:ChemoDrug)
#         WHERE toLower(c.name) CONTAINS toLower($chemo_drug)
#         OPTIONAL MATCH (n)-[:DESCRIBED_BY]->(i:DrugInteraction)
#         OPTIONAL MATCH (i)-[:COMPOUNDS_EATING_EFFECT]->(e:EatingAdverseEffect)
#         OPTIONAL MATCH (n)-[:TREATS]->(a:Ailment)
#         RETURN n.name                          AS non_chemo_drug,
#                n.drug_class                    AS drug_class,
#                a.name                          AS treats_ailment,
#                i.severity                      AS severity,
#                i.description                   AS interaction_description,
#                i.clinical_action               AS recommended_action,
#                i.eating_relevance              AS eating_relevance,
#                e.name                          AS compounded_eating_effect
#         ORDER BY i.severity DESC
#         LIMIT $top_k
#     """,

#     INTENT_GENERAL_GRAPH: """
#         CALL db.index.fulltext.queryNodes('chemo_text_index', $search_text)
#         YIELD node AS drug, score
#         OPTIONAL MATCH (drug)-[:CAUSES_EATING_EFFECT]->(effect:EatingAdverseEffect)
#         OPTIONAL MATCH (effect)-[:RELIEVED_BY]->(eat:FoodItem)
#         OPTIONAL MATCH (effect)-[:WORSENED_BY]->(avoid:FoodItem)
#         RETURN drug.name                       AS drug,
#                drug.drug_class                 AS drug_class,
#                drug.mechanism                  AS mechanism,
#                score,
#                collect(DISTINCT effect.name)    AS eating_effects,
#                collect(DISTINCT eat.name)       AS foods_to_eat,
#                collect(DISTINCT avoid.name)     AS foods_to_avoid
#         ORDER BY score DESC
#         LIMIT $top_k
#     """,
# }


# # ── Graph retriever ───────────────────────────────────────────────────────────

# class GraphRetriever:
#     """
#     Runs the correct Cypher query for the detected intent.
#     Uses the hand-crafted medical knowledge graph loaded from .cypher files.
#     """

#     def __init__(self) -> None:
#         self.driver = GraphDatabase.driver(
#             NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
#         )

#     def retrieve(
#         self,
#         query:          str,
#         patient_report: str = "",
#         cancer_filter:  str = "",
#     ) -> tuple[str, str]:
#         """Returns (graph_context_str, reasoning_path_str)."""
#         intent_info = detect_query_intent(query, patient_report)
#         intent      = intent_info["intent"]

#         print(f"   🕸️  Graph intent={intent} | "
#               f"cancer={intent_info['cancer_name']} | "
#               f"chemo={intent_info['chemo_drug']} | "
#               f"non_chemo={intent_info['non_chemo_drug']}")

#         params = {
#             "cancer_name":    intent_info.get("cancer_name")    or "",
#             "chemo_drug":     intent_info.get("chemo_drug")     or "",
#             "non_chemo_drug": intent_info.get("non_chemo_drug") or "",
#             "protocol":       intent_info.get("protocol")       or "",
#             "eating_effect":  intent_info.get("eating_effect")  or "",
#             "search_text":    query[:100],
#             "top_k":          GRAPH_TOP_K_RESULTS,
#         }

#         rows = self._run_query(intent, params)

#         # Fallback to fulltext search if primary query returned nothing
#         if not rows and intent != INTENT_GENERAL_GRAPH:
#             print(f"   ⚠️  Primary graph query empty → fulltext fallback")
#             rows = self._run_query(INTENT_GENERAL_GRAPH, params)

#         if not rows:
#             return "", f"graph: no results for intent={intent}"

#         graph_context  = _format_graph_context(rows, intent)
#         reasoning_path = (
#             f"intent={intent} | rows={len(rows)} | "
#             f"cancer={intent_info['cancer_name']} | "
#             f"drug={intent_info['chemo_drug'] or intent_info['non_chemo_drug']}"
#         )
#         return graph_context, reasoning_path

#     def _run_query(self, intent: str, params: dict) -> list[dict]:
#         cypher = CYPHER_QUERIES.get(intent, "")
#         if not cypher:
#             return []
#         try:
#             with self.driver.session(database=NEO4J_DATABASE) as session:
#                 result = session.run(cypher, **params)
#                 return [dict(record) for record in result]
#         except Exception as e:
#             print(f"   ❌ Cypher error ({intent}): {str(e)[:100]}")
#             return []

#     def close(self) -> None:
#         self.driver.close()


# _graph_retriever: Optional[GraphRetriever] = None

# def get_graph_retriever() -> GraphRetriever:
#     global _graph_retriever
#     if _graph_retriever is None:
#         _graph_retriever = GraphRetriever()
#     return _graph_retriever


# def _format_graph_context(rows: list[dict], intent: str) -> str:
#     """
#     Convert Cypher result rows into structured text for the LLM prompt.
#     Graph facts appear as [G1], [G2] etc. — referenced in the prompt
#     instructions alongside [1], [2] chunk citations.
#     """
#     if not rows:
#         return ""

#     lines = [
#         "## Graph Knowledge Base",
#         "Structured medical facts retrieved from the cancer treatment "
#         "knowledge graph:\n",
#     ]

#     for i, row in enumerate(rows[:15], 1):
#         lines.append(f"[G{i}]")

#         # Drug info
#         for field, label in [
#             ("drug",           "Drug          "),
#             ("non_chemo_drug", "Non-chemo drug"),
#             ("drug_class",     "Class         "),
#             ("non_chemo_class","Drug class    "),
#             ("drug_notes",     "Clinical notes"),
#             ("mechanism",      "Mechanism     "),
#         ]:
#             if row.get(field):
#                 lines.append(f"  {label} : {row[field]}")

#         # Cancer / protocol info
#         for field, label in [
#             ("cancer",               "Cancer        "),
#             ("protocol",             "Protocol      "),
#             ("protocol_description", "Description   "),
#             ("setting",              "Setting       "),
#             ("treats_ailment",       "Treats        "),
#         ]:
#             if row.get(field):
#                 lines.append(f"  {label} : {row[field]}")

#         # Eating effects
#         if row.get("eating_effect"):
#             lines.append(f"  Eating effect  : {row['eating_effect']}")
#         effects = [e for e in (row.get("eating_effects") or []) if e]
#         if effects:
#             lines.append(f"  Eating effects : {', '.join(effects)}")
#         if row.get("severity"):
#             lines.append(f"  Severity       : {row['severity']}")
#         if row.get("management_tip"):
#             lines.append(f"  Management     : {row['management_tip']}")

#         # Food guidance
#         avoid = [f for f in (row.get("foods_to_avoid") or []) if f]
#         eat   = [f for f in (row.get("foods_to_eat")   or []) if f]
#         if avoid:
#             lines.append(f"  Foods to AVOID : {', '.join(avoid)}")
#         if eat:
#             lines.append(f"  Foods to EAT   : {', '.join(eat)}")

#         # Drug interaction info
#         for field, label in [
#             ("interaction_description", "Interaction   "),
#             ("recommended_action",      "Action needed "),
#             ("eating_relevance",        "Eating impact "),
#             ("compounded_eating_effect","Compounds     "),
#             ("mitigation",              "Mitigation    "),
#         ]:
#             if row.get(field):
#                 lines.append(f"  {label} : {row[field]}")

#         # Guidelines
#         guidelines = [g for g in (row.get("mandatory_guidelines") or []) if g]
#         guidelines += [g for g in (row.get("effect_guidelines")    or []) if g]
#         for gi, g in enumerate(guidelines[:3], 1):
#             lines.append(f"  Guideline {gi}    : {g}")

#         # Side effects
#         side_effects = [s for s in (row.get("side_effects") or []) if s]
#         if side_effects:
#             lines.append(f"  Side effects   : {', '.join(side_effects)}")

#         lines.append("")

#     return "\n".join(lines)

# # =============================================================================
# # CONTEXT BUILDER
# # =============================================================================

# def build_context(
#     vector_docs:   List[Document],
#     graph_context: str = "",
# ) -> str:
#     """
#     Combines graph structured facts (first) with PDF chunk narrative text.

#     Structure:
#       ## Graph Knowledge Base     ← [G1], [G2] ... structured facts
#         ...

#       [1] Source: ... | Cancer: ... | Section: ...
#           <chunk text — may contain [IMAGE: filename.png] tags>
#       [2] ...
#     """
#     parts = []
#     if graph_context:
#         parts.append(graph_context)

#     for i, doc in enumerate(vector_docs, 1):
#         sf = (
#             doc.metadata.get("source_file")
#             or re.sub(r'_cap_\d+$|_\d{4}$', '', doc.metadata.get("chunk_id", ""))
#             or "unknown"
#         )
#         parts.append(
#             f"[{i}] Source: {sf} | "
#             f"Cancer: {doc.metadata.get('cancer_type', 'general')} | "
#             f"Section: {doc.metadata.get('section_hierarchy', 'Body')}\n"
#             f"{doc.page_content}"
#         )

#     return "\n\n".join(parts)

# # =============================================================================
# # LLM PROMPT
# # =============================================================================

# def _build_prompt(
#     query:          str,
#     patient_report: str,
#     context_text:   str,
#     history_text:   str,
#     query_mode:     str,
#     reasoning_path: str = "",
# ) -> str:
#     mode_instruction = {
#         QUERY_MODE_RESEARCH: (
#             "You are answering from peer-reviewed clinical literature. "
#             "Cite source numbers [1], [2] etc. for statistics and claims."
#         ),
#         QUERY_MODE_GRAPH: (
#             "You are answering primarily from a structured medical knowledge graph "
#             "([G1], [G2] etc.) with clinical literature for context. "
#             "Prioritise graph facts for drug/food/interaction specifics."
#         ),
#         QUERY_MODE_AUTO: (
#             "You are answering from both a structured knowledge graph ([G1], [G2] etc.) "
#             "and peer-reviewed literature ([1], [2] etc.). "
#             "Use graph facts for drug/food/interaction specifics and "
#             "literature for clinical evidence and statistics."
#         ),
#     }.get(query_mode, "")

#     return f"""You are an empathetic medical AI assistant helping cancer patients \
# and clinicians understand medical information.

# {mode_instruction}

# PATIENT REPORT:
# {patient_report if patient_report else "No patient report provided."}

# CONVERSATION HISTORY:
# {history_text if history_text else "No prior conversation."}

# CLINICAL CONTEXT:
# {context_text}

# QUESTION:
# {query}

# INSTRUCTIONS:
# - Answer using ONLY the information in the clinical context above.
# - Explain clearly — avoid heavy jargon where possible.
# - For graph results [G1], [G2] etc.: cite them for drug interactions, \
# food guidance, and nutrition guidelines.
# - For literature results [1], [2] etc.: cite them for survival statistics, \
# trial data, and clinical evidence.
# - IMPORTANT — Image references: if the clinical context contains chunks with \
# image references, include them using this EXACT format: [IMAGE: filename.png]
#   Example: "As shown in [IMAGE: breast-cancer-review_picture_1.png], ..."
#   Only reference images explicitly listed in the context. \
# Never invent or guess image filenames.
# - If the answer is not in the context, clearly state you do not have \
# enough information.
# - End with a disclaimer advising consultation with a qualified oncologist.

# REASONING PATH (internal — do not include in answer):
# Mode={query_mode} | {reasoning_path}
# """

# # =============================================================================
# # HELPERS
# # =============================================================================

# def _rag_has_no_answer(answer: str) -> bool:
#     return any(p in answer.lower() for p in NO_ANSWER_PHRASES)


# def _build_sources(
#     vector_docs:  List[Document],
#     graph_intent: str = "",
# ) -> list[dict]:
#     sources: list[dict] = []
#     seen:    set        = set()

#     if graph_intent and graph_intent != INTENT_GENERAL_GRAPH:
#         sources.append({"label": "Cancer Treatment Knowledge Graph", "url": ""})
#         seen.add("Cancer Treatment Knowledge Graph")

#     for doc in vector_docs:
#         sf = doc.metadata.get("source_file", "").strip()
#         if not sf:
#             sf = re.sub(r'_cap_\d+$|_\d{4}$', '', doc.metadata.get("chunk_id", ""))
#         if not sf or sf in seen:
#             continue
#         seen.add(sf)
#         sources.append({"label": sf, "url": doc.metadata.get("source_url", "")})

#     return sources


# def _generate_followups(answer: str, query: str, query_mode: str) -> list[str]:
#     mode_hint = {
#         QUERY_MODE_GRAPH:    "Focus on food, nutrition, drug interactions, and side effects.",
#         QUERY_MODE_RESEARCH: "Focus on clinical evidence, survival rates, and treatment rationale.",
#         QUERY_MODE_AUTO:     "Mix of practical patient questions and clinical questions.",
#     }.get(query_mode, "")

#     try:
#         client = Groq(api_key=GROQ_API_KEY)
#         prompt = (
#             f"Based on this medical question and answer, generate exactly 3 "
#             f"short follow-up questions a cancer patient might ask next. "
#             f"{mode_hint} "
#             f"Each question on its own line, no numbering.\n\n"
#             f"Question: {query}\n\nAnswer excerpt: {answer[:400]}"
#         )
#         resp = client.chat.completions.create(
#             model=GROQ_MODEL_QUERY, temperature=0.3,
#             messages=[{"role": "user", "content": prompt}],
#         )
#         lines = (resp.choices[0].message.content or "").strip().split("\n")
#         return [l.strip() for l in lines if l.strip() and len(l.strip()) > 10][:3]
#     except Exception:
#         return []

# # =============================================================================
# # WEB FALLBACK
# # =============================================================================

# def _duckduckgo_search(query: str, max_results: int = 5) -> list[dict]:
#     if not _DDG_AVAILABLE:
#         return []
#     try:
#         results = []
#         with DDGS() as ddgs:
#             for r in ddgs.text(f"{query} medical oncology", max_results=max_results):
#                 results.append({
#                     "title":   r.get("title", ""),
#                     "url":     r.get("href", ""),
#                     "snippet": r.get("body", ""),
#                 })
#         return results
#     except Exception as e:
#         print(f"   ⚠️  DuckDuckGo search error: {e}")
#         return []


# def _web_search_fallback(
#     rag_answer: str, query: str, patient_report: str
# ) -> tuple[str, list]:
#     client      = Groq(api_key=GROQ_API_KEY)
#     web_results = _duckduckgo_search(query)
#     web_sources = [{"label": r["url"], "url": r["url"]}
#                    for r in web_results if r.get("url")]

#     if web_results:
#         web_context = "\n\n".join([
#             f"[W{i+1}] {r['title']}\nURL: {r['url']}\n{r['snippet']}"
#             for i, r in enumerate(web_results)
#         ])
#         web_prompt = (
#             f"You are a medical AI assistant.\n"
#             f"PATIENT REPORT:\n{patient_report or 'No patient report.'}\n\n"
#             f"WEB SEARCH RESULTS:\n{web_context}\n\n"
#             f"QUESTION: {query}\n\n"
#             f"Answer using the web results. Cite [W1], [W2] etc. "
#             f"End with a disclaimer to consult a qualified doctor."
#         )
#     else:
#         web_prompt = (
#             f"You are a medical AI assistant.\n"
#             f"QUESTION: {query}\n\n"
#             f"Provide a clear, accurate answer. "
#             f"End with a disclaimer to consult a qualified doctor."
#         )
#         web_sources = [
#             {"label": "https://www.cancer.gov",
#              "url":   "https://www.cancer.gov"},
#             {"label": "https://www.ncbi.nlm.nih.gov/pubmed/",
#              "url":   "https://www.ncbi.nlm.nih.gov/pubmed/"},
#         ]

#     try:
#         resp = client.chat.completions.create(
#             model=GROQ_MODEL_QUERY, temperature=GROQ_TEMP_QUERY,
#             messages=[{"role": "user", "content": web_prompt}],
#         )
#         web_answer = resp.choices[0].message.content or ""
#     except Exception as e:
#         web_answer = f"Could not generate web answer: {e}"

#     combined = (rag_answer.strip()
#                 + "\n\n---\n\n🌐 **Web Search Result:**\n\n"
#                 + web_answer.strip())
#     return combined, web_sources

# # =============================================================================
# # THREE-MODE ROUTING CORE
# # =============================================================================

# def _run_research_mode(
#     query: str, patient_report: str,
#     chat_history: list, cancer_filter: str,
# ) -> tuple[str, list, list, str]:
#     """PDF chunks only."""
#     vector_docs    = _vector_retrieve(query, cancer_filter, RESEARCH_MODE_TOP_K)
#     context_text   = build_context(vector_docs)
#     sources        = _build_sources(vector_docs)
#     reasoning_path = f"research mode | {len(vector_docs)} chunks"
#     return context_text, vector_docs, sources, reasoning_path


# def _run_graph_mode(
#     query: str, patient_report: str,
#     chat_history: list, cancer_filter: str,
# ) -> tuple[str, list, list, str]:
#     """Graph first, small vector enrichment for narrative context."""
#     graph_retriever           = get_graph_retriever()
#     graph_context, graph_path = graph_retriever.retrieve(
#         query=query, patient_report=patient_report, cancer_filter=cancer_filter,
#     )
#     vector_docs    = _vector_retrieve(query, cancer_filter, GRAPH_MODE_VECTOR_ENRICHMENT)
#     context_text   = build_context(vector_docs, graph_context)
#     intent_info    = detect_query_intent(query, patient_report)
#     sources        = _build_sources(vector_docs, intent_info["intent"])
#     reasoning_path = f"graph mode | {graph_path}"
#     return context_text, vector_docs, sources, reasoning_path


# def _run_auto_mode(
#     query: str, patient_report: str,
#     chat_history: list, cancer_filter: str,
# ) -> tuple[str, list, list, str]:
#     """Both paths. Graph fires when medical entities detected in query."""
#     combined = f"{query} {patient_report[:200]}".lower()
#     use_graph = (
#         any(kw in combined for kw in FOOD_KEYWORDS)
#         or any(kw in combined for kw in INTERACTION_KEYWORDS)
#         or any(c  in combined for c  in KNOWN_CANCERS)
#         or any(d  in combined for d  in KNOWN_CHEMO_DRUGS)
#         or any(d  in combined for d  in KNOWN_NON_CHEMO_DRUGS)
#     )

#     graph_context = ""
#     graph_path    = "graph skipped — no medical entities detected"
#     intent_used   = ""

#     if use_graph:
#         graph_retriever           = get_graph_retriever()
#         graph_context, graph_path = graph_retriever.retrieve(
#             query=query, patient_report=patient_report, cancer_filter=cancer_filter,
#         )
#         intent_used = detect_query_intent(query, patient_report)["intent"]

#     vector_docs    = _vector_retrieve(query, cancer_filter, K_MMR_FINAL)
#     context_text   = build_context(vector_docs, graph_context)
#     sources        = _build_sources(vector_docs, intent_used)
#     reasoning_path = (
#         f"auto mode | graph={'yes' if use_graph else 'no'} | {graph_path}"
#     )
#     return context_text, vector_docs, sources, reasoning_path

# # =============================================================================
# # PUBLIC API
# # =============================================================================

# def generate_answer(
#     query:          str,
#     patient_report: str  = "",
#     chat_history:   list = None,
#     cancer_filter:  str  = "",
#     query_mode:     str  = QUERY_MODE_DEFAULT,
# ) -> tuple[str, list]:
#     """
#     Main entry point. Returns (answer_str, sources_list).
#     query_mode: QUERY_MODE_RESEARCH | QUERY_MODE_GRAPH | QUERY_MODE_AUTO
#     Images travel inline as [IMAGE: filename.png] in answer_str.
#     """
#     chat_history = chat_history or []

#     try:
#         print(f"\n🔍 [v5] Query: {query[:70]}... | mode={query_mode}")

#         if query_mode == QUERY_MODE_RESEARCH:
#             context_text, vector_docs, sources, reasoning_path = _run_research_mode(
#                 query, patient_report, chat_history, cancer_filter
#             )
#         elif query_mode == QUERY_MODE_GRAPH:
#             context_text, vector_docs, sources, reasoning_path = _run_graph_mode(
#                 query, patient_report, chat_history, cancer_filter
#             )
#         else:
#             context_text, vector_docs, sources, reasoning_path = _run_auto_mode(
#                 query, patient_report, chat_history, cancer_filter
#             )

#         if not context_text.strip():
#             return _web_search_fallback(
#                 "No relevant information found.", query, patient_report
#             )

#         history_text = ""
#         if chat_history:
#             history_text = "\n".join([
#                 f"{m['role'].upper()}: {m['content'][:300]}"
#                 for m in chat_history[-4:]
#             ])

#         prompt = _build_prompt(
#             query, patient_report, context_text,
#             history_text, query_mode, reasoning_path
#         )

#         client   = Groq(api_key=GROQ_API_KEY)
#         response = client.chat.completions.create(
#             model=GROQ_MODEL_QUERY, temperature=GROQ_TEMP_QUERY,
#             messages=[{"role": "user", "content": prompt}],
#         )
#         answer = response.choices[0].message.content or ""

#         if _rag_has_no_answer(answer):
#             print("   ⚠️  Insufficient → web fallback...")
#             return _web_search_fallback(answer, query, patient_report)

#         return answer, sources

#     except Exception as e:
#         import traceback; traceback.print_exc()
#         return f"Error in retrieval pipeline: {str(e)}", []


# def generate_answer_stream(
#     query:          str,
#     patient_report: str  = "",
#     chat_history:   list = None,
#     cancer_filter:  str  = "",
#     query_mode:     str  = QUERY_MODE_DEFAULT,
# ):
#     """
#     Streaming version. Yields text tokens.
#     Stores full answer + sources + followups in Streamlit session_state.
#     """
#     import streamlit as st
#     chat_history = chat_history or []

#     try:
#         print(f"\n🔍 [v5 stream] Query: {query[:70]}... | mode={query_mode}")

#         if query_mode == QUERY_MODE_RESEARCH:
#             context_text, vector_docs, sources, reasoning_path = _run_research_mode(
#                 query, patient_report, chat_history, cancer_filter
#             )
#         elif query_mode == QUERY_MODE_GRAPH:
#             context_text, vector_docs, sources, reasoning_path = _run_graph_mode(
#                 query, patient_report, chat_history, cancer_filter
#             )
#         else:
#             context_text, vector_docs, sources, reasoning_path = _run_auto_mode(
#                 query, patient_report, chat_history, cancer_filter
#             )

#         if not context_text.strip():
#             yield "No relevant information found for this query."
#             st.session_state["stream_sources"]   = []
#             st.session_state["stream_followups"] = []
#             return

#         history_text = ""
#         if chat_history:
#             history_text = "\n".join([
#                 f"{m['role'].upper()}: {m['content'][:300]}"
#                 for m in chat_history[-4:]
#             ])

#         prompt = _build_prompt(
#             query, patient_report, context_text,
#             history_text, query_mode, reasoning_path
#         )

#         client = Groq(api_key=GROQ_API_KEY)
#         stream = client.chat.completions.create(
#             model=GROQ_MODEL_QUERY, temperature=GROQ_TEMP_QUERY,
#             messages=[{"role": "user", "content": prompt}],
#             stream=True,
#         )

#         full_answer = ""
#         for chunk in stream:
#             token = chunk.choices[0].delta.content or ""
#             full_answer += token
#             yield token

#         followups = _generate_followups(full_answer, query, query_mode)
#         st.session_state["stream_buffer"]    = full_answer
#         st.session_state["stream_sources"]   = sources
#         st.session_state["stream_followups"] = followups
#         st.session_state["stream_reasoning"] = reasoning_path

#     except Exception as e:
#         yield f"Stream error: {str(e)}"
#         import traceback; traceback.print_exc()

# # =============================================================================
# # DEMO
# # =============================================================================

# if __name__ == "__main__":
#     print("=" * 70)
#     print("  cancer_retrieval.py — v5 Three-Mode Graph RAG")
#     print("=" * 70)

#     tests = [
#         ("What eating problems does cisplatin cause and what foods to avoid?",
#          QUERY_MODE_GRAPH),
#         ("Breast cancer patient on AC-T also taking warfarin — risks?",
#          QUERY_MODE_GRAPH),
#         ("Mandatory nutritional guidelines for pemetrexed?",
#          QUERY_MODE_GRAPH),
#         ("5-year survival rate for osteosarcoma?",
#          QUERY_MODE_RESEARCH),
#         ("What does the PRISMA flowchart show?",
#          QUERY_MODE_AUTO),
#         ("Voriconazole and vincristine interaction in leukemia?",
#          QUERY_MODE_GRAPH),
#     ]

#     for q, mode in tests:
#         print(f"\n{'─'*70}")
#         print(f"❓ [{mode.upper()}] {q}")
#         answer, sources = generate_answer(q, query_mode=mode)
#         print(f"📝 {answer[:400]}...")
#         print(f"🖼️  Images : {'YES ✅' if '[IMAGE:' in answer else 'NO'}")
#         print(f"📚 Sources : {[s['label'] for s in sources]}")


# Attempt 3

# =============================================================================
# cancer_retrieval.py — v5.1  Bug-fix release
#
# FIXES FROM v5:
#
#   FIX 1 — Image chunk retrieval (Issue 1)
#     Problem: MMR was dropping image chunks because they scored lower than
#              content-rich text chunks. Images were never reaching the LLM.
#     Solution: _retrieve_image_chunks() runs a SEPARATE BM25 search that
#               ONLY looks at chunks with has_image_tags=True. These chunks
#               bypass MMR entirely and are injected into context as a
#               dedicated ## Visual References section. They can never be
#               competed out by text chunks.
#
#   FIX 2 — Image prompt instruction (Issue 1)
#     Problem: "if the context contains image references, include them"
#              gave the LLM permission to skip images. Too weak.
#     Solution: Restored original v3 "MUST" language. LLM is now explicitly
#               instructed to ALWAYS include [IMAGE:] tags when present.
#               Added a dedicated VISUAL REFERENCES INSTRUCTIONS block in the
#               prompt that is unmissable.
#
#   FIX 3 — Web fallback trigger too strict (Issue 2)
#     Problem: _rag_has_no_answer() used exact phrase matching. The LLM
#              often wrapped its "I don't know" in paragraphs of discussion,
#              so the phrase was present but buried. The osteosarcoma vaccine
#              example answered with partial information but never triggered
#              the fallback even though it explicitly said it lacked the answer.
#     Solution: Rewritten _rag_has_no_answer() with three tiers:
#               Tier 1 — Short answer (<300 chars) + any no-answer phrase → fallback
#               Tier 2 — Long answer + multiple no-answer phrases → fallback
#               Tier 3 — Proactive out-of-corpus detection: queries about
#                        vaccine approvals, FDA decisions, recent trials,
#                        drug approvals → proactive web search BEFORE LLM
#
#   FIX 4 — Web fallback answer presentation (Issue 2)
#     Problem: Fallback prepended the failed RAG answer before the web result,
#              creating a confusing "I don't know... but here's the answer"
#              response the patient had to read through.
#     Solution: When RAG genuinely has no answer (short/empty), web answer
#               stands alone. When RAG has partial information, the divider
#               is kept but clearly labelled "Additional web sources found:".
#               Web answer is no longer buried after a long "I don't know".
#
#   FIX 5 — DuckDuckGo query quality (Issue 2)
#     Problem: Search query was "{query} medical oncology" which added noise
#              for specific factual queries like vaccine approvals.
#     Solution: Query is now passed clean. The medical oncology suffix is
#               added only when the query doesn't already contain medical terms.
#
# ALL OTHER FEATURES PRESERVED EXACTLY:
#   Three-mode toggle (RESEARCH / GRAPH / AUTO)
#   BM25 + dense + RRF + MMR vector pipeline
#   GraphRetriever with 6 Cypher queries
#   detect_query_intent() entity matching
#   build_context() structure
#   _generate_followups() with mode hints
#   generate_answer() and generate_answer_stream() signatures unchanged
#   All session_state keys unchanged
# =============================================================================

from __future__ import annotations

import re
import json
import math
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from groq import Groq
from neo4j import GraphDatabase

try:
    from duckduckgo_search import DDGS
    _DDG_AVAILABLE = True
except ImportError:
    _DDG_AVAILABLE = False

from config import (
    CHUNK_DIR,
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE,
    NEO4J_CHUNK_INDEX, NEO4J_CHUNK_LABEL,
    NEO4J_CHUNK_TEXT_PROP, NEO4J_CHUNK_EMBEDDING_PROP,
    EMBEDDING_MODEL, GROQ_API_KEY, GROQ_MODEL_QUERY, GROQ_TEMP_QUERY,
    QUERY_MODE_RESEARCH, QUERY_MODE_GRAPH, QUERY_MODE_AUTO,
    QUERY_MODE_DEFAULT,
    KNOWN_CANCERS, KNOWN_CHEMO_DRUGS, KNOWN_NON_CHEMO_DRUGS,
    KNOWN_PROTOCOLS, KNOWN_EATING_EFFECTS,
    FOOD_KEYWORDS, INTERACTION_KEYWORDS,
    GRAPH_TOP_K_RESULTS, GRAPH_MODE_VECTOR_ENRICHMENT,
    RESEARCH_MODE_TOP_K,
    INTENT_CANCER_DRUGS_EFFECTS, INTENT_DRUG_INTERACTIONS,
    INTENT_FOOD_GUIDANCE, INTENT_PROTOCOL_DETAIL,
    INTENT_NON_CHEMO_INTERACTION, INTENT_GENERAL_GRAPH,
    K_DENSE, K_SPARSE, K_RRF_FINAL, K_MMR_FINAL, MMR_LAMBDA, RRF_K,
    IMAGE_TAG_PATTERN,
    NO_ANSWER_PHRASES,
)

load_dotenv()

# =============================================================================
# EMBEDDINGS
# =============================================================================

_embed_model: Optional[HuggingFaceEmbeddings] = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embed_model
    if _embed_model is None:
        print("   🔢 Loading embedding model (once)...")
        _embed_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embed_model

# =============================================================================
# VECTOR PIPELINE — BM25 + Dense + RRF + MMR
# Unchanged from v5
# =============================================================================

# def get_dense_retriever(cancer_filter: str = "") -> BaseRetriever:
#     kwargs: dict = {"k": K_DENSE}
#     if cancer_filter:
#         kwargs["filter"] = {"cancer_type": cancer_filter}
#     vector_store = Neo4jVector.from_existing_index(
#         embedding=get_embeddings(),
#         url=NEO4J_URI,
#         username=NEO4J_USERNAME,
#         password=NEO4J_PASSWORD,
#         database=NEO4J_DATABASE,
#         index_name=NEO4J_CHUNK_INDEX,
#         node_label=NEO4J_CHUNK_LABEL,
#         text_node_property=NEO4J_CHUNK_TEXT_PROP,
#         embedding_node_property=NEO4J_CHUNK_EMBEDDING_PROP,
#     )
#     return vector_store.as_retriever(search_kwargs=kwargs)

# =============================================================================
# Add this global cache dictionary right above the function
_VECTOR_STORE_CACHE = {}

def get_dense_retriever(cancer_filter: str = "") -> BaseRetriever:
    global _VECTOR_STORE_CACHE
    
    kwargs: dict = {"k": K_DENSE}
    if cancer_filter:
        kwargs["filter"] = {"cancer_type": cancer_filter}
        
    # Check if we already have an active Neo4j connection in the cache
    if "neo4j_store" not in _VECTOR_STORE_CACHE:
        # If not, create it and store it (this only happens once!)
        _VECTOR_STORE_CACHE["neo4j_store"] = Neo4jVector.from_existing_index(
            embedding=get_embeddings(),
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            index_name=NEO4J_CHUNK_INDEX,
            node_label=NEO4J_CHUNK_LABEL,
            text_node_property=NEO4J_CHUNK_TEXT_PROP,
            embedding_node_property=NEO4J_CHUNK_EMBEDDING_PROP,
        )
        
    # Retrieve the active connection from the cache
    vector_store = _VECTOR_STORE_CACHE["neo4j_store"]
    
    return vector_store.as_retriever(search_kwargs=kwargs)


_bm25_retriever:       Optional[BM25Retriever] = None
_image_bm25_retriever: Optional[BM25Retriever] = None   # FIX 1: dedicated image index

def get_bm25_retriever() -> BM25Retriever:
    """Full-corpus BM25 for main content retrieval."""
    global _bm25_retriever
    if _bm25_retriever is not None:
        return _bm25_retriever
    print("   📖 Building BM25 index from chunk files...")
    documents = []
    for json_path in sorted(CHUNK_DIR.glob("*_chunks.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            for chunk in json.load(f):
                documents.append(Document(
                    page_content=chunk.get("content", ""),
                    metadata=chunk,
                ))
    if not documents:
        raise FileNotFoundError(
            f"No chunk files in {CHUNK_DIR}. Run cancer_ingestion.py first."
        )
    _bm25_retriever   = BM25Retriever.from_documents(documents)
    _bm25_retriever.k = K_SPARSE
    print(f"   ✅ BM25 ready: {len(documents)} chunks indexed")
    return _bm25_retriever


def get_image_bm25_retriever() -> Optional[BM25Retriever]:
    """
    FIX 1 — Dedicated BM25 index containing ONLY image-tagged chunks.
    Built once from chunks where has_image_tags=True or content
    contains '[IMAGE:'. These chunks are never mixed with content
    chunks so MMR cannot drop them.
    """
    global _image_bm25_retriever
    if _image_bm25_retriever is not None:
        return _image_bm25_retriever

    image_docs = []
    for json_path in sorted(CHUNK_DIR.glob("*_chunks.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            for chunk in json.load(f):
                content = chunk.get("content", "")
                has_tag = (
                    chunk.get("has_image_tags", False)
                    or "[IMAGE:" in content.upper()
                )
                if has_tag:
                    image_docs.append(Document(
                        page_content=content,
                        metadata=chunk,
                    ))

    if not image_docs:
        print("   ℹ️  No image-tagged chunks found — image retrieval disabled")
        return None

    _image_bm25_retriever   = BM25Retriever.from_documents(image_docs)
    _image_bm25_retriever.k = 3   # always return top 3 image chunks
    print(f"   🖼️  Image BM25 ready: {len(image_docs)} image-tagged chunks")
    return _image_bm25_retriever


def reciprocal_rank_fusion(
    dense_docs:  List[Document],
    sparse_docs: List[Document],
    k:     int = RRF_K,
    top_n: int = K_RRF_FINAL,
) -> List[Document]:
    scores:  dict[str, float]    = {}
    doc_map: dict[str, Document] = {}
    for rank, doc in enumerate(dense_docs):
        did = doc.metadata.get("chunk_id", str(id(doc)))
        scores[did]  = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
        doc_map[did] = doc
    for rank, doc in enumerate(sparse_docs):
        did = doc.metadata.get("chunk_id", str(id(doc)))
        scores[did]  = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
        doc_map[did] = doc
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[i] for i in sorted_ids[:top_n]]


def _cosine(v1: List[float], v2: List[float]) -> float:
    dot   = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return 0.0 if (norm1 == 0 or norm2 == 0) else dot / (norm1 * norm2)


def mmr_rerank(
    query:       str,
    candidates:  List[Document],
    embed_model: HuggingFaceEmbeddings,
    k:           int   = K_MMR_FINAL,
    lambda_mult: float = MMR_LAMBDA,
) -> List[Document]:
    if not candidates or len(candidates) <= k:
        return candidates
    query_vec = embed_model.embed_query(query)
    doc_vecs  = embed_model.embed_documents([d.page_content for d in candidates])
    relevance = [_cosine(v, query_vec) for v in doc_vecs]
    selected:  List[int] = []
    remaining: List[int] = list(range(len(candidates)))
    for _ in range(min(k, len(candidates))):
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            best, best_score = -1, float("-inf")
            for idx in remaining:
                max_sim = max(_cosine(doc_vecs[idx], doc_vecs[s]) for s in selected)
                score   = lambda_mult * relevance[idx] - (1 - lambda_mult) * max_sim
                if score > best_score:
                    best_score, best = score, idx
        selected.append(best)
        remaining.remove(best)
    return [candidates[i] for i in selected]


class HybridMMRRetriever(BaseRetriever):
    dense_ret:   Any = None
    sparse_ret:  Any = None
    embed_model: Any = None

    def _get_relevant_documents(self, query: str) -> List[Document]:
        dense_docs  = self.dense_ret.invoke(query)
        sparse_docs = self.sparse_ret.invoke(query)
        rrf_results = reciprocal_rank_fusion(dense_docs, sparse_docs)
        return mmr_rerank(query, rrf_results, self.embed_model)


def get_hybrid_mmr_retriever(cancer_filter: str = "") -> HybridMMRRetriever:
    return HybridMMRRetriever(
        dense_ret   = get_dense_retriever(cancer_filter),
        sparse_ret  = get_bm25_retriever(),
        embed_model = get_embeddings(),
    )


def _vector_retrieve(
    query: str, cancer_filter: str = "", top_k: int = K_MMR_FINAL
) -> List[Document]:
    """Run BM25 + dense + RRF + MMR. Returns top content chunks."""
    retriever = get_hybrid_mmr_retriever(cancer_filter)
    return retriever.invoke(query)[:top_k]


def _retrieve_image_chunks(query: str) -> List[Document]:
    """
    FIX 1 — Dedicated image chunk retrieval.

    Runs a separate BM25 search over image-only chunks.
    These chunks NEVER go through MMR so they cannot be dropped.

    Logic:
      1. If query contains figure/chart/image/table/flowchart keywords
         → always retrieve image chunks regardless of query topic
      2. Otherwise → retrieve image chunks matching the query topic
         (e.g. asking about osteosarcoma survival → get survival curve images)

    Returns up to 2 image chunks to avoid bloating the LLM context.
    """
    img_retriever = get_image_bm25_retriever()
    if img_retriever is None:
        return []

    try:
        # Boost the search if query explicitly mentions visual content
        visual_keywords = {
            "figure", "fig", "table", "chart", "graph", "image",
            "flowchart", "diagram", "show", "illustrat", "display",
            "kaplan", "survival curve", "forest plot", "prisma",
        }
        query_lower = query.lower()
        has_visual_intent = any(kw in query_lower for kw in visual_keywords)

        # Always search — if visual intent detected, weight the query
        # towards visual terms; otherwise use the raw query
        search_query = query
        if not has_visual_intent:
            # Add the primary subject to find relevant images
            # even when the user didn't explicitly ask for visuals
            search_query = query

        results = img_retriever.invoke(search_query)

        # Filter: only return image chunks actually relevant to the query
        # (BM25 can return false positives on short queries)
        # If the image chunk source file contains any cancer/drug terms
        # from the query, it is relevant
        query_tokens = set(query_lower.split())
        filtered = []
        for doc in results:
            content_lower = doc.page_content.lower()
            # Always include if the chunk content overlaps with query tokens
            overlap = sum(1 for t in query_tokens if len(t) > 4 and t in content_lower)
            if overlap >= 1 or has_visual_intent:
                filtered.append(doc)

        return filtered[:2]   # cap at 2 image chunks per query

    except Exception as e:
        print(f"   ⚠️  Image chunk retrieval error: {e}")
        return []

# =============================================================================
# GRAPH PIPELINE — unchanged from v5
# =============================================================================

def detect_query_intent(query: str, patient_report: str = "") -> dict:
    """Deterministic entity matching. Unchanged from v5."""
    combined = f"{query} {patient_report[:300]}".lower()

    cancer_name   = next((c for c in KNOWN_CANCERS        if c in combined), None)
    chemo_drug    = next((d for d in KNOWN_CHEMO_DRUGS     if d in combined), None)
    non_chemo     = next((d for d in KNOWN_NON_CHEMO_DRUGS if d in combined), None)
    protocol      = next((p for p in KNOWN_PROTOCOLS       if p in combined), None)
    eating_effect = next((e for e in KNOWN_EATING_EFFECTS  if e in combined), None)

    has_food        = any(kw in combined for kw in FOOD_KEYWORDS)
    has_interaction = any(kw in combined for kw in INTERACTION_KEYWORDS)

    if non_chemo and (chemo_drug or has_interaction):
        intent = INTENT_NON_CHEMO_INTERACTION
    elif protocol:
        intent = INTENT_PROTOCOL_DETAIL
    elif chemo_drug and has_food:
        intent = INTENT_FOOD_GUIDANCE
    elif cancer_name and (has_food or chemo_drug):
        intent = INTENT_CANCER_DRUGS_EFFECTS
    elif chemo_drug:
        intent = INTENT_FOOD_GUIDANCE
    elif cancer_name:
        intent = INTENT_CANCER_DRUGS_EFFECTS
    elif eating_effect:
        intent = INTENT_GENERAL_GRAPH
    else:
        intent = INTENT_GENERAL_GRAPH

    return {
        "intent":                 intent,
        "cancer_name":            cancer_name,
        "chemo_drug":             chemo_drug,
        "non_chemo_drug":         non_chemo,
        "protocol":               protocol,
        "eating_effect":          eating_effect,
        "has_food_signal":        has_food,
        "has_interaction_signal": has_interaction,
    }


CYPHER_QUERIES: dict[str, str] = {

    INTENT_CANCER_DRUGS_EFFECTS: """
        MATCH (c:Cancer)-[:TREATED_WITH]->(d:ChemoDrug)
        WHERE toLower(c.name)    CONTAINS toLower($cancer_name)
           OR toLower(c.subtype) CONTAINS toLower($cancer_name)
        OPTIONAL MATCH (d)-[r:CAUSES_EATING_EFFECT]->(e:EatingAdverseEffect)
        OPTIONAL MATCH (e)-[:WORSENED_BY]->(bad:FoodItem)
        OPTIONAL MATCH (e)-[:RELIEVED_BY]->(good:FoodItem)
        RETURN c.name                     AS cancer,
               d.name                     AS drug,
               d.drug_class               AS drug_class,
               d.notes                    AS drug_notes,
               e.name                     AS eating_effect,
               r.severity                 AS severity,
               e.management_tip           AS management_tip,
               collect(DISTINCT bad.name)  AS foods_to_avoid,
               collect(DISTINCT good.name) AS foods_to_eat
        ORDER BY d.name, e.name
        LIMIT $top_k
    """,

    INTENT_FOOD_GUIDANCE: """
        MATCH (d:ChemoDrug)
        WHERE toLower(d.name) CONTAINS toLower($chemo_drug)
        OPTIONAL MATCH (d)-[r:CAUSES_EATING_EFFECT]->(e:EatingAdverseEffect)
        OPTIONAL MATCH (e)-[:WORSENED_BY]->(avoid:FoodItem)
        OPTIONAL MATCH (e)-[:RELIEVED_BY]->(eat:FoodItem)
        OPTIONAL MATCH (g:NutritionGuideline)-[:REQUIRED_FOR|MANAGES]->(d)
        OPTIONAL MATCH (g2:NutritionGuideline)-[:MANAGES]->(e)
        RETURN d.name                          AS drug,
               d.drug_class                    AS drug_class,
               d.notes                         AS drug_notes,
               e.name                          AS eating_effect,
               r.severity                      AS severity,
               e.management_tip                AS management_tip,
               collect(DISTINCT avoid.name)     AS foods_to_avoid,
               collect(DISTINCT eat.name)       AS foods_to_eat,
               collect(DISTINCT g.text)         AS mandatory_guidelines,
               collect(DISTINCT g2.text)        AS effect_guidelines
        ORDER BY e.name
        LIMIT $top_k
    """,

    INTENT_NON_CHEMO_INTERACTION: """
        MATCH (n:NonChemoDrug)
        WHERE toLower(n.name) CONTAINS toLower($non_chemo_drug)
        OPTIONAL MATCH (n)-[:HAS_INTERACTION_WITH]->(c:ChemoDrug)
        OPTIONAL MATCH (n)-[:DESCRIBED_BY]->(i:DrugInteraction)
        OPTIONAL MATCH (i)-[:COMPOUNDS_EATING_EFFECT]->(e:EatingAdverseEffect)
        OPTIONAL MATCH (n)-[:TREATS]->(a:Ailment)
        RETURN n.name                          AS non_chemo_drug,
               n.drug_class                    AS non_chemo_class,
               a.name                          AS treats_ailment,
               c.name                          AS chemo_drug,
               i.severity                      AS severity,
               i.description                   AS interaction_description,
               i.clinical_action               AS recommended_action,
               i.eating_relevance              AS eating_relevance,
               e.name                          AS compounded_eating_effect,
               i.mitigation                    AS mitigation
        ORDER BY i.severity DESC
        LIMIT $top_k
    """,

    INTENT_PROTOCOL_DETAIL: """
        MATCH (p:TreatmentProtocol)
        WHERE toLower(p.name)   CONTAINS toLower($protocol)
           OR toLower(p.cancer) CONTAINS toLower($protocol)
        OPTIONAL MATCH (p)-[:INCLUDES_DRUG]->(d:ChemoDrug)
        OPTIONAL MATCH (d)-[:CAUSES_EATING_EFFECT]->(e:EatingAdverseEffect)
        OPTIONAL MATCH (g:NutritionGuideline)-[:REQUIRED_FOR]->(d)
        OPTIONAL MATCH (d)-[:MAY_CAUSE]->(s:SideEffect)
        RETURN p.name                          AS protocol,
               p.description                   AS protocol_description,
               p.setting                       AS setting,
               d.name                          AS drug,
               d.drug_class                    AS drug_class,
               d.notes                         AS drug_notes,
               collect(DISTINCT e.name)         AS eating_effects,
               collect(DISTINCT g.text)         AS mandatory_guidelines,
               collect(DISTINCT s.name)         AS side_effects
        ORDER BY d.name
        LIMIT $top_k
    """,

    INTENT_DRUG_INTERACTIONS: """
        MATCH (n:NonChemoDrug)-[:HAS_INTERACTION_WITH]->(c:ChemoDrug)
        WHERE toLower(c.name) CONTAINS toLower($chemo_drug)
        OPTIONAL MATCH (n)-[:DESCRIBED_BY]->(i:DrugInteraction)
        OPTIONAL MATCH (i)-[:COMPOUNDS_EATING_EFFECT]->(e:EatingAdverseEffect)
        OPTIONAL MATCH (n)-[:TREATS]->(a:Ailment)
        RETURN n.name                          AS non_chemo_drug,
               n.drug_class                    AS drug_class,
               a.name                          AS treats_ailment,
               i.severity                      AS severity,
               i.description                   AS interaction_description,
               i.clinical_action               AS recommended_action,
               i.eating_relevance              AS eating_relevance,
               e.name                          AS compounded_eating_effect
        ORDER BY i.severity DESC
        LIMIT $top_k
    """,

    INTENT_GENERAL_GRAPH: """
        CALL db.index.fulltext.queryNodes('chemo_text_index', $search_text)
        YIELD node AS drug, score
        OPTIONAL MATCH (drug)-[:CAUSES_EATING_EFFECT]->(effect:EatingAdverseEffect)
        OPTIONAL MATCH (effect)-[:RELIEVED_BY]->(eat:FoodItem)
        OPTIONAL MATCH (effect)-[:WORSENED_BY]->(avoid:FoodItem)
        RETURN drug.name                       AS drug,
               drug.drug_class                 AS drug_class,
               drug.mechanism                  AS mechanism,
               score,
               collect(DISTINCT effect.name)    AS eating_effects,
               collect(DISTINCT eat.name)       AS foods_to_eat,
               collect(DISTINCT avoid.name)     AS foods_to_avoid
        ORDER BY score DESC
        LIMIT $top_k
    """,
}


class GraphRetriever:
    """Runs Cypher queries against hand-crafted knowledge graph. Unchanged from v5."""

    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )

    def retrieve(
        self,
        query:          str,
        patient_report: str = "",
        cancer_filter:  str = "",
    ) -> tuple[str, str]:
        intent_info = detect_query_intent(query, patient_report)
        intent      = intent_info["intent"]

        print(f"   🕸️  Graph intent={intent} | "
              f"cancer={intent_info['cancer_name']} | "
              f"chemo={intent_info['chemo_drug']} | "
              f"non_chemo={intent_info['non_chemo_drug']}")

        params = {
            "cancer_name":    intent_info.get("cancer_name")    or "",
            "chemo_drug":     intent_info.get("chemo_drug")     or "",
            "non_chemo_drug": intent_info.get("non_chemo_drug") or "",
            "protocol":       intent_info.get("protocol")       or "",
            "eating_effect":  intent_info.get("eating_effect")  or "",
            "search_text":    query[:100],
            "top_k":          GRAPH_TOP_K_RESULTS,
        }

        rows = self._run_query(intent, params)

        if not rows and intent != INTENT_GENERAL_GRAPH:
            print(f"   ⚠️  Primary graph query empty → fulltext fallback")
            rows = self._run_query(INTENT_GENERAL_GRAPH, params)

        if not rows:
            return "", f"graph: no results for intent={intent}"

        graph_context  = _format_graph_context(rows, intent)
        reasoning_path = (
            f"intent={intent} | rows={len(rows)} | "
            f"cancer={intent_info['cancer_name']} | "
            f"drug={intent_info['chemo_drug'] or intent_info['non_chemo_drug']}"
        )
        return graph_context, reasoning_path

    def _run_query(self, intent: str, params: dict) -> list[dict]:
        cypher = CYPHER_QUERIES.get(intent, "")
        if not cypher:
            return []
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(cypher, **params)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"   ❌ Cypher error ({intent}): {str(e)[:100]}")
            return []

    def close(self) -> None:
        self.driver.close()


_graph_retriever: Optional[GraphRetriever] = None

def get_graph_retriever() -> GraphRetriever:
    global _graph_retriever
    if _graph_retriever is None:
        _graph_retriever = GraphRetriever()
    return _graph_retriever


def _format_graph_context(rows: list[dict], intent: str) -> str:
    """Convert Cypher rows to [G1], [G2] structured text. Unchanged from v5."""
    if not rows:
        return ""

    lines = [
        "## Graph Knowledge Base",
        "Structured medical facts from the cancer treatment knowledge graph:\n",
    ]

    for i, row in enumerate(rows[:15], 1):
        lines.append(f"[G{i}]")

        for field, label in [
            ("drug",           "Drug          "),
            ("non_chemo_drug", "Non-chemo drug"),
            ("drug_class",     "Class         "),
            ("non_chemo_class","Drug class    "),
            ("drug_notes",     "Clinical notes"),
            ("mechanism",      "Mechanism     "),
        ]:
            if row.get(field):
                lines.append(f"  {label} : {row[field]}")

        for field, label in [
            ("cancer",               "Cancer        "),
            ("protocol",             "Protocol      "),
            ("protocol_description", "Description   "),
            ("setting",              "Setting       "),
            ("treats_ailment",       "Treats        "),
        ]:
            if row.get(field):
                lines.append(f"  {label} : {row[field]}")

        if row.get("eating_effect"):
            lines.append(f"  Eating effect  : {row['eating_effect']}")
        effects = [e for e in (row.get("eating_effects") or []) if e]
        if effects:
            lines.append(f"  Eating effects : {', '.join(effects)}")
        if row.get("severity"):
            lines.append(f"  Severity       : {row['severity']}")
        if row.get("management_tip"):
            lines.append(f"  Management     : {row['management_tip']}")

        avoid = [f for f in (row.get("foods_to_avoid") or []) if f]
        eat   = [f for f in (row.get("foods_to_eat")   or []) if f]
        if avoid:
            lines.append(f"  Foods to AVOID : {', '.join(avoid)}")
        if eat:
            lines.append(f"  Foods to EAT   : {', '.join(eat)}")

        for field, label in [
            ("interaction_description", "Interaction   "),
            ("recommended_action",      "Action needed "),
            ("eating_relevance",        "Eating impact "),
            ("compounded_eating_effect","Compounds     "),
            ("mitigation",              "Mitigation    "),
        ]:
            if row.get(field):
                lines.append(f"  {label} : {row[field]}")

        guidelines = [g for g in (row.get("mandatory_guidelines") or []) if g]
        guidelines += [g for g in (row.get("effect_guidelines")    or []) if g]
        for gi, g in enumerate(guidelines[:3], 1):
            lines.append(f"  Guideline {gi}    : {g}")

        side_effects = [s for s in (row.get("side_effects") or []) if s]
        if side_effects:
            lines.append(f"  Side effects   : {', '.join(side_effects)}")

        lines.append("")

    return "\n".join(lines)

# =============================================================================
# CONTEXT BUILDER — FIX 1: separate Visual References section
# =============================================================================

def build_context(
    vector_docs:   List[Document],
    graph_context: str = "",
    image_docs:    List[Document] = None,   # FIX 1: new parameter
) -> str:
    """
    Combines graph structured facts + PDF chunk narrative + image chunks.

    Structure:
      ## Graph Knowledge Base        ← [G1], [G2] structured facts
        ...

      [1] Source: ... | Cancer: ... | Section: ...
          <chunk text>
      [2] ...

      ## Visual References           ← FIX 1: new dedicated section
      [IMG1] Source: ... (contains figure/table references)
          <image chunk with [IMAGE: filename.png] tags>

    Image chunks are placed in their OWN section so the LLM sees them
    clearly and cannot miss them. They are labelled [IMG1], [IMG2] to
    distinguish from content chunks [1], [2] etc.
    """
    parts = []

    # Graph context always first
    if graph_context:
        parts.append(graph_context)

    # Main content chunks
    for i, doc in enumerate(vector_docs, 1):
        sf = (
            doc.metadata.get("source_file")
            or re.sub(r'_cap_\d+$|_\d{4}$', '', doc.metadata.get("chunk_id", ""))
            or "unknown"
        )
        parts.append(
            f"[{i}] Source: {sf} | "
            f"Cancer: {doc.metadata.get('cancer_type', 'general')} | "
            f"Section: {doc.metadata.get('section_hierarchy', 'Body')}\n"
            f"{doc.page_content}"
        )

    # FIX 1: Dedicated visual references section
    # Placed AFTER content so it does not disrupt the main answer flow
    # but is clearly visible as a distinct labelled section
    if image_docs:
        visual_parts = ["## Visual References",
                        "The following visual assets are available. "
                        "Reference them in your answer using [IMAGE: filename.png] tags.\n"]
        for j, doc in enumerate(image_docs, 1):
            sf = (
                doc.metadata.get("source_file")
                or re.sub(r'_cap_\d+$|_\d{4}$', '', doc.metadata.get("chunk_id", ""))
                or "unknown"
            )
            visual_parts.append(
                f"[IMG{j}] Source: {sf} | "
                f"Cancer: {doc.metadata.get('cancer_type', 'general')}\n"
                f"{doc.page_content}"
            )
        parts.append("\n".join(visual_parts))

    return "\n\n".join(parts)

# =============================================================================
# LLM PROMPT — FIX 2: restored MUST language + dedicated visual instruction block
# =============================================================================

def _build_prompt(
    query:          str,
    patient_report: str,
    context_text:   str,
    history_text:   str,
    query_mode:     str,
    reasoning_path: str = "",
) -> str:
    mode_instruction = {
        QUERY_MODE_RESEARCH: (
            "You are answering from peer-reviewed clinical literature. "
            "Cite source numbers [1], [2] etc. for statistics and claims."
        ),
        QUERY_MODE_GRAPH: (
            "You are answering primarily from a structured medical knowledge graph "
            "([G1], [G2] etc.) with clinical literature for context. "
            "Prioritise graph facts for drug/food/interaction specifics."
        ),
        QUERY_MODE_AUTO: (
            "You are answering from both a structured knowledge graph ([G1], [G2] etc.) "
            "and peer-reviewed literature ([1], [2] etc.). "
            "Use graph facts for drug/food/interaction specifics and "
            "literature for clinical evidence and statistics."
        ),
    }.get(query_mode, "")

    # FIX 2 — Dedicated, unmissable visual references instruction block
    # Placed as a clearly labelled section so the LLM cannot overlook it.
    # Uses "MUST" and "ALWAYS" — mandatory language from original v3.
    visual_instruction = """
VISUAL REFERENCES INSTRUCTIONS (mandatory — read carefully):
- If the CLINICAL CONTEXT above contains a "## Visual References" section
  with [IMG1], [IMG2] etc. entries that contain [IMAGE: filename.png] tags,
  you MUST include the relevant image tags in your answer.
- Use this EXACT format in your answer: [IMAGE: filename.png]
  Example: "As shown in [IMAGE: breast-cancer-review_picture_1.png], the
  survival curve demonstrates..."
- ALWAYS reference images when the question asks about figures, charts,
  flowcharts, tables, or when a visual would support your answer.
- ONLY reference [IMAGE:] filenames that are explicitly listed in the
  context. NEVER invent or guess filenames.
- If multiple relevant images exist, include all of them.
"""

    return f"""You are an empathetic medical AI assistant helping cancer patients \
and clinicians understand medical information.

{mode_instruction}

PATIENT REPORT:
{patient_report if patient_report else "No patient report provided."}

CONVERSATION HISTORY:
{history_text if history_text else "No prior conversation."}

CLINICAL CONTEXT:
{context_text}

QUESTION:
{query}

{visual_instruction}
ANSWER INSTRUCTIONS:
- Answer using ONLY the information in the clinical context above.
- Explain clearly — avoid heavy jargon where possible.
- For graph results [G1], [G2] etc.: cite them for drug interactions,
  food guidance, and nutrition guidelines.
- For literature results [1], [2] etc.: cite them for survival statistics,
  trial data, and clinical evidence.
- If the answer is not in the context, clearly state you do not have
  enough information in the provided sources.
- End with a disclaimer advising consultation with a qualified oncologist.

REASONING PATH (internal — do not include in answer):
Mode={query_mode} | {reasoning_path}
"""

# =============================================================================
# HELPERS
# =============================================================================

# FIX 3 — Out-of-corpus topic patterns that should trigger proactive web search
# These are topics not covered by the 6 review papers or the hand-crafted graph
_OUT_OF_CORPUS_PATTERNS = [
    r'\bvaccine\b',
    r'\bvaccination\b',
    r'\bfda.approv',
    r'\bapproved.for',
    r'\bdrug.approv',
    r'\brecent.trial',
    r'\blatest.research',
    r'\bnew.treatment',
    r'\b202[3-9]\b',      # years beyond likely paper publication
    r'\b2030\b',
    r'\bclinical.trial.result',
    r'\bphase [123] trial',
    r'\bbreakthrough',
    r'\bfda.cleared',
    r'\bcdc.recommend',
    r'\bnhs.guideline',
    r'\bwho.recommend',
    r'\bcurrent.standard',
    r'\blatest.guideline',
]


def _is_out_of_corpus_query(query: str) -> bool:
    """
    FIX 3 — Proactive detection of queries outside the corpus.
    Returns True when the query is about topics the 6 review papers
    and hand-crafted graph are unlikely to cover well, so web search
    should fire proactively rather than waiting for the LLM to fail.
    """
    q = query.lower()
    return any(re.search(p, q) for p in _OUT_OF_CORPUS_PATTERNS)


def _rag_has_no_answer(answer: str) -> bool:
    """
    FIX 3 — Rewritten with three-tier logic.

    Tier 1: Short answer (<300 chars) containing any no-answer phrase
            → definitely no answer, trigger fallback immediately.

    Tier 2: Long answer containing MULTIPLE no-answer phrases
            → LLM admitted it doesn't know despite long context padding,
            trigger fallback.

    Tier 3: Answer contains the specific pattern of stating a topic is
            NOT in the context after providing some partial discussion
            → trigger fallback for better web-sourced answer.

    NOT triggered when: Answer is long and contains only one no-answer
    phrase in passing (the LLM acknowledged a limit but still answered).
    """
    answer_lower  = answer.lower()
    answer_length = len(answer.strip())

    # Count how many no-answer phrases appear
    phrase_hits = sum(1 for p in NO_ANSWER_PHRASES if p in answer_lower)

    # Tier 1: short answer with any no-answer signal
    if answer_length < 300 and phrase_hits >= 1:
        return True

    # Tier 2: long answer with multiple no-answer signals
    if phrase_hits >= 2:
        return True

    # Tier 3: specific pattern — LLM found partial context but stated
    # the specific question cannot be answered from that context
    # This catches the osteosarcoma vaccine case exactly:
    # "I do not have enough information to provide an answer about..."
    partial_no_answer_patterns = [
        r"i do not have enough information to (provide|answer|give)",
        r"cannot (provide|give|find) (a |an )?(specific |definitive )?answer",
        r"not (able|possible) to (answer|confirm|verify)",
        r"(this|the) (specific )?(question|topic|information) is not (covered|mentioned|available)",
        r"no (specific |definitive )?(information|data|evidence) (is |)(available|found|provided)",
    ]
    for pat in partial_no_answer_patterns:
        if re.search(pat, answer_lower):
            return True

    return False


def _build_sources(
    vector_docs:  List[Document],
    graph_intent: str = "",
    image_docs:   List[Document] = None,   # FIX 1: image docs in sources
) -> list[dict]:
    sources: list[dict] = []
    seen:    set        = set()

    if graph_intent and graph_intent != INTENT_GENERAL_GRAPH:
        sources.append({"label": "Cancer Treatment Knowledge Graph", "url": ""})
        seen.add("Cancer Treatment Knowledge Graph")

    for doc in vector_docs:
        sf = doc.metadata.get("source_file", "").strip()
        if not sf:
            sf = re.sub(r'_cap_\d+$|_\d{4}$', '', doc.metadata.get("chunk_id", ""))
        if not sf or sf in seen:
            continue
        seen.add(sf)
        sources.append({"label": sf, "url": doc.metadata.get("source_url", "")})

    # FIX 1: Include image chunk sources (may add sources not in main chunks)
    if image_docs:
        for doc in image_docs:
            sf = doc.metadata.get("source_file", "").strip()
            if not sf:
                sf = re.sub(r'_cap_\d+$|_\d{4}$', '', doc.metadata.get("chunk_id", ""))
            if not sf or sf in seen:
                continue
            seen.add(sf)
            sources.append({"label": sf, "url": doc.metadata.get("source_url", "")})

    return sources


def _generate_followups(answer: str, query: str, query_mode: str) -> list[str]:
    """Unchanged from v5."""
    mode_hint = {
        QUERY_MODE_GRAPH:    "Focus on food, nutrition, drug interactions, and side effects.",
        QUERY_MODE_RESEARCH: "Focus on clinical evidence, survival rates, and treatment rationale.",
        QUERY_MODE_AUTO:     "Mix of practical patient questions and clinical questions.",
    }.get(query_mode, "")

    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = (
            f"Based on this medical question and answer, generate exactly 3 "
            f"short follow-up questions a cancer patient might ask next. "
            f"{mode_hint} "
            f"Each question on its own line, no numbering.\n\n"
            f"Question: {query}\n\nAnswer excerpt: {answer[:400]}"
        )
        resp = client.chat.completions.create(
            model=GROQ_MODEL_QUERY, temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        lines = (resp.choices[0].message.content or "").strip().split("\n")
        return [l.strip() for l in lines if l.strip() and len(l.strip()) > 10][:3]
    except Exception:
        return []

# =============================================================================
# WEB FALLBACK — FIX 4 + FIX 5
# =============================================================================

def _duckduckgo_search(query: str, max_results: int = 5) -> list[dict]:
    """
    FIX 5 — Cleaner search query construction.
    Appends 'cancer oncology' only when the query doesn't already
    contain medical terms, to avoid query pollution on specific
    factual questions like vaccine approvals.
    """
    if not _DDG_AVAILABLE:
        return []

    medical_terms = {
        "cancer", "oncology", "tumor", "tumour", "chemotherapy",
        "treatment", "survival", "clinical", "drug", "vaccine",
        "osteosarcoma", "leukemia", "melanoma", "breast", "lung",
    }
    query_lower  = query.lower()
    already_medical = any(t in query_lower for t in medical_terms)
    search_query = query if already_medical else f"{query} cancer oncology"

    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(search_query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results
    except Exception as e:
        print(f"   ⚠️  DuckDuckGo search error: {e}")
        return []


def _web_search_fallback(
    rag_answer: str,
    query:      str,
    patient_report: str,
    rag_is_empty: bool = False,   # FIX 4: controls presentation
) -> tuple[str, list]:
    """
    FIX 4 — Improved web fallback presentation.

    When rag_is_empty=True (RAG had nothing): web answer stands alone.
    No "I don't know" preamble. Clean, direct answer from web sources.

    When rag_is_empty=False (RAG had partial info): RAG answer shown
    first as partial context, web answer clearly labelled as additional
    sources. Patient sees both without confusion.
    """
    print("   🌐 Running web search fallback...")
    client      = Groq(api_key=GROQ_API_KEY)
    web_results = _duckduckgo_search(query)
    web_sources = [{"label": r["url"], "url": r["url"]}
                   for r in web_results if r.get("url")]

    if web_results:
        web_context = "\n\n".join([
            f"[W{i+1}] {r['title']}\nURL: {r['url']}\n{r['snippet']}"
            for i, r in enumerate(web_results)
        ])
        web_prompt = (
            f"You are a medical AI assistant.\n"
            f"PATIENT REPORT:\n{patient_report or 'No patient report.'}\n\n"
            f"WEB SEARCH RESULTS:\n{web_context}\n\n"
            f"QUESTION: {query}\n\n"
            f"Provide a clear, accurate answer based on the web search results above. "
            f"Cite [W1], [W2] etc. when referencing specific sources. "
            f"End with a disclaimer to consult a qualified oncologist."
        )
    else:
        # No web results — use LLM general knowledge
        web_prompt = (
            f"You are a medical AI assistant.\n"
            f"PATIENT REPORT:\n{patient_report or 'No patient report.'}\n\n"
            f"QUESTION: {query}\n\n"
            f"Provide a clear, accurate answer from your medical knowledge. "
            f"Be explicit about uncertainty where appropriate. "
            f"End with a disclaimer to consult a qualified oncologist."
        )
        web_sources = [
            {"label": "https://www.cancer.gov",
             "url":   "https://www.cancer.gov"},
            {"label": "https://www.ncbi.nlm.nih.gov/pubmed/",
             "url":   "https://www.ncbi.nlm.nih.gov/pubmed/"},
        ]

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL_QUERY, temperature=GROQ_TEMP_QUERY,
            messages=[{"role": "user", "content": web_prompt}],
        )
        web_answer = resp.choices[0].message.content or ""
    except Exception as e:
        web_answer = f"Could not generate web answer: {e}"

    # FIX 4 — Presentation logic
    if rag_is_empty:
        # RAG had nothing — web answer stands alone, clean presentation
        final_answer = (
            "🌐 **Answer sourced from web search** "
            "(not found in clinical literature database):\n\n"
            + web_answer.strip()
        )
    else:
        # RAG had partial information — show both clearly labelled
        final_answer = (
            rag_answer.strip()
            + "\n\n---\n\n"
            + "🌐 **Additional sources found via web search:**\n\n"
            + web_answer.strip()
        )

    return final_answer, web_sources

# =============================================================================
# THREE-MODE ROUTING CORE — updated to use image_docs
# =============================================================================

def _run_research_mode(
    query: str, patient_report: str,
    chat_history: list, cancer_filter: str,
) -> tuple[str, list, list, str]:
    """
    PDF chunks + image chunks. Graph not involved.
    FIX 1: image chunks retrieved separately and passed to build_context.
    """
    vector_docs = _vector_retrieve(query, cancer_filter, RESEARCH_MODE_TOP_K)
    image_docs  = _retrieve_image_chunks(query)   # FIX 1

    context_text   = build_context(vector_docs, image_docs=image_docs)
    sources        = _build_sources(vector_docs, image_docs=image_docs)
    reasoning_path = (
        f"research mode | {len(vector_docs)} chunks | "
        f"{len(image_docs)} image chunks"
    )
    return context_text, vector_docs, sources, reasoning_path


def _run_graph_mode(
    query: str, patient_report: str,
    chat_history: list, cancer_filter: str,
) -> tuple[str, list, list, str]:
    """
    Graph first, small vector enrichment, image chunks injected.
    FIX 1: image chunks added separately from main vector retrieval.
    """
    graph_retriever           = get_graph_retriever()
    graph_context, graph_path = graph_retriever.retrieve(
        query=query, patient_report=patient_report, cancer_filter=cancer_filter,
    )
    vector_docs = _vector_retrieve(query, cancer_filter, GRAPH_MODE_VECTOR_ENRICHMENT)
    image_docs  = _retrieve_image_chunks(query)   # FIX 1

    context_text   = build_context(vector_docs, graph_context, image_docs)
    intent_info    = detect_query_intent(query, patient_report)
    sources        = _build_sources(vector_docs, intent_info["intent"], image_docs)
    reasoning_path = (
        f"graph mode | {graph_path} | "
        f"{len(image_docs)} image chunks"
    )
    return context_text, vector_docs, sources, reasoning_path


def _run_auto_mode(
    query: str, patient_report: str,
    chat_history: list, cancer_filter: str,
) -> tuple[str, list, list, str]:
    """
    Both paths. Graph fires when entities detected.
    FIX 1: image chunks added to all auto-mode responses.
    """
    combined = f"{query} {patient_report[:200]}".lower()
    use_graph = (
        any(kw in combined for kw in FOOD_KEYWORDS)
        or any(kw in combined for kw in INTERACTION_KEYWORDS)
        or any(c  in combined for c  in KNOWN_CANCERS)
        or any(d  in combined for d  in KNOWN_CHEMO_DRUGS)
        or any(d  in combined for d  in KNOWN_NON_CHEMO_DRUGS)
    )

    graph_context = ""
    graph_path    = "graph skipped — no medical entities detected"
    intent_used   = ""

    if use_graph:
        graph_retriever           = get_graph_retriever()
        graph_context, graph_path = graph_retriever.retrieve(
            query=query, patient_report=patient_report, cancer_filter=cancer_filter,
        )
        intent_used = detect_query_intent(query, patient_report)["intent"]

    vector_docs = _vector_retrieve(query, cancer_filter, K_MMR_FINAL)
    image_docs  = _retrieve_image_chunks(query)   # FIX 1

    context_text   = build_context(vector_docs, graph_context, image_docs)
    sources        = _build_sources(vector_docs, intent_used, image_docs)
    reasoning_path = (
        f"auto mode | graph={'yes' if use_graph else 'no'} | "
        f"{graph_path} | {len(image_docs)} image chunks"
    )
    return context_text, vector_docs, sources, reasoning_path

# =============================================================================
# PUBLIC API — generate_answer and generate_answer_stream
# Signatures unchanged. FIX 3: proactive out-of-corpus check added.
# =============================================================================

def generate_answer(
    query:          str,
    patient_report: str  = "",
    chat_history:   list = None,
    cancer_filter:  str  = "",
    query_mode:     str  = QUERY_MODE_DEFAULT,
) -> tuple[str, list]:
    """
    Main entry point. Returns (answer_str, sources_list).
    Images travel inline as [IMAGE: filename.png] in answer_str.
    FIX 3: Out-of-corpus queries now trigger proactive web search.
    """
    chat_history = chat_history or []

    try:
        print(f"\n🔍 [v5.1] Query {query[:70]}... | mode={query_mode}")

        # FIX 3 — Proactive web search for out-of-corpus topics
        # Checks BEFORE running any retrieval to avoid wasting time
        # on queries the corpus cannot answer
        if _is_out_of_corpus_query(query):
            print(f"   🌐 Out-of-corpus query detected → proactive web search")
            return _web_search_fallback(
                "", query, patient_report, rag_is_empty=True
            )

        # ── Route to correct retrieval path ───────────────────
        if query_mode == QUERY_MODE_RESEARCH:
            context_text, vector_docs, sources, reasoning_path = _run_research_mode(
                query, patient_report, chat_history, cancer_filter
            )
        elif query_mode == QUERY_MODE_GRAPH:
            context_text, vector_docs, sources, reasoning_path = _run_graph_mode(
                query, patient_report, chat_history, cancer_filter
            )
        else:
            context_text, vector_docs, sources, reasoning_path = _run_auto_mode(
                query, patient_report, chat_history, cancer_filter
            )

        if not context_text.strip():
            return _web_search_fallback(
                "", query, patient_report, rag_is_empty=True
            )

        history_text = ""
        if chat_history:
            history_text = "\n".join([
                f"{m['role'].upper()}: {m['content'][:300]}"
                for m in chat_history[-4:]
            ])

        prompt = _build_prompt(
            query, patient_report, context_text,
            history_text, query_mode, reasoning_path
        )

        client   = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL_QUERY, temperature=GROQ_TEMP_QUERY,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content or ""

        # FIX 3/4 — Smarter fallback trigger with correct presentation
        if _rag_has_no_answer(answer):
            print("   ⚠️  RAG insufficient → web fallback...")
            return _web_search_fallback(
                answer, query, patient_report,
                rag_is_empty=(len(answer.strip()) < 300)
            )

        return answer, sources

    except Exception as e:
        import traceback; traceback.print_exc()
        return f"Error in retrieval pipeline: {str(e)}", []


def generate_answer_stream(
    query:          str,
    patient_report: str  = "",
    chat_history:   list = None,
    cancer_filter:  str  = "",
    query_mode:     str  = QUERY_MODE_DEFAULT,
):
    """
    Streaming version. Yields text tokens.
    FIX 3: Out-of-corpus queries trigger web search before streaming.
    """
    import streamlit as st
    chat_history = chat_history or []

    try:
        print(f"\n🔍 [v5.1 stream] Query: {query[:70]}... | mode={query_mode}")

        # FIX 3 — Proactive web search for out-of-corpus queries
        if _is_out_of_corpus_query(query):
            print(f"   🌐 Out-of-corpus → proactive web search (stream)")
            answer, sources = _web_search_fallback(
                "", query, patient_report, rag_is_empty=True
            )
            yield answer
            st.session_state["stream_buffer"]    = answer
            st.session_state["stream_sources"]   = sources
            st.session_state["stream_followups"] = []
            st.session_state["stream_reasoning"] = "proactive web search — out-of-corpus query"
            return

        # ── Route to correct retrieval path ───────────────────
        if query_mode == QUERY_MODE_RESEARCH:
            context_text, vector_docs, sources, reasoning_path = _run_research_mode(
                query, patient_report, chat_history, cancer_filter
            )
        elif query_mode == QUERY_MODE_GRAPH:
            context_text, vector_docs, sources, reasoning_path = _run_graph_mode(
                query, patient_report, chat_history, cancer_filter
            )
        else:
            context_text, vector_docs, sources, reasoning_path = _run_auto_mode(
                query, patient_report, chat_history, cancer_filter
            )

        if not context_text.strip():
            fallback_answer, fallback_sources = _web_search_fallback(
                "", query, patient_report, rag_is_empty=True
            )
            yield fallback_answer
            st.session_state["stream_buffer"]    = fallback_answer
            st.session_state["stream_sources"]   = fallback_sources
            st.session_state["stream_followups"] = []
            st.session_state["stream_reasoning"] = "web fallback — no context found"
            return

        history_text = ""
        if chat_history:
            history_text = "\n".join([
                f"{m['role'].upper()}: {m['content'][:300]}"
                for m in chat_history[-4:]
            ])

        prompt = _build_prompt(
            query, patient_report, context_text,
            history_text, query_mode, reasoning_path
        )

        client = Groq(api_key=GROQ_API_KEY)
        stream = client.chat.completions.create(
            model=GROQ_MODEL_QUERY, temperature=GROQ_TEMP_QUERY,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        full_answer = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            full_answer += token
            yield token

        # FIX 3/4 — Post-stream fallback check
        if _rag_has_no_answer(full_answer):
            print("   ⚠️  Stream answer insufficient → web fallback appended")
            fallback_answer, fallback_sources = _web_search_fallback(
                full_answer, query, patient_report,
                rag_is_empty=(len(full_answer.strip()) < 300)
            )
            # Yield the additional web content as continuation tokens
            suffix = fallback_answer[len(full_answer):]
            for char in suffix:
                yield char
            full_answer   = fallback_answer
            sources       = fallback_sources

        followups = _generate_followups(full_answer, query, query_mode)
        st.session_state["stream_buffer"]    = full_answer
        st.session_state["stream_sources"]   = sources
        st.session_state["stream_followups"] = followups
        st.session_state["stream_reasoning"] = reasoning_path

    except Exception as e:
        yield f"Stream error: {str(e)}"
        import traceback; traceback.print_exc()

# =============================================================================
# DEMO — includes out-of-corpus test case
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  cancer_retrieval.py — v5.1 Bug-fix release")
    print("=" * 70)

    tests = [
        # Image retrieval tests
        ("What does the PRISMA flowchart show for systematic review?",
         QUERY_MODE_RESEARCH),
        ("Show me survival curves for osteosarcoma",
         QUERY_MODE_RESEARCH),

        # Graph mode tests
        ("What eating problems does cisplatin cause and what foods to avoid?",
         QUERY_MODE_GRAPH),
        ("Breast cancer patient on AC-T also taking warfarin — risks?",
         QUERY_MODE_GRAPH),

        # Out-of-corpus test — should trigger web search
        ("What vaccine is approved for preventing osteosarcoma?",
         QUERY_MODE_AUTO),

        # Normal research test
        ("What is the 5-year survival rate for osteosarcoma?",
         QUERY_MODE_RESEARCH),
    ]

    for q, mode in tests:
        print(f"\n{'─'*70}")
        print(f"❓ [{mode.upper()}] {q}")
        answer, sources = generate_answer_stream(q, query_mode=mode)
        print(f"📝 {answer[:400]}...")
        print(f"🖼️  Images : {'YES ✅' if '[IMAGE:' in answer else 'NO'}")
        print(f"🌐 Web    : {'YES ✅' if '🌐' in answer else 'NO'}")
        print(f"📚 Sources: {[s['label'] for s in sources]}")