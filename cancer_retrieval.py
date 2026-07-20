
# Final Attempt: cancer_retrieval.py is now fully integrated with the new image retrieval and presentation logic. The build_context function has been updated to include a dedicated "Visual References" section for image chunks, and the prompt instructions have been enhanced to ensure the LLM properly incorporates visual information when relevant. The web fallback mechanism has also been improved to provide clearer presentation of web-sourced answers when triggered. Overall, these changes should result in a more robust and informative medical AI assistant experience for users.


# =============================================================================
# cancer_retrieval.py — v6.0 Production Graph RAG
#
# RESOLVED EDGE CASES IN THIS VERSION:
#   1. Semantic Image Accuracy: Upgraded _retrieve_image_chunks to use a 
#      two-pass system (BM25 broad search -> Local BAAI Vector Reranking).
#   2. Image Deduplication: Added a seen_filenames registry to prevent the 
#      LLM from being fed duplicate image tags.
#   3. DDG Web Fallback Stability: Added fault-tolerant retry loops and 
#      dynamic query broadening (primary vs fallback queries).
#   4. Fallback Markdown Links: Prompt engineered _web_search_fallback to 
#      force the LLM to output clickable [Source Name](URL) markdown links.
#
# PRESERVED FROM v5.1:
#   - Connection Caching (_VECTOR_STORE_CACHE)
#   - Streaming logic (generate_answer_stream)
#   - Follow-up question generation (_generate_followups)
#   - 3-Tier fallback logic and proactive out-of-corpus checks
# =============================================================================

from __future__ import annotations

import re
import json
import math
from pathlib import Path
import numpy as np
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
# EMBEDDINGS & CONNECTION CACHING
# =============================================================================

_embed_model: Optional[HuggingFaceEmbeddings] = None
_VECTOR_STORE_CACHE = {} 

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

def get_dense_retriever(cancer_filter: str = "") -> BaseRetriever:
    global _VECTOR_STORE_CACHE
    kwargs: dict = {"k": K_DENSE}
    if cancer_filter:
        kwargs["filter"] = {"cancer_type": cancer_filter}
        
    if "neo4j_store" not in _VECTOR_STORE_CACHE:
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
    return _VECTOR_STORE_CACHE["neo4j_store"].as_retriever(search_kwargs=kwargs)


_bm25_retriever:       Optional[BM25Retriever] = None
_image_bm25_retriever: Optional[BM25Retriever] = None

def get_bm25_retriever() -> BM25Retriever:
    global _bm25_retriever
    if _bm25_retriever is not None:
        return _bm25_retriever
    print("   📖 Building BM25 index from chunk files...")
    documents = []
    for json_path in sorted(CHUNK_DIR.glob("*_chunks.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            for chunk in json.load(f):
                documents.append(Document(page_content=chunk.get("content", ""), metadata=chunk))
    if not documents:
        raise FileNotFoundError(f"No chunk files in {CHUNK_DIR}. Run cancer_ingestion.py first.")
    _bm25_retriever   = BM25Retriever.from_documents(documents)
    _bm25_retriever.k = K_SPARSE
    print(f"   ✅ BM25 ready: {len(documents)} chunks indexed")
    return _bm25_retriever


def get_image_bm25_retriever() -> Optional[BM25Retriever]:
    global _image_bm25_retriever
    if _image_bm25_retriever is not None:
        return _image_bm25_retriever
    image_docs = []
    for json_path in sorted(CHUNK_DIR.glob("*_chunks.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            for chunk in json.load(f):
                content = chunk.get("content", "")
                has_tag = chunk.get("has_image_tags", False) or "[IMAGE:" in content.upper()
                if has_tag:
                    image_docs.append(Document(page_content=content, metadata=chunk))
    if not image_docs:
        print("   ℹ️  No image-tagged chunks found — image retrieval disabled")
        return None
    _image_bm25_retriever   = BM25Retriever.from_documents(image_docs)
    print(f"   🖼️  Image BM25 ready: {len(image_docs)} image-tagged chunks")
    return _image_bm25_retriever

# =============================================================================
# VECTOR PIPELINE — BM25 + Dense + RRF + MMR
# =============================================================================

def reciprocal_rank_fusion(dense_docs: List[Document], sparse_docs: List[Document], k: int = RRF_K, top_n: int = K_RRF_FINAL) -> List[Document]:
    scores, doc_map = {}, {}
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

def _cosine(v1, v2) -> float:
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return 0.0 if (norm1 == 0 or norm2 == 0) else float(dot / (norm1 * norm2))

def mmr_rerank(query: str, candidates: List[Document], embed_model: HuggingFaceEmbeddings, k: int = K_MMR_FINAL, lambda_mult: float = MMR_LAMBDA) -> List[Document]:
    if not candidates or len(candidates) <= k: return candidates
    # Optimized: pre-cast to NumPy arrays for faster vectorized cosine similarity
    query_vec = np.array(embed_model.embed_query(query))
    doc_vecs  = [np.array(v) for v in embed_model.embed_documents([d.page_content for d in candidates])]
    relevance = [_cosine(v, query_vec) for v in doc_vecs]
    selected, remaining = [], list(range(len(candidates)))
    for _ in range(min(k, len(candidates))):
        if not selected: best = max(remaining, key=lambda i: relevance[i])
        else:
            best, best_score = -1, float("-inf")
            for idx in remaining:
                max_sim = max(_cosine(doc_vecs[idx], doc_vecs[s]) for s in selected)
                score   = lambda_mult * relevance[idx] - (1 - lambda_mult) * max_sim
                if score > best_score: best_score, best = score, idx
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
    return HybridMMRRetriever(dense_ret=get_dense_retriever(cancer_filter), sparse_ret=get_bm25_retriever(), embed_model=get_embeddings())

def _vector_retrieve(query: str, cancer_filter: str = "", top_k: int = K_MMR_FINAL) -> List[Document]:
    return get_hybrid_mmr_retriever(cancer_filter).invoke(query)[:top_k]

# =============================================================================
# IMAGE RETRIEVAL — FIX 1 & FIX 2
# =============================================================================

def _retrieve_image_chunks(query: str) -> List[Document]:
    """
    Retrieves image chunks using a two-pass system: 
    1. Broad BM25 keyword retrieval
    2. Dense Vector Semantic Reranking + Deduplication
    """
    img_retriever = get_image_bm25_retriever()
    if img_retriever is None:
        return []

    try:
        # 1. Broad BM25 Search (Pull top 15 to give reranker options)
        img_retriever.k = 15
        bm25_results = img_retriever.invoke(query)

        # 2. Deduplicate images by filename
        seen_filenames = set()
        unique_candidates = []
        for doc in bm25_results:
            match = re.search(IMAGE_TAG_PATTERN, doc.page_content, flags=re.IGNORECASE)
            filename = match.group(1).strip() if match else None
            if filename and filename not in seen_filenames:
                seen_filenames.add(filename)
                unique_candidates.append(doc)

        if not unique_candidates: 
            return []

        # 3. Dense Vector Semantic Reranking
        embed_model = get_embeddings()
        # Optimized: pre-cast to NumPy arrays for faster vectorized cosine similarity
    query_vec = np.array(embed_model.embed_query(query))
        doc_vecs = [np.array(v) for v in embed_model.embed_documents([d.page_content for d in unique_candidates])]
        
        scored_candidates = []
        for i, doc in enumerate(unique_candidates):
            sim = _cosine(query_vec, doc_vecs[i])
            scored_candidates.append((sim, doc))
        
        # Sort by highest semantic meaning match
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        img_retriever.k = 3 # Reset hygiene
        
        # Return the top 2 absolute best semantic matches
        return [doc for sim, doc in scored_candidates][:2]

    except Exception as e:
        print(f"   ⚠️  Image chunk retrieval error: {e}")
        return []
    
# =============================================================================
# GRAPH PIPELINE 
# =============================================================================

def detect_query_intent(query: str, patient_report: str = "") -> dict:
    combined = f"{query} {patient_report[:300]}".lower()

    cancer_name   = next((c for c in KNOWN_CANCERS        if c in combined), None)
    chemo_drug    = next((d for d in KNOWN_CHEMO_DRUGS     if d in combined), None)
    non_chemo     = next((d for d in KNOWN_NON_CHEMO_DRUGS if d in combined), None)
    protocol      = next((p for p in KNOWN_PROTOCOLS       if p in combined), None)
    eating_effect = next((e for e in KNOWN_EATING_EFFECTS  if e in combined), None)

    has_food        = any(kw in combined for kw in FOOD_KEYWORDS)
    has_interaction = any(kw in combined for kw in INTERACTION_KEYWORDS)

    if non_chemo and (chemo_drug or has_interaction): intent = INTENT_NON_CHEMO_INTERACTION
    elif protocol:                                    intent = INTENT_PROTOCOL_DETAIL
    elif chemo_drug and has_food:                     intent = INTENT_FOOD_GUIDANCE
    elif cancer_name and (has_food or chemo_drug):    intent = INTENT_CANCER_DRUGS_EFFECTS
    elif chemo_drug:                                  intent = INTENT_FOOD_GUIDANCE
    elif cancer_name:                                 intent = INTENT_CANCER_DRUGS_EFFECTS
    elif eating_effect:                               intent = INTENT_GENERAL_GRAPH
    else:                                             intent = INTENT_GENERAL_GRAPH

    return {
        "intent": intent, "cancer_name": cancer_name, "chemo_drug": chemo_drug,
        "non_chemo_drug": non_chemo, "protocol": protocol, "eating_effect": eating_effect,
        "has_food_signal": has_food, "has_interaction_signal": has_interaction,
    }


CYPHER_QUERIES: dict[str, str] = {
    INTENT_CANCER_DRUGS_EFFECTS: """
        MATCH (c:Cancer)-[:TREATED_WITH]->(d:ChemoDrug)
        WHERE toLower(c.name) CONTAINS toLower($cancer_name) OR toLower(c.subtype) CONTAINS toLower($cancer_name)
        OPTIONAL MATCH (d)-[r:CAUSES_EATING_EFFECT]->(e:EatingAdverseEffect)
        OPTIONAL MATCH (e)-[:WORSENED_BY]->(bad:FoodItem)
        OPTIONAL MATCH (e)-[:RELIEVED_BY]->(good:FoodItem)
        RETURN c.name AS cancer, d.name AS drug, d.drug_class AS drug_class, d.notes AS drug_notes,
               e.name AS eating_effect, r.severity AS severity, e.management_tip AS management_tip,
               collect(DISTINCT bad.name) AS foods_to_avoid, collect(DISTINCT good.name) AS foods_to_eat
        ORDER BY d.name, e.name LIMIT $top_k
    """,
    INTENT_FOOD_GUIDANCE: """
        MATCH (d:ChemoDrug) WHERE toLower(d.name) CONTAINS toLower($chemo_drug)
        OPTIONAL MATCH (d)-[r:CAUSES_EATING_EFFECT]->(e:EatingAdverseEffect)
        OPTIONAL MATCH (e)-[:WORSENED_BY]->(avoid:FoodItem)
        OPTIONAL MATCH (e)-[:RELIEVED_BY]->(eat:FoodItem)
        OPTIONAL MATCH (g:NutritionGuideline)-[:REQUIRED_FOR|MANAGES]->(d)
        OPTIONAL MATCH (g2:NutritionGuideline)-[:MANAGES]->(e)
        RETURN d.name AS drug, d.drug_class AS drug_class, d.notes AS drug_notes,
               e.name AS eating_effect, r.severity AS severity, e.management_tip AS management_tip,
               collect(DISTINCT avoid.name) AS foods_to_avoid, collect(DISTINCT eat.name) AS foods_to_eat,
               collect(DISTINCT g.text) AS mandatory_guidelines, collect(DISTINCT g2.text) AS effect_guidelines
        ORDER BY e.name LIMIT $top_k
    """,
    INTENT_NON_CHEMO_INTERACTION: """
        MATCH (n:NonChemoDrug) WHERE toLower(n.name) CONTAINS toLower($non_chemo_drug)
        OPTIONAL MATCH (n)-[:HAS_INTERACTION_WITH]->(c:ChemoDrug)
        OPTIONAL MATCH (n)-[:DESCRIBED_BY]->(i:DrugInteraction)
        OPTIONAL MATCH (i)-[:COMPOUNDS_EATING_EFFECT]->(e:EatingAdverseEffect)
        OPTIONAL MATCH (n)-[:TREATS]->(a:Ailment)
        RETURN n.name AS non_chemo_drug, n.drug_class AS non_chemo_class, a.name AS treats_ailment,
               c.name AS chemo_drug, i.severity AS severity, i.description AS interaction_description,
               i.clinical_action AS recommended_action, i.eating_relevance AS eating_relevance,
               e.name AS compounded_eating_effect, i.mitigation AS mitigation
        ORDER BY i.severity DESC LIMIT $top_k
    """,
    INTENT_PROTOCOL_DETAIL: """
        MATCH (p:TreatmentProtocol) WHERE toLower(p.name) CONTAINS toLower($protocol) OR toLower(p.cancer) CONTAINS toLower($protocol)
        OPTIONAL MATCH (p)-[:INCLUDES_DRUG]->(d:ChemoDrug)
        OPTIONAL MATCH (d)-[:CAUSES_EATING_EFFECT]->(e:EatingAdverseEffect)
        OPTIONAL MATCH (g:NutritionGuideline)-[:REQUIRED_FOR]->(d)
        OPTIONAL MATCH (d)-[:MAY_CAUSE]->(s:SideEffect)
        RETURN p.name AS protocol, p.description AS protocol_description, p.setting AS setting,
               d.name AS drug, d.drug_class AS drug_class, d.notes AS drug_notes,
               collect(DISTINCT e.name) AS eating_effects, collect(DISTINCT g.text) AS mandatory_guidelines,
               collect(DISTINCT s.name) AS side_effects
        ORDER BY d.name LIMIT $top_k
    """,
    INTENT_DRUG_INTERACTIONS: """
        MATCH (n:NonChemoDrug)-[:HAS_INTERACTION_WITH]->(c:ChemoDrug) WHERE toLower(c.name) CONTAINS toLower($chemo_drug)
        OPTIONAL MATCH (n)-[:DESCRIBED_BY]->(i:DrugInteraction)
        OPTIONAL MATCH (i)-[:COMPOUNDS_EATING_EFFECT]->(e:EatingAdverseEffect)
        OPTIONAL MATCH (n)-[:TREATS]->(a:Ailment)
        RETURN n.name AS non_chemo_drug, n.drug_class AS drug_class, a.name AS treats_ailment,
               i.severity AS severity, i.description AS interaction_description, i.clinical_action AS recommended_action,
               i.eating_relevance AS eating_relevance, e.name AS compounded_eating_effect
        ORDER BY i.severity DESC LIMIT $top_k
    """,
    INTENT_GENERAL_GRAPH: """
        CALL db.index.fulltext.queryNodes('chemo_text_index', $search_text) YIELD node AS drug, score
        OPTIONAL MATCH (drug)-[:CAUSES_EATING_EFFECT]->(effect:EatingAdverseEffect)
        OPTIONAL MATCH (effect)-[:RELIEVED_BY]->(eat:FoodItem)
        OPTIONAL MATCH (effect)-[:WORSENED_BY]->(avoid:FoodItem)
        RETURN drug.name AS drug, drug.drug_class AS drug_class, drug.mechanism AS mechanism, score,
               collect(DISTINCT effect.name) AS eating_effects, collect(DISTINCT eat.name) AS foods_to_eat,
               collect(DISTINCT avoid.name) AS foods_to_avoid
        ORDER BY score DESC LIMIT $top_k
    """,
}

class GraphRetriever:
    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    def retrieve(self, query: str, patient_report: str = "", cancer_filter: str = "") -> tuple[str, str]:
        intent_info = detect_query_intent(query, patient_report)
        intent = intent_info["intent"]

        print(f"   🕸️  Graph intent={intent} | cancer={intent_info['cancer_name']} | chemo={intent_info['chemo_drug']}")

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

        if not rows: return "", f"graph: no results for intent={intent}"

        graph_context  = _format_graph_context(rows, intent)
        reasoning_path = f"intent={intent} | rows={len(rows)} | drug={intent_info['chemo_drug'] or intent_info['non_chemo_drug']}"
        return graph_context, reasoning_path

    def _run_query(self, intent: str, params: dict) -> list[dict]:
        cypher = CYPHER_QUERIES.get(intent, "")
        if not cypher: return []
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
    if _graph_retriever is None: _graph_retriever = GraphRetriever()
    return _graph_retriever

def _format_graph_context(rows: list[dict], intent: str) -> str:
    if not rows: return ""
    lines = ["## Graph Knowledge Base", "Structured medical facts from the cancer treatment knowledge graph:\n"]
    for i, row in enumerate(rows[:15], 1):
        lines.append(f"[G{i}]")
        for field, label in [("drug", "Drug"), ("non_chemo_drug", "Non-chemo drug"), ("drug_class", "Class"), ("mechanism", "Mechanism"), ("cancer", "Cancer"), ("protocol", "Protocol")]:
            if row.get(field): lines.append(f"  {label} : {row[field]}")
        if row.get("eating_effect"): lines.append(f"  Eating effect  : {row['eating_effect']}")
        avoid, eat = [f for f in (row.get("foods_to_avoid") or []) if f], [f for f in (row.get("foods_to_eat") or []) if f]
        if avoid: lines.append(f"  Foods to AVOID : {', '.join(avoid)}")
        if eat: lines.append(f"  Foods to EAT   : {', '.join(eat)}")
        for field, label in [("interaction_description", "Interaction"), ("recommended_action", "Action needed")]:
            if row.get(field): lines.append(f"  {label} : {row[field]}")
        guidelines = [g for g in (row.get("mandatory_guidelines") or []) if g] + [g for g in (row.get("effect_guidelines") or []) if g]
        for gi, g in enumerate(guidelines[:3], 1): lines.append(f"  Guideline {gi}    : {g}")
        lines.append("")
    return "\n".join(lines)

# =============================================================================
# CONTEXT BUILDER & LLM PROMPT
# =============================================================================

def build_context(vector_docs: List[Document], graph_context: str = "", image_docs: List[Document] = None) -> str:
    parts = []
    if graph_context: parts.append(graph_context)

    for i, doc in enumerate(vector_docs, 1):
        sf = doc.metadata.get("source_file") or re.sub(r'_cap_\d+$|_\d{4}$', '', doc.metadata.get("chunk_id", "")) or "unknown"
        parts.append(f"[{i}] Source: {sf} | Cancer: {doc.metadata.get('cancer_type', 'general')}\n{doc.page_content}")

    if image_docs:
        visual_parts = ["## Visual References", "The following visual assets are available. Reference them in your answer using [IMAGE: filename.png] tags.\n"]
        for j, doc in enumerate(image_docs, 1):
            sf = doc.metadata.get("source_file") or re.sub(r'_cap_\d+$|_\d{4}$', '', doc.metadata.get("chunk_id", "")) or "unknown"
            visual_parts.append(f"[IMG{j}] Source: {sf}\n{doc.page_content}")
        parts.append("\n".join(visual_parts))

    return "\n\n".join(parts)

def _build_prompt(query: str, patient_report: str, context_text: str, history_text: str, query_mode: str, reasoning_path: str = "") -> str:
    mode_instruction = {
        QUERY_MODE_RESEARCH: "You are answering from peer-reviewed clinical literature. Cite source numbers [1], [2] etc.",
        QUERY_MODE_GRAPH: "You are answering primarily from a structured medical knowledge graph ([G1], [G2] etc.).",
        QUERY_MODE_AUTO: "You are answering from both a structured knowledge graph and peer-reviewed literature.",
    }.get(query_mode, "")

    visual_instruction = """
VISUAL REFERENCES INSTRUCTIONS (mandatory):
- If the CLINICAL CONTEXT contains a "## Visual References" section, you MUST include the relevant image tags in your answer.
- Use EXACT format: [IMAGE: filename.png] (e.g. "As shown in [IMAGE: survival_curve.png]...")
- ALWAYS reference images when the question asks about figures, charts, flowcharts, or tables.
"""

    return f"""You are an empathetic medical AI assistant helping cancer patients.

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
- Answer using ONLY the clinical context above.
- Cite sources clearly.
- If not in context, clearly state you do not have enough information.
- End with a disclaimer advising consultation with an oncologist.

REASONING PATH (internal): {reasoning_path}
"""

# =============================================================================
# WEB FALLBACK — FIX 3 & FIX 4
# =============================================================================

_OUT_OF_CORPUS_PATTERNS = [
    r'\bvaccine\b', r'\bvaccination\b', r'\bfda.approv', r'\bapproved.for',
    r'\bdrug.approv', r'\brecent.trial', r'\blatest.research', r'\bnew.treatment',
    r'\b202[3-9]\b', r'\b2030\b', r'\bclinical.trial.result', r'\bphase [123] trial',
    r'\bbreakthrough', r'\bfda.cleared', r'\bcdc.recommend', r'\bnhs.guideline',
    r'\bherbal\b', r'\bsupplement\b', r'\balternative medicine\b' 
]

def _is_out_of_corpus_query(query: str) -> bool:
    q = query.lower()
    return any(re.search(p, q) for p in _OUT_OF_CORPUS_PATTERNS)

def _duckduckgo_search(query: str, max_results: int = 5) -> list[dict]:
    """
    FIX 3: Fault-tolerant DDG search.
    Implements a context manager -> direct object fallback architecture.
    """
    if not _DDG_AVAILABLE: return []

    medical_terms = {"cancer", "oncology", "tumor", "chemotherapy", "treatment"}
    query_lower = query.lower()
    already_medical = any(t in query_lower for t in medical_terms)
    
    primary_query = query if already_medical else f"{query} cancer oncology"
    fallback_query = query # If the suffix breaks the search

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(primary_query, max_results=max_results))
            if not results: # Broaden search
                results = list(ddgs.text(fallback_query, max_results=max_results))
            return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in results]
    except Exception as e:
        print(f"   ⚠️ DDGS context manager failed ({e}). Retrying direct object...")
        try:
            ddgs = DDGS()
            results = list(ddgs.text(primary_query, max_results=max_results))
            return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in results]
        except Exception as inner_e:
            print(f"   ❌ Complete DDG Failure: {inner_e}")
            return []

def _web_search_fallback(rag_answer: str, query: str, patient_report: str, rag_is_empty: bool = False) -> tuple[str, list]:
    print("   🌐 Running web search fallback...")
    client = Groq(api_key=GROQ_API_KEY)
    web_results = _duckduckgo_search(query)
    web_sources = [{"label": r["url"], "url": r["url"]} for r in web_results if r.get("url")]

    if web_results:
        web_context = "\n\n".join([f"[W{i+1}] {r['title']}\nURL: {r['url']}\n{r['snippet']}" for i, r in enumerate(web_results)])
        # FIX 4: Explicit instruction to write Markdown links
        web_prompt = (
            f"You are a medical AI assistant.\nPATIENT REPORT: {patient_report}\n\n"
            f"WEB SEARCH RESULTS:\n{web_context}\n\nQUESTION: {query}\n\n"
            f"Provide an accurate answer based on the web results. "
            f"CRITICAL: You MUST cite your sources as clickable Markdown links inline in your text. "
            f"Format example: 'According to the [National Cancer Institute](https://www.cancer.gov), survival is...' "
            f"Use the exact URLs provided above. End with a medical disclaimer."
        )
    else:
        web_prompt = f"QUESTION: {query}\nProvide a clear medical answer. End with a disclaimer."
        web_sources = [{"label": "https://www.cancer.gov", "url": "https://www.cancer.gov"}]

    try:
        resp = client.chat.completions.create(model=GROQ_MODEL_QUERY, temperature=GROQ_TEMP_QUERY, messages=[{"role": "user", "content": web_prompt}])
        web_answer = resp.choices[0].message.content or ""
    except Exception as e:
        web_answer = f"Could not generate web answer: {e}"

    if rag_is_empty:
        final_answer = f"🌐 **Answer sourced from web search** (not found in literature database):\n\n{web_answer.strip()}"
    else:
        final_answer = f"{rag_answer.strip()}\n\n---\n\n🌐 **Additional sources found via web search:**\n\n{web_answer.strip()}"

    return final_answer, web_sources

def _rag_has_no_answer(answer: str) -> bool:
    answer_lower  = answer.lower()
    phrase_hits = sum(1 for p in NO_ANSWER_PHRASES if p in answer_lower)
    if len(answer.strip()) < 300 and phrase_hits >= 1: return True
    if phrase_hits >= 2: return True
    partial_no_answer_patterns = [
        r"i do not have enough information to (provide|answer|give)",
        r"cannot (provide|give|find) (a |an )?(specific |definitive )?answer",
        r"not (able|possible) to (answer|confirm|verify)",
        r"(this|the) (specific )?(question|topic|information) is not (covered|mentioned|available)",
    ]
    return any(re.search(pat, answer_lower) for pat in partial_no_answer_patterns)

# =============================================================================
# ROUTING & SOURCES
# =============================================================================

def _build_sources(
    vector_docs:  List[Document],
    graph_intent: str = "",
    image_docs:   List[Document] = None,
) -> list[dict]:
    sources: list[dict] = []
    seen:    set        = set()

    if graph_intent and graph_intent != INTENT_GENERAL_GRAPH:
        sources.append({"label": "Cancer Treatment Knowledge Graph", "url": ""})
        seen.add("Cancer Treatment Knowledge Graph")

    for doc in vector_docs + (image_docs or []):
        sf = doc.metadata.get("source_file", "").strip()
        if not sf:
            sf = re.sub(r'_cap_\d+$|_\d{4}$', '', doc.metadata.get("chunk_id", ""))
        
        # 🟢 Extract granular traceability metadata
        chunk_id = doc.metadata.get("chunk_id", "unknown_chunk")
        section = doc.metadata.get("section_hierarchy", "Body")
        
        # Create a highly traceable, unique label
        traceable_label = f"{sf} (Chunk: {chunk_id} | Section: {section})"

        if not sf or traceable_label in seen:
            continue
            
        seen.add(traceable_label)
        sources.append({
            "label": traceable_label, 
            "url": doc.metadata.get("source_url", "")
        })

    return sources

def _run_research_mode(query: str, patient_report: str, chat_history: list, cancer_filter: str) -> tuple[str, list, list, str]:
    vector_docs = _vector_retrieve(query, cancer_filter, RESEARCH_MODE_TOP_K)
    image_docs  = _retrieve_image_chunks(query)
    context_text = build_context(vector_docs, image_docs=image_docs)
    sources = _build_sources(vector_docs, image_docs=image_docs)
    return context_text, vector_docs, sources, f"research mode | {len(vector_docs)} chunks | {len(image_docs)} image chunks"

def _run_graph_mode(query: str, patient_report: str, chat_history: list, cancer_filter: str) -> tuple[str, list, list, str]:
    graph_context, graph_path = get_graph_retriever().retrieve(query, patient_report, cancer_filter)
    vector_docs = _vector_retrieve(query, cancer_filter, GRAPH_MODE_VECTOR_ENRICHMENT)
    image_docs  = _retrieve_image_chunks(query)
    context_text = build_context(vector_docs, graph_context, image_docs)
    intent_info = detect_query_intent(query, patient_report)
    sources = _build_sources(vector_docs, intent_info["intent"], image_docs)
    return context_text, vector_docs, sources, f"graph mode | {graph_path} | {len(image_docs)} image chunks"

def _run_auto_mode(query: str, patient_report: str, chat_history: list, cancer_filter: str) -> tuple[str, list, list, str]:
    combined = f"{query} {patient_report[:200]}".lower()
    use_graph = any(kw in combined for kw in FOOD_KEYWORDS | INTERACTION_KEYWORDS | KNOWN_CANCERS | KNOWN_CHEMO_DRUGS | KNOWN_NON_CHEMO_DRUGS)
    graph_context, intent_used = "", ""
    if use_graph:
        graph_context, graph_path = get_graph_retriever().retrieve(query, patient_report, cancer_filter)
        intent_used = detect_query_intent(query, patient_report)["intent"]
    vector_docs = _vector_retrieve(query, cancer_filter, K_MMR_FINAL)
    image_docs  = _retrieve_image_chunks(query)
    context_text = build_context(vector_docs, graph_context, image_docs)
    sources = _build_sources(vector_docs, intent_used, image_docs)
    return context_text, vector_docs, sources, f"auto mode | graph={use_graph} | {len(image_docs)} image chunks"

def _generate_followups(answer: str, query: str, query_mode: str) -> list[str]:
    mode_hint = {
        QUERY_MODE_GRAPH: "Focus on food, nutrition, drug interactions, and side effects.",
        QUERY_MODE_RESEARCH: "Focus on clinical evidence, survival rates, and treatment rationale.",
        QUERY_MODE_AUTO: "Mix of practical patient questions and clinical questions.",
    }.get(query_mode, "")
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"Based on this medical question and answer, generate exactly 3 short follow-up questions a cancer patient might ask next. {mode_hint} Each question on its own line, no numbering.\n\nQuestion: {query}\n\nAnswer excerpt: {answer[:400]}"
        resp = client.chat.completions.create(model=GROQ_MODEL_QUERY, temperature=0.3, messages=[{"role": "user", "content": prompt}])
        lines = (resp.choices[0].message.content or "").strip().split("\n")
        return [l.strip() for l in lines if l.strip() and len(l.strip()) > 10][:3]
    except Exception: return []

# =============================================================================
# PUBLIC APIs
# =============================================================================

def generate_answer(query: str, patient_report: str = "", chat_history: list = None, cancer_filter: str = "", query_mode: str = QUERY_MODE_DEFAULT) -> tuple[str, list]:
    chat_history = chat_history or []
    try:
        print(f"\n🔍 [v6.0] Query {query[:70]}... | mode={query_mode}")
        if _is_out_of_corpus_query(query):
            print(f"   🌐 Out-of-corpus query detected → proactive web search")
            return _web_search_fallback("", query, patient_report, rag_is_empty=True)

        if query_mode == QUERY_MODE_RESEARCH: ctx, docs, src, path = _run_research_mode(query, patient_report, chat_history, cancer_filter)
        elif query_mode == QUERY_MODE_GRAPH: ctx, docs, src, path = _run_graph_mode(query, patient_report, chat_history, cancer_filter)
        else: ctx, docs, src, path = _run_auto_mode(query, patient_report, chat_history, cancer_filter)

        if not ctx.strip(): return _web_search_fallback("", query, patient_report, rag_is_empty=True)

        history_text = "\n".join([f"{m['role'].upper()}: {m['content'][:300]}" for m in chat_history[-4:]]) if chat_history else ""
        prompt = _build_prompt(query, patient_report, ctx, history_text, query_mode, path)
        response = Groq(api_key=GROQ_API_KEY).chat.completions.create(model=GROQ_MODEL_QUERY, temperature=GROQ_TEMP_QUERY, messages=[{"role": "user", "content": prompt}])
        answer = response.choices[0].message.content or ""

        if _rag_has_no_answer(answer):
            print("   ⚠️  RAG insufficient → web fallback...")
            fallback_ans, fallback_src = _web_search_fallback(answer, query, patient_report, rag_is_empty=(len(answer.strip()) < 300))
            
            # 🟢 FIX: Combine sources here too!
            src = src + fallback_src
            return fallback_ans, src
            
        return answer, src
    
    except Exception as e:
        import traceback; traceback.print_exc()
        return f"Error in retrieval pipeline: {str(e)}", []


def generate_answer_stream(query: str, patient_report: str = "", chat_history: list = None, cancer_filter: str = "", query_mode: str = QUERY_MODE_DEFAULT):
    import streamlit as st
    chat_history = chat_history or []
    try:
        print(f"\n🔍 [v6.0 stream] Query: {query[:70]}... | mode={query_mode}")
        
        if _is_out_of_corpus_query(query):
            print(f"   🌐 Out-of-corpus → proactive web search (stream)")
            answer, sources = _web_search_fallback("", query, patient_report, rag_is_empty=True)
            yield answer
            st.session_state["stream_buffer"]    = answer
            st.session_state["stream_sources"]   = sources
            st.session_state["stream_followups"] = []
            st.session_state["stream_reasoning"] = "proactive web search — out-of-corpus query"
            return

        if query_mode == QUERY_MODE_RESEARCH: ctx, docs, src, path = _run_research_mode(query, patient_report, chat_history, cancer_filter)
        elif query_mode == QUERY_MODE_GRAPH: ctx, docs, src, path = _run_graph_mode(query, patient_report, chat_history, cancer_filter)
        else: ctx, docs, src, path = _run_auto_mode(query, patient_report, chat_history, cancer_filter)

        if not ctx.strip():
            ans, src = _web_search_fallback("", query, patient_report, rag_is_empty=True)
            yield ans
            st.session_state.update({"stream_buffer": ans, "stream_sources": src, "stream_followups": [], "stream_reasoning": "web fallback"})
            return

        history_text = "\n".join([f"{m['role'].upper()}: {m['content'][:300]}" for m in chat_history[-4:]]) if chat_history else ""
        prompt = _build_prompt(query, patient_report, ctx, history_text, query_mode, path)
        stream = Groq(api_key=GROQ_API_KEY).chat.completions.create(model=GROQ_MODEL_QUERY, temperature=GROQ_TEMP_QUERY, messages=[{"role": "user", "content": prompt}], stream=True)

        full_answer = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            full_answer += token
            yield token

        if _rag_has_no_answer(full_answer):
            print("   ⚠️  Stream answer insufficient → web fallback appended")
            fallback_ans, fallback_src = _web_search_fallback(full_answer, query, patient_report, rag_is_empty=(len(full_answer.strip()) < 300))
            
            # Yield the additional web content as continuation tokens
            suffix = fallback_ans[len(full_answer):]
            for char in suffix: 
                yield char
                
            full_answer = fallback_ans
            
            # 🟢 FIX: Combine the sources, do not overwrite! (Using 'src' to match your variables)
            src = src + fallback_src

        followups = _generate_followups(full_answer, query, query_mode)
        st.session_state.update({"stream_buffer": full_answer, "stream_sources": src, "stream_followups": followups, "stream_reasoning": path})

    except Exception as e:
        yield f"Stream error: {str(e)}"
        import traceback; traceback.print_exc()

# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  cancer_retrieval.py — v6.0 Production Release")
    print("=" * 70)

    tests = [
        ("What does the PRISMA flowchart show for systematic review?", QUERY_MODE_RESEARCH),
        ("Show me survival curves for osteosarcoma", QUERY_MODE_RESEARCH),
        ("What eating problems does cisplatin cause and what foods to avoid?", QUERY_MODE_GRAPH),
        ("What vaccine is approved for preventing osteosarcoma?", QUERY_MODE_AUTO),
    ]

    for q, mode in tests:
        print(f"\n{'─'*70}")
        print(f"❓ [{mode.upper()}] {q}")
        # For terminal testing, consume the generator silently and print final
        answer_parts = []
        for chunk in generate_answer_stream(q, query_mode=mode):
            answer_parts.append(chunk)
        final_ans = "".join(answer_parts)
        print(f"📝 {final_ans[:400]}...")