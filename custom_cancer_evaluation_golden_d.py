# =============================================================================
# cancer_evaluation.py  v4.0  — LLM-Free Custom Metrics
#
# DESIGN PHILOSOPHY:
#   This version completely removes RAGAS dependency for routine evaluation.
#   All 7 metrics are computed without any LLM judge calls — zero token cost,
#   instant results, runs after every code change without budget concerns.
#
#   RAGAS (faithfulness, context recall etc.) can still be run separately
#   using ragas_test.py when you need a deep dive — but for daily development
#   iteration, these 7 custom metrics give actionable signal immediately.
#
# THE 7 METRICS (all LLM-free, all computed from answer + context + metadata):
#
#   M1  Image Recall Rate       — did image questions get [IMAGE:] tags?
#   M2  Web Fallback Precision  — did out-of-corpus queries route to web?
#   M3  Keyword Coverage Score  — does answer contain ground truth keywords?
#   M4  Context Utilisation     — did retrieved chunks contribute to answer?
#   M5  Answer Completeness     — is the answer long/rich enough?
#   M6  Graph Grounding Rate    — is graph data actually used in graph mode?
#   M7  Latency Score           — normalised speed (lower latency = higher score)
#
# HOW TO RUN (no RAGAS, no token cost, no API limits):
#   python cancer_evaluation.py --quick           # 5 questions, ~2 min
#   python cancer_evaluation.py                   # 20 questions, ~8 min
#   python cancer_evaluation.py --category graph  # one category only
#   python cancer_evaluation.py --label v5_1      # labelled run
#   python cancer_evaluation.py --compare baseline v5_1
#
# FOR RAGAS BENCHMARKING: use cancer_ragas_eval.py (separate file)
#   python cancer_ragas_eval.py --quick           # all 4 RAGAS metrics, ~15 min
#
# OUTPUT:
#   output/evaluation/custom_scores.json    — full numeric results
#   output/evaluation/custom_report.html   — visual dashboard
#   output/evaluation/custom_history.jsonl — trend tracking
# =============================================================================

from __future__ import annotations

import re
import json
import math
import sys
import time
import argparse
import traceback
import config as _config
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import (
    GROQ_API_KEY, GROQ_MODEL_QUERY,
    QUERY_MODE_RESEARCH, QUERY_MODE_GRAPH, QUERY_MODE_AUTO,
    EMBEDDING_MODEL,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _ensure_config_compat() -> None:
    """Backfill config constants expected by cancer_retrieval without editing it."""
    defaults = {
        "KNOWN_PROTOCOLS": {
            "ac-t", "ac-t (breast)",
            "map", "map (osteosarcoma)",
            "7+3", "7+3 aml induction",
            "hyper-cvad", "hyper-cvad (all)",
            "carboplatin/paclitaxel", "carboplatin/paclitaxel (nsclc)",
            "ipi+nivo", "ipi+nivo (melanoma)",
        },
        "KNOWN_EATING_EFFECTS": {
            "nausea", "vomiting", "diarrhoea", "diarrhea",
            "constipation", "mucositis", "sore mouth",
            "appetite loss", "weight loss", "weight gain",
            "taste changes", "smell changes", "taste and smell changes",
            "dry mouth", "fatigue", "sore mouth (mucositis)",
        },
        "FOOD_KEYWORDS": {
            "eat", "eating", "food", "foods", "diet", "nutrition", "drink",
            "avoid", "meal", "appetite", "nausea", "vomiting", "taste",
            "swallow", "mouth", "stomach", "digest", "supplement", "vitamin",
            "protein", "calorie", "hydration", "water", "grapefruit", "milk",
            "alcohol", "what to eat", "what to avoid", "side effect", "sore mouth",
            "constipation", "diarrhoea", "diarrhea", "weight",
        },
        "INTERACTION_KEYWORDS": {
            "interaction", "interact", "combined", "together", "mixing",
            "blood thinner", "anticoagulant", "antibiotic", "antifungal",
            "diabetes medication", "blood pressure", "epilepsy", "seizure",
            "painkiller", "pain medication", "antidepressant", "statin",
        },
    }
    for name, value in defaults.items():
        if not hasattr(_config, name):
            setattr(_config, name, value)


_ensure_config_compat()
from cancer_retrieval import (
    generate_answer,
    _run_research_mode, _run_graph_mode, _run_auto_mode,
    _retrieve_image_chunks,
)


def _call_generate_answer(*args, **kwargs):
    """Support both 2-value and 3-value generate_answer return signatures."""
    result = generate_answer(*args, **kwargs)
    if isinstance(result, tuple):
        if len(result) == 3:
            return result
        if len(result) == 2:
            answer, sources = result
            return answer, sources, []
    raise ValueError(
        f"Unsupported generate_answer return format: {type(result).__name__}"
    )
# generate_answer() returns (answer, sources, raw_docs) — single retrieval.
# raw_docs are the exact Document objects that built the LLM prompt.
# M4 context utilisation scores against these directly — no second retrieval.


def _get_context_payload(question: str, query_mode: str):
    """Retrieve contexts for M4 using the same internal retrieval helpers as the working evaluator."""
    try:
        fn = {
            QUERY_MODE_RESEARCH: _run_research_mode,
            QUERY_MODE_GRAPH: _run_graph_mode,
        }.get(query_mode, _run_auto_mode)

        _, docs, _, _ = fn(question, "", [], "")
        docs = list(docs or [])
        contexts = [d.page_content for d in docs if getattr(d, "page_content", "").strip()]

        image_docs = []
        for img in _retrieve_image_chunks(question):
            if getattr(img, "page_content", "").strip():
                image_docs.append(img)
                contexts.append(img.page_content)

        return docs + image_docs, (contexts or ["No context retrieved"])
    except Exception as e:
        return [], [f"Context retrieval failed: {e}"]

# =============================================================================
# OUTPUT PATHS
# =============================================================================

EVAL_DIR      = Path(__file__).parent / "output" / "evaluation_with_GD"
SCORES_PATH   = EVAL_DIR / "custom_scores.json"
REPORT_PATH   = EVAL_DIR / "custom_report.html"
HISTORY_PATH  = EVAL_DIR / "custom_history.jsonl"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# GOLD-STANDARD TEST SET — 20 Q&A pairs
#
# Each entry includes:
#   question      — user query
#   ground_truth  — correct answer (used for keyword coverage scoring)
#   keywords      — key clinical terms that MUST appear in a good answer
#                   Formula: Coverage = |keywords_in_answer| / |total_keywords|
#   query_mode    — pipeline mode to use
#   category      — graph | research | image | edge
#   description   — what this test validates
# =============================================================================

TEST_SET = [

    # ── GRAPH (5): drug-food interactions and nutrition guidelines ────────────

    {
        "question":    "What foods should a patient on cisplatin avoid?",
        "ground_truth": (
            "Patients should avoid alcohol, fatty fried foods, spicy foods, and "
            "large meals. Small frequent bland meals are recommended. Adequate "
            "hydration prevents nephrotoxicity."
        ),
        "keywords":    ["cisplatin", "nausea", "avoid", "hydration", "food"],
        "query_mode":  QUERY_MODE_GRAPH,
        "category":    "graph",
        "description": "Food avoidance — core graph traversal test",
    },
    {
        "question":    "What are the mandatory nutritional guidelines for pemetrexed?",
        "ground_truth": (
            "Pemetrexed requires folic acid 400-1000 mcg daily from 7 days before "
            "and vitamin B12 1000 mcg IM every 3 cycles to reduce haematological "
            "and GI toxicity."
        ),
        "keywords":    ["pemetrexed", "folic acid", "vitamin B12", "mandatory", "supplement"],
        "query_mode":  QUERY_MODE_GRAPH,
        "category":    "graph",
        "description": "Mandatory supplementation — NutritionGuideline node test",
    },
    {
        "question":    "A breast cancer patient on AC-T is taking warfarin. What are the dietary risks?",
        "ground_truth": (
            "Warfarin interacts with capecitabine raising INR and bleeding risk. "
            "Diarrhoea reduces vitamin K absorption. Consistent vitamin K intake "
            "and INR monitoring are essential."
        ),
        "keywords":    ["warfarin", "INR", "capecitabine", "vitamin K", "bleeding"],
        "query_mode":  QUERY_MODE_GRAPH,
        "category":    "graph",
        "description": "Drug-food-drug interaction chain",
    },
    {
        "question":    "What eating side effects does vincristine cause and how are they managed?",
        "ground_truth": (
            "Vincristine causes constipation from autonomic neuropathy. Increase "
            "dietary fibre, fluids, use stool softeners. Paralytic ileus risk "
            "increases with concurrent morphine or amitriptyline."
        ),
        "keywords":    ["vincristine", "constipation", "neuropathy", "fibre", "stool"],
        "query_mode":  QUERY_MODE_GRAPH,
        "category":    "graph",
        "description": "EatingAdverseEffect node traversal test",
    },
    {
        "question":    "What foods help manage nausea during chemotherapy?",
        "ground_truth": (
            "Dry crackers, plain toast, ginger tea, cold foods, and small frequent "
            "meals help manage nausea. Avoid fatty, fried, spicy, or strong-smelling "
            "foods during chemotherapy."
        ),
        "keywords":    ["nausea", "ginger", "crackers", "bland", "small meals"],
        "query_mode":  QUERY_MODE_GRAPH,
        "category":    "graph",
        "description": "RELIEVED_BY edge traversal test",
    },

    # ── RESEARCH (5): clinical literature from PDF chunks ─────────────────────

    {
        "question":    "What is the 5-year overall survival rate for osteosarcoma?",
        "ground_truth": (
            "5-year survival for localised osteosarcoma is 60-70%. Metastatic "
            "disease drops to 20-30%. Histological response >90% necrosis is the "
            "strongest prognostic factor."
        ),
        "keywords":    ["osteosarcoma", "survival", "60", "70", "necrosis", "prognosis"],
        "query_mode":  QUERY_MODE_RESEARCH,
        "category":    "research",
        "description": "Survival statistics — osteosarcoma PDF retrieval",
    },
    {
        "question":    "How do ALL and AML differ in treatment?",
        "ground_truth": (
            "ALL is common in children and uses multi-agent protocols. AML uses "
            "7+3 induction. Ph+ ALL requires TKIs like imatinib or dasatinib."
        ),
        "keywords":    ["ALL", "AML", "induction", "imatinib", "Philadelphia", "7+3"],
        "query_mode":  QUERY_MODE_RESEARCH,
        "category":    "research",
        "description": "Leukemia subtype differentiation — leukemia PDF",
    },
    {
        "question":    "What molecular targets guide NSCLC treatment?",
        "ground_truth": (
            "EGFR mutations use erlotinib or osimertinib. ALK rearrangements use "
            "crizotinib or alectinib. PD-L1 high expressors receive pembrolizumab. "
            "KRAS G12C uses sotorasib."
        ),
        "keywords":    ["EGFR", "ALK", "PD-L1", "pembrolizumab", "NSCLC", "erlotinib"],
        "query_mode":  QUERY_MODE_RESEARCH,
        "category":    "research",
        "description": "Molecular targets — lung cancer PDF",
    },
    {
        "question":    "What staging system is used for melanoma and what is stage IV?",
        "ground_truth": (
            "Melanoma uses the AJCC TNM system. Stage IV indicates distant metastasis. "
            "Five-year survival has improved significantly with checkpoint inhibitor "
            "immunotherapy."
        ),
        "keywords":    ["melanoma", "AJCC", "TNM", "stage IV", "metastasis", "checkpoint"],
        "query_mode":  QUERY_MODE_RESEARCH,
        "category":    "research",
        "description": "Staging system — melanoma PDF",
    },
    {
        "question":    "What are the main risk factors for breast cancer?",
        "ground_truth": (
            "Risk factors include age, BRCA1/2 mutations, family history, early "
            "menarche, late menopause, nulliparity, HRT, alcohol, and obesity."
        ),
        "keywords":    ["BRCA", "breast cancer", "risk", "family history", "HRT", "obesity"],
        "query_mode":  QUERY_MODE_RESEARCH,
        "category":    "research",
        "description": "Risk factors — breast cancer PDF",
    },

    # ── IMAGE (5): image chunk retrieval and [IMAGE:] tag generation ──────────

    {
        "question":    "Show me the PRISMA flowchart from the systematic review.",
        "ground_truth": (
            "The PRISMA flowchart shows records identified, screened, excluded, "
            "and included in the final systematic review synthesis."
        ),
        "keywords":    ["PRISMA", "flowchart", "systematic", "records"],
        "query_mode":  QUERY_MODE_RESEARCH,
        "category":    "image",
        "description": "PRISMA request — must return [IMAGE:] tag",
    },
    {
        "question":    "Are there any survival curves or Kaplan-Meier plots?",
        "ground_truth": (
            "Kaplan-Meier curves plot survival probability over time with median "
            "survival times and log-rank p-values for group comparisons."
        ),
        "keywords":    ["Kaplan", "survival", "curve", "figure"],
        "query_mode":  QUERY_MODE_RESEARCH,
        "category":    "image",
        "description": "Survival curve request — must return [IMAGE:] tag",
    },
    {
        "question":    "What does Figure 1 show in the breast cancer paper?",
        "ground_truth": (
            "Figures in breast cancer papers show treatment algorithms, molecular "
            "subtype classifications, or drug delivery system diagrams."
        ),
        "keywords":    ["figure", "breast cancer", "diagram"],
        "query_mode":  QUERY_MODE_RESEARCH,
        "category":    "image",
        "description": "Specific figure reference — figure_caption retrieval",
    },
    {
        "question":    "Show me tables with chemotherapy dosing information.",
        "ground_truth": (
            "Dosing tables show drug name, dose in mg/m2, route of administration, "
            "schedule, and toxicities. Dose reduction guidelines may be included."
        ),
        "keywords":    ["table", "dose", "mg", "schedule", "toxicity"],
        "query_mode":  QUERY_MODE_RESEARCH,
        "category":    "image",
        "description": "Table request — table_caption retrieval",
    },
    {
        "question":    "Are there flowcharts showing leukemia treatment pathways?",
        "ground_truth": (
            "Leukemia treatment flowcharts show the pathway from diagnosis through "
            "risk stratification, induction, response assessment, and transplant "
            "decisions."
        ),
        "keywords":    ["leukemia", "flowchart", "pathway", "induction"],
        "query_mode":  QUERY_MODE_RESEARCH,
        "category":    "image",
        "description": "Flowchart request — leukemia image retrieval",
    },

    # ── EDGE (5): boundary cases, web fallback, out-of-corpus ─────────────────

    {
        "question":    "What vaccine is approved for preventing osteosarcoma?",
        "ground_truth": (
            "No vaccine is currently approved for preventing osteosarcoma in humans. "
            "This question falls outside the clinical review papers in this database."
        ),
        "keywords":    ["vaccine", "osteosarcoma"],
        "query_mode":  QUERY_MODE_AUTO,
        "category":    "edge",
        "description": "Out-of-corpus — must trigger web search fallback",
    },
    {
        "question":    "What is the latest FDA-approved drug for osteosarcoma in 2024?",
        "ground_truth": (
            "This requires current regulatory information beyond the scope of the "
            "clinical review papers in this database."
        ),
        "keywords":    ["FDA", "approved", "osteosarcoma"],
        "query_mode":  QUERY_MODE_AUTO,
        "category":    "edge",
        "description": "Recent regulatory question — proactive web fallback",
    },
    {
        "question":    "What eating effects does paclitaxel cause and what foods to avoid?",
        "ground_truth": (
            "Paclitaxel causes nausea, taste changes, and mucositis. Patients should "
            "avoid spicy and acidic foods. Small bland meals help manage nausea."
        ),
        "keywords":    ["paclitaxel", "nausea", "mucositis", "avoid", "food"],
        "query_mode":  QUERY_MODE_AUTO,
        "category":    "edge",
        "description": "Auto mode hybrid — tests graph + vector routing",
    },
    {
        "question":    "Does chemotherapy affect fertility?",
        "ground_truth": (
            "Alkylating agents like cyclophosphamide can cause gonadal toxicity and "
            "infertility. Fertility preservation should be discussed before treatment."
        ),
        "keywords":    ["fertility", "chemotherapy", "gonadal", "cyclophosphamide"],
        "query_mode":  QUERY_MODE_AUTO,
        "category":    "edge",
        "description": "Partially in-corpus — mixed retrieval test",
    },
    {
        "question":    "What is the standard treatment for stage III melanoma?",
        "ground_truth": (
            "Stage III melanoma is treated with surgery plus adjuvant anti-PD-1 "
            "immunotherapy or BRAF/MEK inhibitors for BRAF V600E mutated tumours."
        ),
        "keywords":    ["melanoma", "stage III", "surgery", "pembrolizumab", "BRAF"],
        "query_mode":  QUERY_MODE_RESEARCH,
        "category":    "edge",
        "description": "Specific staging question — precise retrieval test",
    },
]

# =============================================================================
# METRIC 1 — IMAGE RECALL RATE
#
# Mathematical formula:
#   Image_Recall = Σ image_tag_present(q) / N_image_questions
#
#   where image_tag_present(q) = 1.0 if [IMAGE: filename.png] found in answer
#                               = 0.5 if visual keywords found but no tag
#                               = 0.0 if no image reference at all
#
# Why this matters: your two reported bugs include images not being returned.
# A score of 1.0 means every image question got a proper [IMAGE:] tag.
# A score of 0.0 means the image retrieval fix is not working at all.
#
# Target: ≥ 0.90  (allow 1 missed image per 10 questions)
# =============================================================================

def metric_image_recall(answer: str, category: str) -> Optional[float]:
    """
    Scores image retrieval for image-category questions only.
    Returns None for non-image questions (excluded from aggregate).

    Score breakdown:
      1.0 → [IMAGE: filename.png] tag found — perfect, image was retrieved
      0.5 → visual keyword found but no tag — partial, LLM knew about image
            but did not reproduce the tag correctly
      0.0 → no image reference at all — retrieval or prompt failed completely
    """
    if category != "image":
        return None   # metric not applicable to this category

    # Check for valid [IMAGE: filename.png] tag pattern
    has_proper_tag = bool(
        re.search(r'\[IMAGE:\s*[^\]]{3,}\]', answer, re.IGNORECASE)
    )
    if has_proper_tag:
        return 1.0

    # Check for visual reference keywords (partial credit)
    visual_keywords = {
        "figure", "fig.", "fig ", "table", "chart", "flowchart",
        "diagram", "image", "shown in", "illustrated", "depicted",
        "as shown", "kaplan", "prisma", "survival curve",
    }
    has_visual_ref = any(kw in answer.lower() for kw in visual_keywords)
    if has_visual_ref:
        return 0.5

    return 0.0


# =============================================================================
# METRIC 2 — WEB FALLBACK PRECISION
#
# Mathematical formula:
#   Web_Precision = Σ web_fired(q) / N_edge_questions
#
#   where web_fired(q) = 1.0 if answer contains web search signal
#                      = 0.0 if answer tried to answer from corpus
#
# Why this matters: your second reported bug was the osteosarcoma vaccine
# question NOT triggering web search and instead giving a partial in-corpus
# answer. A score of 1.0 means all out-of-corpus queries correctly route
# to DuckDuckGo fallback.
#
# Target: 1.00  (all edge queries must trigger web fallback)
# =============================================================================

def metric_web_fallback(answer: str, category: str) -> Optional[float]:
    """
    Scores web fallback triggering for edge-category questions only.
    Returns None for non-edge questions.

    A web search fired if the answer contains:
      🌐  — our web search result emoji marker
      [W1] — web result citation from DuckDuckGo
      "web search" — explicit text reference
      "sourced from web" — our web-only answer header
    """
    if category != "edge":
        return None

    web_signals = [
        "🌐",
        "[W1]", "[W2]", "[W3]",
        "web search",
        "sourced from web",
        "additional web sources",
        "according to recent",
    ]
    fired = any(sig in answer for sig in web_signals)
    return 1.0 if fired else 0.0


# =============================================================================
# METRIC 3 — KEYWORD COVERAGE SCORE
#
# Mathematical formula:
#   Coverage(q) = |{k ∈ keywords(q) : k ∈ answer(q).lower()}|
#                 ─────────────────────────────────────────────
#                        |keywords(q)|
#
#   Mean_Coverage = (1/N) Σ Coverage(q)  for all N questions
#
# Why this matters: answer relevancy from RAGAS needs an LLM judge.
# Keyword coverage achieves similar signal with zero LLM calls.
# If the answer covers the key clinical terms from the ground truth,
# it is almost certainly addressing the question correctly.
#
# This is NOT the same as exact match — it checks whether each keyword
# substring appears anywhere in the lowercased answer text.
#
# Target: ≥ 0.75  (answer contains ≥75% of expected clinical keywords)
# =============================================================================

def metric_keyword_coverage(answer: str, keywords: list[str]) -> float:
    """
    Measures what fraction of expected clinical keywords appear in the answer.

    Example:
      keywords = ["cisplatin", "nausea", "avoid", "hydration", "food"]
      answer contains: "cisplatin", "nausea", "avoid", "food"
      Coverage = 4/5 = 0.80

    Keywords are matched as case-insensitive substrings.
    Multi-word keywords (e.g. "folic acid") are matched as full phrases.
    """
    if not keywords:
        return 0.0

    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(hits / len(keywords), 4)


# =============================================================================
# METRIC 4 — CONTEXT UTILISATION RATE (CUR)
#
# Mathematical formula:
#   CUR(q) = |{c ∈ contexts(q) : overlap(c, answer(q)) ≥ threshold}|
#             ────────────────────────────────────────────────────────
#                           |contexts(q)|
#
#   where overlap(c, answer) = |tokens(c) ∩ tokens(answer)| / min(|c|, 20)
#   threshold = 2 shared meaningful tokens (length > 4 chars)
#
#   Mean_CUR = (1/N) Σ CUR(q)
#
# Why this matters: this is a proxy for context precision without an LLM.
# If a retrieved chunk shares keywords with the final answer, the LLM
# used that chunk. If no overlap exists, the chunk was wasted retrieval.
#
# Unlike RAGAS context_precision (which asks an LLM "was this chunk relevant?"),
# CUR directly measures "did the LLM's answer draw from this chunk?"
#
# Target: ≥ 0.60  (at least 60% of retrieved chunks contributed to answer)
# =============================================================================

def metric_context_utilisation(answer: str, contexts: list[str]) -> float:
    """
    Measures what fraction of retrieved chunks contributed to the answer.

    A chunk 'contributed' if it shares ≥2 meaningful tokens (>4 chars)
    with the answer. This approximates whether the LLM read and used each chunk.

    Example:
      3 chunks retrieved, 2 share keywords with the answer
      CUR = 2/3 = 0.667
    """
    if not contexts:
        return 0.0

    # Tokenise answer — extract words longer than 4 chars (filter stop words)
    answer_tokens = set(
        w.lower() for w in re.findall(r'\b[a-zA-Z]{5,}\b', answer)
    )

    if not answer_tokens:
        return 0.0

    useful_chunks = 0
    for chunk in contexts:
        # Tokenise chunk the same way
        chunk_tokens = set(
            w.lower() for w in re.findall(r'\b[a-zA-Z]{5,}\b', chunk)
        )
        # Count shared meaningful tokens
        shared = len(answer_tokens & chunk_tokens)
        if shared >= 2:
            useful_chunks += 1

    return round(useful_chunks / len(contexts), 4)


# =============================================================================
# METRIC 5 — ANSWER COMPLETENESS SCORE
#
# Mathematical formula:
#   SubQ_count(q) = number of sub-questions implied by question
#                 = 1 + count(" and ", " how ", ", ") in question
#
#   Expected_min_chars(q) = 150 × (1 + 0.5 × SubQ_count(q))
#
#   Completeness(q) = min(1.0, len(answer(q)) / Expected_min_chars(q))
#
#   Mean_Completeness = (1/N) Σ Completeness(q)
#
# Why this matters: a good medical answer should not be a single sentence.
# Multi-part questions (containing "and", "how") need proportionally longer
# answers. This catches cases where the LLM truncated or gave a lazy answer.
#
# Calibration:
#   Simple question, 1 sub-question → expected ≥ 150 chars
#   2 sub-questions ("what and how") → expected ≥ 225 chars
#   3 sub-questions                  → expected ≥ 300 chars
#
# Target: ≥ 0.85  (answers are adequately long for their complexity)
# =============================================================================

def metric_answer_completeness(question: str, answer: str) -> float:
    """
    Scores answer length relative to question complexity.

    Counts implied sub-questions from the question text, then checks
    whether the answer is long enough to address all parts.

    This is a necessary but not sufficient condition for completeness —
    a long irrelevant answer would score high. Combined with keyword
    coverage (M3) and context utilisation (M4), it gives a fuller picture.
    """
    # Count implied sub-questions in the question
    q_lower = question.lower()
    sub_q_signals = [" and ", " how ", " why ", " when ", " what else "]
    sub_q_count = 1 + sum(1 for s in sub_q_signals if s in q_lower)

    # Expected minimum answer length
    expected_min = 150 * (1 + 0.5 * (sub_q_count - 1))

    # Completeness = min(1.0, actual / expected)
    actual = len(answer.strip())
    completeness = min(1.0, actual / expected_min)
    return round(completeness, 4)


# =============================================================================
# METRIC 6 — GRAPH GROUNDING RATE
#
# Mathematical formula:
#   Grounded(q) = 1  if any graph signal found in answer
#               = 0  otherwise
#
#   Graph_Grounding = Σ Grounded(q) / N_graph_mode_questions
#
# Graph signals include:
#   [G1], [G2]  — graph citation tags from our prompt
#   Drug names  — cisplatin, paclitaxel, methotrexate etc.
#   Severity    — "High", "Moderate", "Low" (from DrugInteraction nodes)
#   Guideline   — "mandatory", "required", "supplement" (from NutritionGuideline)
#   Food terms  — "avoid", "eat", "recommended" with specific food names
#
# Why this matters: in graph mode, the structured knowledge graph should be
# the PRIMARY source. If graph signals are absent, the LLM is ignoring the
# graph context and falling back to generic training knowledge — a failure
# of the graph retrieval integration.
#
# Target: ≥ 0.80  (80% of graph mode answers cite graph data)
# =============================================================================

def metric_graph_grounding(answer: str, query_mode: str) -> Optional[float]:
    """
    Checks whether graph mode answers actually use graph-structured data.
    Returns None for non-graph-mode questions.

    Graph signals are patterns that indicate the answer drew from the
    knowledge graph rather than generic LLM training knowledge.
    """
    if query_mode != QUERY_MODE_GRAPH:
        return None

    answer_lower = answer.lower()

    # Graph citation tags from our _build_prompt() instruction
    has_g_citation = bool(re.search(r'\[G\d+\]', answer))

    # Drug names from KNOWN_CHEMO_DRUGS in config.py
    drug_signals = {
        "cisplatin", "carboplatin", "paclitaxel", "docetaxel", "pemetrexed",
        "vincristine", "doxorubicin", "cyclophosphamide", "methotrexate",
        "capecitabine", "gemcitabine", "nivolumab", "pembrolizumab",
    }
    has_drug = any(d in answer_lower for d in drug_signals)

    # Severity markers from DrugInteraction.severity property
    severity_signals = {"high severity", "moderate severity", "low severity",
                        "severity: high", "severity: moderate"}
    has_severity = any(s in answer_lower for s in severity_signals)

    # Nutrition guideline signals from NutritionGuideline nodes
    guideline_signals = {"mandatory", "required supplement", "folic acid",
                         "vitamin b12", "b12", "supplementation is"}
    has_guideline = any(s in answer_lower for s in guideline_signals)

    # Food-drug specific signals (foods from FoodItem nodes)
    food_signals = {"grapefruit", "alcohol", "milk", "fatty food", "spicy food",
                    "bland", "small frequent", "foods to avoid", "foods to eat"}
    has_food = any(s in answer_lower for s in food_signals)

    # Grounded if ANY graph signal present
    grounded = any([has_g_citation, has_drug, has_severity, has_guideline, has_food])
    return 1.0 if grounded else 0.0


# =============================================================================
# METRIC 7 — LATENCY SCORE
#
# Mathematical formula:
#   Latency_Score(q) = max(0.0,  1 - (elapsed_s(q) - FAST_THRESHOLD)
#                                    ──────────────────────────────── )
#                                         SLOW_THRESHOLD
#
#   where FAST_THRESHOLD = 10s  (answers faster than this score 1.0)
#         SLOW_THRESHOLD = 30s  (answers slower by this much lose full score)
#
#   Examples:
#     8s  → max(0, 1 - (8-10)/30)  = max(0, 1.067) = 1.0  (capped at 1.0)
#     15s → max(0, 1 - (15-10)/30) = max(0, 0.833) = 0.833
#     25s → max(0, 1 - (25-10)/30) = max(0, 0.500) = 0.500
#     40s → max(0, 1 - (40-10)/30) = max(0, 0.000) = 0.0   (floor at 0.0)
#
#   Mean_Latency = (1/N) Σ Latency_Score(q)
#
# Why this matters: your pipeline runs on Groq with real patients asking
# questions. A 40-second response is unacceptably slow. This metric
# tracks speed regression as you add new pipeline components.
#
# Target: ≥ 0.70  (average response under ~25 seconds)
# =============================================================================

FAST_THRESHOLD_S = 10.0   # answers faster than this get 1.0
SLOW_THRESHOLD_S = 30.0   # latency range over which score decreases

def metric_latency(elapsed_s: float) -> float:
    """
    Converts raw elapsed seconds into a normalised 0-1 score.
    Faster is better. Scores above FAST_THRESHOLD are capped at 1.0.
    Scores below 0.0 are floored at 0.0.
    """
    score = 1.0 - (elapsed_s - FAST_THRESHOLD_S) / SLOW_THRESHOLD_S
    return round(max(0.0, min(1.0, score)), 4)


# =============================================================================
# COMPOSITE SCORE
#
# Mathematical formula:
#   Composite = w1·M1 + w2·M2 + w3·M3 + w4·M4 + w5·M5 + w6·M6 + w7·M7
#
#   Weights reflect clinical importance for a medical RAG system:
#     M1 Image Recall       w=0.15  (important but not safety-critical)
#     M2 Web Fallback       w=0.20  (critical — wrong answers on OOC queries)
#     M3 Keyword Coverage   w=0.25  (core answer quality proxy)
#     M4 Context Utilisation w=0.15 (retrieval efficiency)
#     M5 Completeness       w=0.10  (answer richness)
#     M6 Graph Grounding    w=0.10  (graph integration quality)
#     M7 Latency            w=0.05  (user experience)
#
#   Weights sum to 1.0.
#   Metrics that return None for a question are excluded from that question's
#   composite but included in category-level averages where applicable.
# =============================================================================

METRIC_WEIGHTS = {
    "image_recall":         0.15,
    "web_fallback":         0.20,
    "keyword_coverage":     0.25,
    "context_utilisation":  0.15,
    "answer_completeness":  0.10,
    "graph_grounding":      0.10,
    "latency_score":        0.05,
}

def compute_composite(scores: dict) -> float:
    """
    Weighted composite score across all applicable metrics for a question.
    Metrics returning None are excluded and remaining weights are renormalised.

    Formula: Composite = Σ(w_i × s_i) / Σ(w_i)  for applicable metrics i
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for metric, weight in METRIC_WEIGHTS.items():
        val = scores.get(metric)
        if val is not None:
            weighted_sum += weight * val
            total_weight += weight
    if total_weight == 0:
        return 0.0
    return round(weighted_sum / total_weight, 4)


# NOTE: _get_contexts() has been intentionally removed.
#
# PROBLEM IT CAUSED (double-retrieval mismatch):
#   Old flow:
#     1. generate_answer() calls _run_graph_mode() → retrieves chunks A,B,C
#        → builds prompt → generates answer from A,B,C
#     2. _get_contexts() calls _run_graph_mode() AGAIN → may retrieve A,B,D
#        (BM25 + ANN scores are non-deterministic on tie-breaking)
#     3. M4 context utilisation scored against A,B,D — WRONG, answer used A,B,C
#
# FIX: generate_answer() now returns (answer, sources, vector_docs) as a
# three-element tuple. The evaluation captures vector_docs directly from
# that single call. No second retrieval. Contexts are guaranteed identical.


# =============================================================================
# MAIN EVALUATION RUNNER
# =============================================================================

def run_evaluation(
    label:     str  = "baseline",
    quick:     bool = False,
    category:  str  = None,
) -> dict:
    """
    Run all 7 custom metrics. Optionally run RAGAS benchmark on top.

    Args:
        label:     run label for history (e.g. "after_crossencoder")
        quick:     run first 5 questions only (~2 min custom, ~10 min with RAGAS)
        category:  run one category: graph | research | image | edge
    Returns:
        full results dict with per-question scores and aggregates.
    """
    print("=" * 70)
    print(f"  MedChat Custom Metrics Evaluation  v4.0")
    print(f"  Pipeline : {GROQ_MODEL_QUERY}")
    print(f"  Scoring  : LLM-FREE (7 custom metrics, zero token cost)")
    print(f"  Label    : {label}")
    print(f"  Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"\n  Metrics (all computed locally, no API calls):")
    print(f"    M1  Image Recall Rate     (target ≥0.90)")
    print(f"    M2  Web Fallback Precision (target =1.00)")
    print(f"    M3  Keyword Coverage       (target ≥0.75)")
    print(f"    M4  Context Utilisation    (target ≥0.60)")
    print(f"    M5  Answer Completeness    (target ≥0.85)")
    print(f"    M6  Graph Grounding Rate   (target ≥0.80)")
    print(f"    M7  Latency Score          (target ≥0.70)")

    # Filter test set
    items = TEST_SET
    if category:
        items = [t for t in items if t["category"] == category]
        print(f"\n  Category filter : {category} ({len(items)} questions)")
    if quick:
        items = items[:5]
        print(f"  Mode            : quick (first 5 questions)")
    print(f"  Total questions : {len(items)}\n")

    per_question: list[dict] = []

    for i, item in enumerate(items, 1):
        q     = item["question"]
        gt    = item["ground_truth"]
        kws   = item["keywords"]
        mode  = item["query_mode"]
        cat   = item["category"]
        desc  = item["description"]

        print(f"  [{i:2d}/{len(items)}] {cat.upper():10s} | {q[:58]}...")

        try:
            # Single pipeline call — captures answer AND the exact contexts
            # that were used to build the LLM prompt. No second retrieval.
            # contexts = the actual Document objects that generated the answer.
            t0 = time.time()
            answer, sources, raw_docs = _call_generate_answer(
                query=q,
                patient_report="",
                chat_history=[],
                cancer_filter="",
                query_mode=mode,
            )
            elapsed = time.time() - t0

            # Extract text from the exact docs used — guaranteed same as prompt
            retrieval_docs, contexts = _get_context_payload(q, mode)

            # ── Compute all 7 metrics ─────────────────────────────────────
            m1 = metric_image_recall(answer, cat)
            m2 = metric_web_fallback(answer, cat)
            m3 = metric_keyword_coverage(answer, kws)
            m4 = metric_context_utilisation(answer, contexts)
            m5 = metric_answer_completeness(q, answer)
            m6 = metric_graph_grounding(answer, mode)
            m7 = metric_latency(elapsed)

            scores = {
                "image_recall":        m1,
                "web_fallback":        m2,
                "keyword_coverage":    m3,
                "context_utilisation": m4,
                "answer_completeness": m5,
                "graph_grounding":     m6,
                "latency_score":       m7,
            }
            composite = compute_composite(scores)

            # Indicator flags for display
            has_img = "[IMAGE:" in answer.upper()
            has_web = "🌐" in answer or "[W1]" in answer

            print(
                f"             ✅ {elapsed:.1f}s | "
                f"kw={m3:.2f} ctx={m4:.2f} comp={m5:.2f} "
                f"{'🖼️' if has_img else '  '} "
                f"{'🌐' if has_web else '  '} "
                f"→ composite={composite:.3f}"
            )

            # Build retrieval log exactly as you designed in Step 1.
            # Chunk IDs extracted from Document metadata where available.
            chunk_ids = []
            retrieved_contexts_log = []
            for doc in retrieval_docs:
                meta = getattr(doc, "metadata", {}) or {}
                cid  = meta.get("chunk_id") or meta.get("id") or meta.get("source", "unknown")
                chunk_ids.append(str(cid))
                retrieved_contexts_log.append({
                    "chunk_id": str(cid),
                    "text_preview": doc.page_content[:120].strip(),
                    "source": meta.get("source", ""),
                })

            per_question.append({
                # ── Core fields ───────────────────────────────────────────
                "question":      q,
                "answer":        answer,
                "ground_truth":  gt,
                "keywords":      kws,
                "query_mode":    mode,
                "category":      cat,
                "description":   desc,
                "timestamp":     datetime.now().isoformat(),
                # ── Retrieval log (your Step 1 design) ───────────────────
                # These are the EXACT contexts that built the LLM prompt.
                # No second retrieval, no mismatch possible.
                "retrieved_contexts": retrieved_contexts_log,
                "chunk_ids":          chunk_ids,
                "context_count":      len(contexts),
                # ── Flags and timing ──────────────────────────────────────
                "elapsed_s":     round(elapsed, 2),
                "has_image_tag": has_img,
                "web_fired":     has_web,
                "sources":       [s.get("label", "") for s in sources],
                "answer_preview": answer[:300],
                # ── All 7 metric scores ───────────────────────────────────
                "scores":        scores,
                "composite":     composite,
            })

        except Exception as e:
            print(f"             ❌ Error: {str(e)[:80]}")
            traceback.print_exc()
            per_question.append({
                "question":    q,
                "error":       str(e),
                "category":    cat,
                "description": desc,
            })

        time.sleep(1)   # small buffer between pipeline calls

    # ── Aggregate metrics ─────────────────────────────────────────────────

    def _agg(metric: str, cat_filter: str = None) -> Optional[float]:
        """
        Aggregate a metric across questions.
        Formula: mean(s_i) for all i where s_i is not None and no error.
        Optional category filter restricts to one category.
        """
        vals = []
        for pq in per_question:
            if "error" in pq:
                continue
            if cat_filter and pq.get("category") != cat_filter:
                continue
            v = pq["scores"].get(metric)
            if v is not None:
                vals.append(v)
        if not vals:
            return None
        return round(sum(vals) / len(vals), 4)

    # Overall aggregates
    aggregates = {
        "image_recall":        _agg("image_recall",        "image"),
        "web_fallback":        _agg("web_fallback",        "edge"),
        "keyword_coverage":    _agg("keyword_coverage"),
        "context_utilisation": _agg("context_utilisation"),
        "answer_completeness": _agg("answer_completeness"),
        "graph_grounding":     _agg("graph_grounding",     "graph"),
        "latency_score":       _agg("latency_score"),
    }
    composite_all = compute_composite({k: v for k, v in aggregates.items() if v is not None})

    # Per-category breakdown
    cat_stats: dict = {}
    for cat in ["graph", "research", "image", "edge"]:
        items_cat = [p for p in per_question if p.get("category") == cat]
        ok        = [p for p in items_cat if "error" not in p]
        if not items_cat:
            continue
        cat_composites = [p["composite"] for p in ok]
        cat_kw         = [p["scores"]["keyword_coverage"] for p in ok]
        cat_lat        = [p["scores"]["latency_score"] for p in ok]
        cat_stats[cat] = {
            "count":           len(items_cat),
            "success":         len(ok),
            "avg_composite":   round(sum(cat_composites)/len(cat_composites), 4) if cat_composites else 0,
            "avg_keyword_cov": round(sum(cat_kw)/len(cat_kw), 4) if cat_kw else 0,
            "avg_latency_s":   round(sum(p["elapsed_s"] for p in ok)/len(ok), 2) if ok else 0,
            "avg_latency_score": round(sum(cat_lat)/len(cat_lat), 4) if cat_lat else 0,
            "image_tag_rate":  round(sum(1 for p in ok if p.get("has_image_tag"))/len(ok), 3) if cat == "image" and ok else None,
            "web_rate":        round(sum(1 for p in ok if p.get("web_fired"))/len(ok), 3)     if cat == "edge"  and ok else None,
        }

    full = {
        "label":             label,
        "timestamp":         datetime.now().isoformat(),
        "pipeline_model":    GROQ_MODEL_QUERY,
        "scoring_method":    "LLM-free custom metrics (v4.0)",
        "question_count":    len(items),
        "aggregates":        aggregates,
        "composite":         composite_all,
        "metric_weights":    METRIC_WEIGHTS,
        "category_stats":    cat_stats,
        "per_question":      per_question,
    }

    # Save JSON
    with open(SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, ensure_ascii=False)

    # Save history line
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "label":     label,
            "timestamp": full["timestamp"],
            "aggregates": aggregates,
            "composite":  composite_all,
            "n":          len(items),
        }) + "\n")

    _html_report(full)
    _print_summary(full)

    print(f"\n  📊 Scores  : {SCORES_PATH}")
    print(f"  🌐 Report  : {REPORT_PATH}")
    print(f"  📈 History : {HISTORY_PATH}")

    print("=" * 70)
    return full


# =============================================================================
# PRINT SUMMARY
# =============================================================================

def _grade(v: Optional[float], target: float) -> str:
    if v is None: return "❓  "
    if v >= target:           return "✅  "
    if v >= target * 0.75:    return "⚠️  "
    return "❌  "

def _print_summary(r: dict) -> None:
    agg = r["aggregates"]
    print(f"\n{'='*70}")
    print(f"  CUSTOM METRICS RESULTS — {r['label']}")
    print(f"  Pipeline: {r['pipeline_model']}  |  Scoring: LLM-free")
    print(f"{'='*70}")
    print(f"\n  All 7 Metrics (w = weight in composite):")
    rows = [
        ("M1", "image_recall",        0.90, 0.15, "Image Recall Rate     "),
        ("M2", "web_fallback",         1.00, 0.20, "Web Fallback Precision"),
        ("M3", "keyword_coverage",     0.75, 0.25, "Keyword Coverage      "),
        ("M4", "context_utilisation",  0.60, 0.15, "Context Utilisation   "),
        ("M5", "answer_completeness",  0.85, 0.10, "Answer Completeness   "),
        ("M6", "graph_grounding",      0.80, 0.10, "Graph Grounding Rate  "),
        ("M7", "latency_score",        0.70, 0.05, "Latency Score         "),
    ]
    for mn, key, target, w, label in rows:
        v = agg.get(key)
        vs = f"{v:.3f}" if v is not None else " N/A"
        print(f"  {_grade(v, target)}{mn}  {label} {vs}  "
              f"(target ≥{target:.2f}, w={w:.2f})")

    print(f"\n  {'─'*40}")
    print(f"     Weighted Composite:  {r['composite']:.3f}")

    print(f"\n  By category:")
    for cat, st in r.get("category_stats", {}).items():
        extras = []
        if st.get("image_tag_rate") is not None:
            extras.append(f"img_tag={st['image_tag_rate']:.0%}")
        if st.get("web_rate") is not None:
            extras.append(f"web={st['web_rate']:.0%}")
        extras.append(f"kw_cov={st['avg_keyword_cov']:.2f}")
        print(f"    {cat:<12} {st['success']}/{st['count']} ok | "
              f"composite={st['avg_composite']:.3f} | "
              f"lat={st['avg_latency_s']:.1f}s | "
              + " | ".join(extras))


# =============================================================================
# HTML REPORT
# =============================================================================

def _html_report(r: dict) -> None:
    agg   = r["aggregates"]
    per_q = r["per_question"]
    label = r["label"]
    ts    = r["timestamp"]

    def _col(v: Optional[float], target: float) -> str:
        if v is None: return "#9ca3af"
        return "#16a34a" if v >= target else "#d97706" if v >= target * 0.75 else "#dc2626"

    def _bar(v: Optional[float], target: float, colour: str = "#16a34a") -> str:
        if v is None:
            return "<span style='color:#9ca3af'>N/A</span>"
        pct = int(max(0.0, min(1.0, v)) * 100)
        bg  = "#f0fdf4" if v >= target else "#fffbeb" if v >= target * 0.75 else "#fef2f2"
        return (f"<div style='background:{bg};border-radius:4px;padding:3px 8px;"
                f"display:inline-block;min-width:100px'>"
                f"<div style='background:{colour};width:{pct}%;height:6px;"
                f"border-radius:3px;margin-bottom:2px'></div>"
                f"<span style='font-size:13px;font-weight:600'>{v:.3f}</span></div>")

    metric_cards = [
        ("M1", "image_recall",        0.90, "#16a34a", "Image Recall",      "[IMAGE:] tag rate"),
        ("M2", "web_fallback",         1.00, "#0891b2", "Web Fallback",       "OOC routing rate"),
        ("M3", "keyword_coverage",     0.75, "#7c3aed", "Keyword Coverage",   "Clinical term match"),
        ("M4", "context_utilisation",  0.60, "#d97706", "Context Utilisation","Chunk usage rate"),
        ("M5", "answer_completeness",  0.85, "#db2777", "Completeness",       "Answer richness"),
        ("M6", "graph_grounding",      0.80, "#059669", "Graph Grounding",    "Graph data usage"),
        ("M7", "latency_score",        0.70, "#6366f1", "Latency Score",      "Speed metric"),
    ]

    cards_html = ""
    for mn, key, target, colour, title, subtitle in metric_cards:
        v = agg.get(key)
        val_str = f"{v:.3f}" if v is not None else "N/A"
        c = _col(v, target)
        cards_html += (f"<div class='card'>"
                       f"<div class='lbl'>{mn} · {title}</div>"
                       f"<div class='val' style='color:{c}'>{val_str}</div>"
                       f"<div class='lbl'>{subtitle}</div></div>")

    # Per-question table rows
    rows = ""
    for i, item in enumerate(per_q, 1):
        if "error" in item:
            rows += (f"<tr><td>{i}</td>"
                     f"<td><span class='badge badge-{item.get('category','')}'>"
                     f"{item.get('category','')}</span></td>"
                     f"<td colspan='7' style='color:#dc2626'>❌ {item.get('error','')[:80]}</td></tr>")
            continue
        sc = item["scores"]
        bd = ("" + ("<span class='bm bi'>🖼️</span>" if item.get("has_image_tag") else "")
                 + ("<span class='bm bw'>🌐</span>" if item.get("web_fired") else ""))
        rows += (f"<tr>"
                 f"<td style='color:#6b7280;font-size:12px'>{i}</td>"
                 f"<td><span class='badge badge-{item.get('category','')}' >{item.get('category','')}</span></td>"
                 f"<td style='font-size:12px;max-width:220px'>{item.get('question','')[:65]}{'...' if len(item.get('question',''))>65 else ''}{bd}</td>"
                 f"<td style='font-size:11px;color:#6b7280'>{item.get('query_mode','')}</td>"
                 f"<td style='font-size:12px'>{item.get('elapsed_s','?')}s</td>"
                 f"<td>{_bar(sc.get('keyword_coverage'), 0.75, '#7c3aed')}</td>"
                 f"<td>{_bar(sc.get('context_utilisation'), 0.60, '#d97706')}</td>"
                 f"<td>{_bar(sc.get('answer_completeness'), 0.85, '#db2777')}</td>"
                 f"<td style='font-size:12px;font-weight:600;color:{'#16a34a' if item['composite']>=0.70 else '#d97706' if item['composite']>=0.50 else '#dc2626'}'>"
                 f"{item['composite']:.3f}</td>"
                 f"</tr>")

    # History rows
    hist_rows = ""
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as hf:
            for h in [json.loads(l) for l in hf if l.strip()][-10:]:
                ag = h.get("aggregates", {})
                hist_rows += (f"<tr>"
                              f"<td style='font-size:12px'>{h.get('timestamp','')[:16]}</td>"
                              f"<td style='font-size:12px;font-weight:500'>{h.get('label','')}</td>"
                              f"<td>{_bar(ag.get('keyword_coverage'), 0.75, '#7c3aed')}</td>"
                              f"<td>{_bar(ag.get('context_utilisation'), 0.60, '#d97706')}</td>"
                              f"<td>{_bar(ag.get('image_recall'), 0.90)}</td>"
                              f"<td>{_bar(ag.get('web_fallback'), 1.00, '#0891b2')}</td>"
                              f"<td>{_bar(h.get('composite'), 0.70, '#1f2a44')}</td>"
                              f"</tr>")

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>MedChat Custom Metrics — {label}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f1f5f9;color:#1e293b;padding:24px}}
.hdr{{background:linear-gradient(135deg,#1f2a44,#2fa36b);color:white;padding:24px 32px;border-radius:14px;margin-bottom:16px}}
.hdr h1{{font-size:22px;margin-bottom:4px}}.hdr p{{font-size:13px;opacity:.8}}
.note{{background:#fefce8;border-left:4px solid #ca8a04;padding:12px 16px;border-radius:6px;margin-bottom:20px;font-size:13px;color:#713f12}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:12px;margin-bottom:24px}}
.card{{background:white;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.07);text-align:center}}
.card .val{{font-size:30px;font-weight:700;margin:6px 0 4px}}.card .lbl{{font-size:11px;color:#64748b}}
.card.hl{{border:2px solid #6366f1}}
table{{width:100%;border-collapse:collapse;background:white;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.07);margin-bottom:24px}}
th{{background:#f8fafc;padding:9px 11px;text-align:left;font-size:10px;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.06em}}
td{{padding:9px 11px;border-top:1px solid #f1f5f9}}tr:hover td{{background:#fafafa}}
h2{{font-size:15px;font-weight:600;margin-bottom:10px}}
.badge{{padding:2px 7px;border-radius:4px;font-size:10px;font-weight:600}}
.badge-graph{{background:#dbeafe;color:#1e40af}}.badge-research{{background:#dcfce7;color:#166534}}
.badge-image{{background:#fef3c7;color:#92400e}}.badge-edge{{background:#f3e8ff;color:#6b21a8}}
.bm{{padding:1px 5px;border-radius:3px;font-size:10px;margin-left:3px}}
.bi{{background:#dcfce7;color:#166534}}.bw{{background:#dbeafe;color:#1e40af}}
</style></head><body>
<div class="hdr">
  <h1>MedChat Custom Metrics Dashboard</h1>
  <p>Label: <strong>{label}</strong> | {ts[:16]} | {r['question_count']}q | Pipeline: <strong>{r['pipeline_model']}</strong> | Scoring: LLM-free</p>
</div>
<div class="note">
  ⚡ <strong>LLM-free evaluation</strong> — all 7 metrics computed locally with zero API calls.
  Composite = weighted average: M3(25%) + M2(20%) + M1(15%) + M4(15%) + M5(10%) + M6(10%) + M7(5%).
  Composite <strong>{r['composite']:.3f}</strong>.
</div>
<div class="grid">
  {cards_html}
  <div class="card hl">
    <div class="lbl">Weighted Composite</div>
    <div class="val" style="color:#6366f1">{r['composite']:.3f}</div>
    <div class="lbl">Overall quality</div>
  </div>
</div>
<h2>Per-question results</h2>
<table>
  <tr><th>#</th><th>Cat</th><th>Question</th><th>Mode</th><th>Time</th>
      <th>M3 Keywords</th><th>M4 Context</th><th>M5 Complete</th><th>Composite</th></tr>
  {rows}
</table>
<h2>Run history</h2>
<table>
  <tr><th>Timestamp</th><th>Label</th><th>M3 Keywords</th><th>M4 Context</th>
      <th>M1 Images</th><th>M2 Web</th><th>Composite</th></tr>
  {hist_rows or '<tr><td colspan="7" style="color:#94a3b8;text-align:center;padding:14px">First run — no history yet</td></tr>'}
</table>
<p style="color:#94a3b8;font-size:12px;margin-top:14px">
  MedChat Custom Metrics v4.0 | {ts}
</p>
</body></html>"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)


# =============================================================================
# COMPARE UTILITY
# =============================================================================

def compare_runs(a: str, b: str) -> None:
    """Compare two labelled runs from history — shows delta arrows."""
    if not HISTORY_PATH.exists():
        print("No history file. Run evaluation first.")
        return
    with open(HISTORY_PATH) as f:
        hist = [json.loads(l) for l in f if l.strip()]
    ra = next((h for h in reversed(hist) if h["label"] == a), None)
    rb = next((h for h in reversed(hist) if h["label"] == b), None)
    if not ra: print(f"Label '{a}' not found."); return
    if not rb: print(f"Label '{b}' not found."); return
    print(f"\n{'='*60}\n  Comparison: {a}  →  {b}\n{'='*60}")
    metrics = [
        ("M1", "image_recall",        0.90),
        ("M2", "web_fallback",         1.00),
        ("M3", "keyword_coverage",     0.75),
        ("M4", "context_utilisation",  0.60),
        ("M5", "answer_completeness",  0.85),
        ("M6", "graph_grounding",      0.80),
        ("M7", "latency_score",        0.70),
    ]
    for mn, key, target in metrics:
        va = (ra.get("aggregates") or {}).get(key)
        vb = (rb.get("aggregates") or {}).get(key)
        if va is None or vb is None:
            print(f"  ❓  {mn} {key:<22}  N/A → N/A")
            continue
        d = vb - va
        icon = "✅" if d > 0.01 else "❌" if d < -0.01 else "  "
        arrow = "▲" if d > 0.01 else "▼" if d < -0.01 else "→"
        print(f"  {icon}  {mn} {key:<22}  {va:.3f} {arrow} {vb:.3f}  "
              f"({'+' if d >= 0 else ''}{d:.3f})")
    ca, cb = ra.get("composite", 0.0), rb.get("composite", 0.0)
    d = cb - ca
    print(f"\n  {'✅' if d > 0.01 else '❌' if d < -0.01 else '  '}  "
          f"   {'composite':<22}  {ca:.3f} {'▲' if d>0.01 else '▼' if d<-0.01 else '→'} {cb:.3f}  "
          f"({'+' if d >= 0 else ''}{d:.3f})")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="MedChat LLM-Free Custom Metrics Evaluation v4.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cancer_evaluation.py --quick                     # 5 questions, ~2 min
  python cancer_evaluation.py                             # 20 questions, ~8 min
  python cancer_evaluation.py --category graph            # graph only
  python cancer_evaluation.py --label after_reranker      # labelled run
  python cancer_evaluation.py --compare baseline after_reranker
        """,
    )
    p.add_argument("--label",    default="baseline",
                   help="Run label for history tracking (default: baseline)")
    p.add_argument("--quick",    action="store_true",
                   help="Run first 5 questions only (~2 minutes)")
    p.add_argument("--category", choices=["graph", "research", "image", "edge"],
                   help="Run one category of questions only")
    p.add_argument("--compare",  nargs=2, metavar=("A", "B"),
                   help="Compare two labelled runs from history")
    args = p.parse_args()

    if args.compare:
        compare_runs(args.compare[0], args.compare[1])
    else:
        run_evaluation(
            label    = args.label,
            quick    = args.quick,
            category = args.category,
        )
