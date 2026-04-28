# # =============================================================================
# # c
# # WHAT IT MEASURES (4 RAGAS scores, 0.0 → 1.0):
# #
# #   faithfulness      — Did every claim in the answer come from retrieved
# #                       context? 1.0 = nothing hallucinated.
# #                       Critical for medical safety.
# #
# #   answer_relevancy  — Does the answer actually address the question?
# #                       1.0 = perfectly on-topic.
# #
# #   context_precision — Of chunks retrieved, what fraction were useful?
# #                       1.0 = every retrieved chunk contributed.
# #
# #   context_recall    — Did retrieved chunks contain all info needed?
# #                       1.0 = nothing relevant was missed.
# #
# # TEST SET STRUCTURE (20 gold-standard Q&A pairs):
# #   5 × Graph mode    — drug-food interactions, nutrition guidelines
# #   5 × Research mode — survival rates, staging, treatment rationale
# #   5 × Image         — PRISMA flowchart, survival curves, figures
# #   5 × Edge cases    — out-of-corpus, web fallback, partial answers
# #
# # OUTPUT:
# #   output/evaluation/ragas_scores.json    — full numeric results
# #   output/evaluation/ragas_report.html   — visual dashboard
# #   output/evaluation/ragas_history.jsonl — one line per run (tracks trends)
# #
# # USAGE:
# #   # Baseline (before any enhancement):
# #   python cancer_evaluation.py
# #
# #   # After implementing cross-encoder reranking:
# #   python cancer_evaluation.py --label "after_crossencoder"
# #
# #   # Quick smoke-test (5 questions only):
# #   python cancer_evaluation.py --quick
# #
# #   # Specific category only:
# #   python cancer_evaluation.py --category graph
# #   python cancer_evaluation.py --category research
# #   python cancer_evaluation.py --category image
# #
# # DEPENDENCIES:
# #   pip install ragas datasets langchain-openai
# #   RAGAS uses an LLM to judge faithfulness and relevancy.
# #   We configure it to use ChatGroq (same as the pipeline — zero extra cost).
# # =============================================================================

# from __future__ import annotations

# import json
# import time
# import argparse
# import traceback
# from datetime import datetime
# from pathlib import Path
# from typing import Optional

# # =============================================================================
# # RAGAS imports — graceful failure if not installed
# # =============================================================================

# try:
#     from ragas import evaluate
#     from ragas.metrics import (
#         faithfulness,
#         answer_relevancy,
#         context_precision,
#         context_recall,
#     )
#     from ragas.llms import LangchainLLMWrapper
#     from ragas.embeddings import LangchainEmbeddingsWrapper
#     from datasets import Dataset
#     _RAGAS_AVAILABLE = True
# except ImportError:
#     _RAGAS_AVAILABLE = False
#     print("⚠️  RAGAS not installed. Run: pip install ragas datasets")

# try:
#     from langchain_groq import ChatGroq
#     _GROQ_LC_AVAILABLE = True
# except ImportError:
#     _GROQ_LC_AVAILABLE = False

# from config import (
#     GROQ_API_KEY, GROQ_MODEL_QUERY,
#     QUERY_MODE_RESEARCH, QUERY_MODE_GRAPH, QUERY_MODE_AUTO,
#     EMBEDDING_MODEL,
# )
# from cancer_retrieval import generate_answer, _vector_retrieve, _run_research_mode, _run_graph_mode, _run_auto_mode

# # =============================================================================
# # OUTPUT DIRECTORIES
# # =============================================================================

# EVAL_DIR = Path(__file__).parent / "output" / "evaluation"
# EVAL_DIR.mkdir(parents=True, exist_ok=True)

# SCORES_PATH  = EVAL_DIR / "ragas_scores.json"
# REPORT_PATH  = EVAL_DIR / "ragas_report.html"
# HISTORY_PATH = EVAL_DIR / "ragas_history.jsonl"

# # =============================================================================
# # GOLD-STANDARD TEST SET
# # 20 Q&A pairs covering all four categories.
# #
# # Structure per entry:
# #   question      — the user's query
# #   ground_truth  — the correct answer (written by a domain expert)
# #                   RAGAS uses this to compute context_recall
# #   query_mode    — which pipeline mode should handle this
# #   category      — graph | research | image | edge
# #   description   — human note explaining what this test validates
# #
# # HOW TO EXTEND:
# #   Add entries to TEST_SET below.
# #   Keep ground_truth concise (2-4 sentences) — longer truths make
# #   context_recall harder to score accurately.
# # =============================================================================

# TEST_SET = [

#     # =========================================================================
#     # CATEGORY: graph — drug-food interactions and nutrition guidelines
#     # These test the hand-crafted knowledge graph retrieval path.
#     # =========================================================================

#     {
#         "question":     "What foods should a patient on cisplatin avoid?",
#         "ground_truth": (
#             "Patients on cisplatin should avoid alcohol, fatty and fried foods, "
#             "spicy foods, and large meals. Cisplatin commonly causes nausea and "
#             "vomiting, so small frequent bland meals are recommended. Adequate "
#             "hydration is mandatory to prevent nephrotoxicity."
#         ),
#         "query_mode":   QUERY_MODE_GRAPH,
#         "category":     "graph",
#         "description":  "Basic food avoidance — core graph traversal test",
#     },

#     {
#         "question":     "What are the mandatory nutritional guidelines for a patient receiving pemetrexed?",
#         "ground_truth": (
#             "Pemetrexed requires mandatory supplementation with folic acid "
#             "400-1000 mcg daily starting 7 days before treatment and vitamin B12 "
#             "1000 mcg intramuscularly 7 days before the first dose and every 3 "
#             "cycles. These supplements are required to reduce toxicity, not optional."
#         ),
#         "query_mode":   QUERY_MODE_GRAPH,
#         "category":     "graph",
#         "description":  "Mandatory supplementation — NutritionGuideline node test",
#     },

#     {
#         "question":     "A breast cancer patient on AC-T protocol is also taking warfarin. What are the dietary risks?",
#         "ground_truth": (
#             "Warfarin interacts with capecitabine (part of some breast cancer "
#             "regimens) causing elevated INR and bleeding risk. Capecitabine-induced "
#             "diarrhoea reduces vitamin K absorption, destabilising warfarin control. "
#             "The patient should avoid sudden changes in vitamin K-rich foods like "
#             "leafy green vegetables and maintain consistent dietary patterns. "
#             "Close INR monitoring is essential."
#         ),
#         "query_mode":   QUERY_MODE_GRAPH,
#         "category":     "graph",
#         "description":  "Multi-drug interaction with dietary impact",
#     },

#     {
#         "question":     "What eating side effects does vincristine cause and how should they be managed?",
#         "ground_truth": (
#             "Vincristine commonly causes constipation due to autonomic neuropathy "
#             "affecting bowel motility. Patients should increase dietary fibre, "
#             "maintain adequate fluid intake, and use stool softeners prophylactically. "
#             "Paralytic ileus is a serious risk, especially when vincristine is "
#             "combined with morphine or amitriptyline."
#         ),
#         "query_mode":   QUERY_MODE_GRAPH,
#         "category":     "graph",
#         "description":  "Eating adverse effect with management — EatingAdverseEffect node",
#     },

#     {
#         "question":     "What foods can help manage nausea during chemotherapy?",
#         "ground_truth": (
#             "Foods that help manage chemotherapy-induced nausea include dry crackers, "
#             "plain toast, ginger tea or ginger biscuits, cold or room-temperature "
#             "foods (which have less smell), small frequent meals, and clear fluids. "
#             "Patients should avoid fatty, fried, spicy, or strong-smelling foods. "
#             "Eating before nausea peaks and avoiding lying down after meals also helps."
#         ),
#         "query_mode":   QUERY_MODE_GRAPH,
#         "category":     "graph",
#         "description":  "Symptom management food guidance — RELIEVED_BY edge test",
#     },

#     # =========================================================================
#     # CATEGORY: research — clinical literature from PDF chunks
#     # These test the vector retrieval path against the 6 review papers.
#     # =========================================================================

#     {
#         "question":     "What is the 5-year overall survival rate for osteosarcoma?",
#         "ground_truth": (
#             "The 5-year overall survival rate for osteosarcoma is approximately "
#             "60-70% for localised disease treated with neoadjuvant chemotherapy "
#             "and surgery. For metastatic osteosarcoma at diagnosis, 5-year survival "
#             "drops to around 20-30%. Histological response to neoadjuvant "
#             "chemotherapy (good response defined as >90% necrosis) is the strongest "
#             "prognostic factor."
#         ),
#         "query_mode":   QUERY_MODE_RESEARCH,
#         "category":     "research",
#         "description":  "Survival statistics — tests osteosarcoma-review.pdf retrieval",
#     },

#     {
#         "question":     "What are the main subtypes of acute leukemia and how do they differ in treatment?",
#         "ground_truth": (
#             "Acute leukemia has two main subtypes: acute lymphoblastic leukemia "
#             "(ALL) and acute myeloid leukemia (AML). ALL is more common in children "
#             "and treated with multi-agent chemotherapy including induction, "
#             "consolidation, and maintenance phases. AML is more common in adults "
#             "and the standard induction regimen is 7+3 (cytarabine plus an "
#             "anthracycline). Philadelphia chromosome-positive ALL requires addition "
#             "of tyrosine kinase inhibitors like imatinib or dasatinib."
#         ),
#         "query_mode":   QUERY_MODE_RESEARCH,
#         "category":     "research",
#         "description":  "Treatment differentiation — tests acute-leukemia-review.pdf",
#     },

#     {
#         "question":     "What molecular targets guide treatment decisions in non-small cell lung cancer?",
#         "ground_truth": (
#             "In NSCLC, key molecular targets include EGFR mutations (treated with "
#             "erlotinib, gefitinib, or osimertinib), ALK gene rearrangements (treated "
#             "with crizotinib, alectinib, or lorlatinib), KRAS G12C mutations "
#             "(sotorasib), and ROS1 rearrangements. PD-L1 expression guides "
#             "immunotherapy selection with pembrolizumab recommended for high "
#             "PD-L1 expressors (≥50%) as first-line monotherapy."
#         ),
#         "query_mode":   QUERY_MODE_RESEARCH,
#         "category":     "research",
#         "description":  "Molecular targeted therapy — tests lung-cancer-review.pdf",
#     },

#     {
#         "question":     "What is the staging system used for melanoma and what determines stage IV?",
#         "ground_truth": (
#             "Melanoma is staged using the AJCC TNM system. Stage IV melanoma "
#             "indicates distant metastasis — spread to distant skin, lymph nodes, "
#             "or visceral organs. Stage IV is further subdivided by site of metastasis "
#             "and lactate dehydrogenase (LDH) levels. The 5-year survival for stage "
#             "IV melanoma has improved significantly with checkpoint inhibitor "
#             "immunotherapy, with some patients achieving durable remission."
#         ),
#         "query_mode":   QUERY_MODE_RESEARCH,
#         "category":     "research",
#         "description":  "Staging criteria — tests melanoma-skin-cancer-review.pdf",
#     },

#     {
#         "question":     "What are the main risk factors for developing breast cancer?",
#         "ground_truth": (
#             "Main risk factors for breast cancer include female sex, increasing age, "
#             "BRCA1/BRCA2 gene mutations, positive family history, early menarche, "
#             "late menopause, nulliparity or late first pregnancy, hormone replacement "
#             "therapy, alcohol consumption, obesity particularly post-menopause, "
#             "and prior chest radiation. HER2 overexpression and ER/PR positivity "
#             "are tumour characteristics that influence treatment rather than causation."
#         ),
#         "query_mode":   QUERY_MODE_RESEARCH,
#         "category":     "research",
#         "description":  "Risk factors — tests breast-cancer-review.pdf",
#     },

#     # =========================================================================
#     # CATEGORY: image — tests image chunk retrieval and [IMAGE:] tag generation
#     # These are the most important tests for Issue 1 validation.
#     # A passing test means the answer contains at least one [IMAGE:] tag.
#     # =========================================================================

#     {
#         "question":     "Show me the PRISMA flowchart from the systematic review.",
#         "ground_truth": (
#             "The PRISMA flowchart illustrates the systematic review selection process, "
#             "showing the number of records identified through database searching, "
#             "records screened, records excluded, full-text articles assessed for "
#             "eligibility, studies excluded with reasons, and studies included in "
#             "the final synthesis. The flowchart provides transparency in the "
#             "literature selection methodology."
#         ),
#         "query_mode":   QUERY_MODE_RESEARCH,
#         "category":     "image",
#         "description":  "Explicit PRISMA flowchart request — must return [IMAGE:] tag",
#     },

#     {
#         "question":     "Are there any figures showing survival curves or Kaplan-Meier plots?",
#         "ground_truth": (
#             "Kaplan-Meier survival curves plot the probability of survival over time "
#             "for patient cohorts. They show overall survival and disease-free survival "
#             "with median survival times. Curves that diverge early and remain separated "
#             "indicate durable treatment benefit. The log-rank test p-value indicates "
#             "whether survival differences between groups are statistically significant."
#         ),
#         "query_mode":   QUERY_MODE_RESEARCH,
#         "category":     "image",
#         "description":  "Survival curve request — must return [IMAGE:] tag",
#     },

#     {
#         "question":     "What does Figure 1 show in the breast cancer paper?",
#         "ground_truth": (
#             "Figures in breast cancer review papers typically illustrate treatment "
#             "algorithms, molecular subtype classifications, or drug delivery systems. "
#             "Common figures include flow diagrams of treatment decision pathways "
#             "based on hormone receptor status and HER2 expression, and schematic "
#             "representations of drug mechanisms of action."
#         ),
#         "query_mode":   QUERY_MODE_RESEARCH,
#         "category":     "image",
#         "description":  "Specific figure reference — tests figure_caption chunk retrieval",
#     },

#     {
#         "question":     "Show me any tables with chemotherapy drug dosing information.",
#         "ground_truth": (
#             "Chemotherapy dosing tables typically show drug name, dose in mg/m², "
#             "route of administration, schedule frequency, and common toxicities. "
#             "They may also include dose reduction guidelines for toxicity and "
#             "renal or hepatic impairment. Standard regimen tables often list "
#             "combination protocols with each drug's contribution."
#         ),
#         "query_mode":   QUERY_MODE_RESEARCH,
#         "category":     "image",
#         "description":  "Table request — tests table_caption chunk retrieval",
#     },

#     {
#         "question":     "Are there any flowcharts showing treatment decision pathways for leukemia?",
#         "ground_truth": (
#             "Treatment decision flowcharts for acute leukemia show the pathway "
#             "from diagnosis through risk stratification, induction chemotherapy, "
#             "response assessment, and subsequent consolidation or stem cell "
#             "transplantation decisions. Risk stratification typically includes "
#             "cytogenetic findings, molecular markers like Philadelphia chromosome, "
#             "and minimal residual disease assessment."
#         ),
#         "query_mode":   QUERY_MODE_RESEARCH,
#         "category":     "image",
#         "description":  "Flowchart request — tests leukemia image retrieval",
#     },

#     # =========================================================================
#     # CATEGORY: edge — boundary cases that test fallback and robustness
#     # =========================================================================

#     {
#         "question":     "What vaccine is approved for preventing osteosarcoma?",
#         "ground_truth": (
#             "As of current knowledge, there is no vaccine approved for preventing "
#             "osteosarcoma in humans. Research is ongoing into immunotherapy approaches "
#             "and investigational vaccines for canine osteosarcoma models, but these "
#             "are not approved for human use. This question falls outside the scope "
#             "of the available clinical literature."
#         ),
#         "query_mode":   QUERY_MODE_AUTO,
#         "category":     "edge",
#         "description":  "Out-of-corpus query — must trigger web search, not hallucinate",
#     },

#     {
#         "question":     "What is the latest FDA-approved drug for osteosarcoma in 2024?",
#         "ground_truth": (
#             "This requires current regulatory information beyond the scope of "
#             "the clinical review papers. The FDA approval landscape changes "
#             "frequently and requires checking current FDA databases or recent "
#             "clinical trial publications."
#         ),
#         "query_mode":   QUERY_MODE_AUTO,
#         "category":     "edge",
#         "description":  "Recent regulatory question — proactive web fallback test",
#     },

#     {
#         "question":     "What are the eating effects of paclitaxel and what foods should be avoided?",
#         "ground_truth": (
#             "Paclitaxel commonly causes nausea, taste changes, and mouth sores "
#             "(mucositis). Patients should avoid spicy, acidic, rough, or very hot "
#             "foods that worsen mouth sores. Alcohol should be avoided. Small frequent "
#             "bland meals help manage nausea. Peripheral neuropathy from paclitaxel "
#             "can make food preparation difficult."
#         ),
#         "query_mode":   QUERY_MODE_AUTO,
#         "category":     "edge",
#         "description":  "Auto mode hybrid query — tests graph + vector path selection",
#     },

#     {
#         "question":     "Does chemotherapy affect fertility?",
#         "ground_truth": (
#             "Many chemotherapy agents, particularly alkylating agents like "
#             "cyclophosphamide, can cause gonadal toxicity leading to temporary "
#             "or permanent infertility. The risk depends on the drug, cumulative "
#             "dose, patient age, and baseline fertility. Fertility preservation "
#             "options including sperm banking and egg/embryo freezing should be "
#             "discussed before starting treatment."
#         ),
#         "query_mode":   QUERY_MODE_AUTO,
#         "category":     "edge",
#         "description":  "Partially in-corpus topic — tests mixed retrieval handling",
#     },

#     {
#         "question":     "What is the standard treatment for stage III melanoma?",
#         "ground_truth": (
#             "Stage III melanoma is treated with surgical resection of the primary "
#             "tumour and regional lymph node dissection. Adjuvant systemic therapy "
#             "is recommended including anti-PD-1 immunotherapy (pembrolizumab or "
#             "nivolumab) or targeted therapy with BRAF/MEK inhibitors (dabrafenib "
#             "plus trametinib) for BRAF V600E-mutated tumours. Radiation therapy "
#             "may be considered for high-risk regional disease."
#         ),
#         "query_mode":   QUERY_MODE_RESEARCH,
#         "category":     "edge",
#         "description":  "Specific staging question — tests precise context retrieval",
#     },
# ]

# # =============================================================================
# # RETRIEVE CONTEXTS for RAGAS
# # RAGAS needs the actual retrieved chunks, not just the final answer.
# # This function mirrors what generate_answer() does internally but
# # returns the contexts separately for RAGAS scoring.
# # =============================================================================

# def _get_contexts_for_query(
#     question:   str,
#     query_mode: str,
# ) -> list[str]:
#     """
#     Retrieve the contexts that the pipeline would use for this question.
#     Returns a list of strings — each string is one retrieved chunk's text.
#     RAGAS uses these to compute context_precision and context_recall.
#     """
#     try:
#         if query_mode == QUERY_MODE_RESEARCH:
#             context_text, vector_docs, sources, _ = _run_research_mode(
#                 question, "", [], ""
#             )
#         elif query_mode == QUERY_MODE_GRAPH:
#             context_text, vector_docs, sources, _ = _run_graph_mode(
#                 question, "", [], ""
#             )
#         else:
#             context_text, vector_docs, sources, _ = _run_auto_mode(
#                 question, "", [], ""
#             )

#         # Return individual chunk texts for RAGAS
#         # Each chunk is a separate context string
#         contexts = [doc.page_content for doc in vector_docs if doc.page_content.strip()]

#         # Also include graph context as a context string if present
#         if context_text and "## Graph Knowledge Base" in context_text:
#             graph_section = context_text.split("## Graph Knowledge Base")[1]
#             if graph_section.strip():
#                 contexts.insert(0, "## Graph Knowledge Base" + graph_section[:2000])

#         return contexts if contexts else ["No context retrieved"]

#     except Exception as e:
#         print(f"   ⚠️  Context retrieval error: {e}")
#         return ["Context retrieval failed"]

# # =============================================================================
# # CUSTOM METRICS for image evaluation
# # RAGAS does not have a built-in image metric — we add our own.
# # =============================================================================

# def _score_image_retrieval(answer: str, category: str) -> float:
#     """
#     Custom metric: did the answer contain [IMAGE:] tags?
#     Only scored for image-category questions.

#     Returns:
#       1.0 — answer contains at least one valid [IMAGE:] tag
#       0.5 — answer references visual content but tag format is wrong
#       0.0 — no image reference at all (image retrieval failed)
#     """
#     if category != "image":
#         return None   # not applicable

#     import re
#     has_tag        = bool(re.search(r'\[IMAGE:\s*[^\]]+\]', answer, re.IGNORECASE))
#     has_visual_ref = any(kw in answer.lower() for kw in [
#         "figure", "fig.", "table", "chart", "flowchart",
#         "diagram", "image", "shown", "illustrated",
#     ])

#     if has_tag:
#         return 1.0
#     elif has_visual_ref:
#         return 0.5
#     else:
#         return 0.0


# def _score_web_fallback(answer: str, category: str) -> float:
#     """
#     Custom metric: did out-of-corpus queries correctly trigger web fallback?

#     Returns:
#       1.0 — answer contains web search indicator (🌐 or [W1] etc.)
#       0.0 — answer tried to answer from context (potential hallucination)
#     """
#     if category != "edge":
#         return None

#     has_web_indicator = (
#         "🌐" in answer
#         or "[W1]" in answer
#         or "[W2]" in answer
#         or "web search" in answer.lower()
#         or "according to" in answer.lower()   # web-style citation
#     )
#     return 1.0 if has_web_indicator else 0.0

# # =============================================================================
# # MAIN EVALUATION RUNNER
# # =============================================================================

# def run_evaluation(
#     label:    str  = "baseline",
#     quick:    bool = False,
#     category: str  = None,
# ) -> dict:
#     """
#     Run RAGAS evaluation on the test set.

#     Args:
#         label:    Run label for history tracking (e.g. "after_crossencoder")
#         quick:    If True, run only 5 questions (smoke test)
#         category: If set, run only that category (graph|research|image|edge)

#     Returns:
#         dict with all scores and per-question results
#     """
#     if not _RAGAS_AVAILABLE:
#         print("❌ RAGAS not installed. Run: pip install ragas datasets langchain-groq")
#         return {}

#     print("=" * 70)
#     print(f"  MedChat RAGAS Evaluation Pipeline")
#     print(f"  Label   : {label}")
#     print(f"  Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print("=" * 70)

#     # ── Filter test set ───────────────────────────────────────────────────────
#     test_items = TEST_SET
#     if category:
#         test_items = [t for t in TEST_SET if t["category"] == category]
#         print(f"  Category filter: {category} ({len(test_items)} questions)")
#     if quick:
#         test_items = test_items[:5]
#         print(f"  Quick mode: first 5 questions only")

#     print(f"  Questions: {len(test_items)}")
#     print()

#     # ── Configure RAGAS to use ChatGroq ───────────────────────────────────────
#     # RAGAS needs an LLM to judge faithfulness and answer relevancy.
#     # We use the same ChatGroq model as the pipeline — zero extra API cost.
#     from langchain_groq import ChatGroq
#     from langchain_huggingface import HuggingFaceEmbeddings

#     ragas_llm = LangchainLLMWrapper(
#         ChatGroq(
#             model=GROQ_MODEL_QUERY,
#             temperature=0,          # deterministic for evaluation
#             api_key=GROQ_API_KEY,
#         )
#     )
#     ragas_embeddings = LangchainEmbeddingsWrapper(
#         HuggingFaceEmbeddings(
#             model_name=EMBEDDING_MODEL,
#             model_kwargs={"device": "cpu"},
#             encode_kwargs={"normalize_embeddings": True},
#         )
#     )

#     # ── Run pipeline for each question ────────────────────────────────────────
#     questions      = []
#     answers        = []
#     contexts_list  = []
#     ground_truths  = []
#     per_question   = []   # detailed results including custom metrics

#     for i, item in enumerate(test_items, 1):
#         q          = item["question"]
#         gt         = item["ground_truth"]
#         mode       = item["query_mode"]
#         cat        = item["category"]
#         desc       = item["description"]

#         print(f"  [{i:2d}/{len(test_items)}] {cat.upper()} | {q[:60]}...")

#         try:
#             # Generate answer through full pipeline
#             t_start = time.time()
#             answer, sources = generate_answer(
#                 query=q,
#                 patient_report="",
#                 chat_history=[],
#                 cancer_filter="",
#                 query_mode=mode,
#             )
#             elapsed = time.time() - t_start

#             # Get contexts for RAGAS scoring
#             contexts = _get_contexts_for_query(q, mode)

#             # Custom metrics
#             image_score   = _score_image_retrieval(answer, cat)
#             web_score     = _score_web_fallback(answer, cat)
#             has_image_tag = "[IMAGE:" in answer.upper()

#             print(f"         ✅ {elapsed:.1f}s | "
#                   f"contexts={len(contexts)} | "
#                   f"image={'✅' if has_image_tag else '❌'} | "
#                   f"chars={len(answer)}")

#             questions.append(q)
#             answers.append(answer)
#             contexts_list.append(contexts)
#             ground_truths.append(gt)

#             per_question.append({
#                 "question":     q,
#                 "answer":       answer,
#                 "ground_truth": gt,
#                 "query_mode":   mode,
#                 "category":     cat,
#                 "description":  desc,
#                 "elapsed_s":    round(elapsed, 2),
#                 "context_count": len(contexts),
#                 "has_image_tag": has_image_tag,
#                 "image_score":  image_score,
#                 "web_score":    web_score,
#                 "sources":      [s.get("label", "") for s in sources],
#                 "answer_preview": answer[:300],
#             })

#             # Rate limit buffer between Groq calls
#             time.sleep(1.5)

#         except Exception as e:
#             print(f"         ❌ Error: {str(e)[:80]}")
#             traceback.print_exc()
#             per_question.append({
#                 "question":    q,
#                 "error":       str(e),
#                 "category":    cat,
#                 "description": desc,
#             })
#             # Still add placeholders to keep lists aligned for RAGAS
#             questions.append(q)
#             answers.append("Error generating answer.")
#             contexts_list.append(["Error"])
#             ground_truths.append(gt)
#             time.sleep(2)

#     print(f"\n  ✅ Pipeline runs complete. Starting RAGAS scoring...")
#     print(f"  ⚠️  RAGAS scoring uses Groq LLM — may take 2-5 minutes...\n")

#     # ── RAGAS scoring ─────────────────────────────────────────────────────────
#     dataset = Dataset.from_dict({
#         "question":   questions,
#         "answer":     answers,
#         "contexts":   contexts_list,
#         "ground_truth": ground_truths,
#     })

#     ragas_scores = {}
#     try:
#         result = evaluate(
#             dataset=dataset,
#             metrics=[
#                 faithfulness,
#                 answer_relevancy,
#                 context_precision,
#                 context_recall,
#             ],
#             llm=ragas_llm,
#             embeddings=ragas_embeddings,
#             raise_exceptions=False,
#         )
#         ragas_scores = {
#             "faithfulness":      round(float(result["faithfulness"]),      4),
#             "answer_relevancy":  round(float(result["answer_relevancy"]),  4),
#             "context_precision": round(float(result["context_precision"]), 4),
#             "context_recall":    round(float(result["context_recall"]),    4),
#         }
#         ragas_scores["composite"] = round(
#             sum(ragas_scores.values()) / len(ragas_scores), 4
#         )

#     except Exception as e:
#         print(f"   ❌ RAGAS scoring error: {e}")
#         traceback.print_exc()
#         ragas_scores = {
#             "faithfulness": 0.0, "answer_relevancy": 0.0,
#             "context_precision": 0.0, "context_recall": 0.0,
#             "composite": 0.0, "error": str(e),
#         }

#     # ── Custom metric aggregates ──────────────────────────────────────────────
#     image_items = [p for p in per_question if p.get("category") == "image"]
#     edge_items  = [p for p in per_question if p.get("category") == "edge"]

#     image_score_avg = (
#         sum(p["image_score"] for p in image_items if p.get("image_score") is not None)
#         / max(len(image_items), 1)
#     ) if image_items else None

#     web_score_avg = (
#         sum(p["web_score"] for p in edge_items if p.get("web_score") is not None)
#         / max(len(edge_items), 1)
#     ) if edge_items else None

#     # ── Category-level breakdown ──────────────────────────────────────────────
#     category_stats: dict = {}
#     for cat in ["graph", "research", "image", "edge"]:
#         items = [p for p in per_question if p.get("category") == cat]
#         if items:
#             category_stats[cat] = {
#                 "count":          len(items),
#                 "avg_elapsed_s":  round(
#                     sum(p.get("elapsed_s", 0) for p in items) / len(items), 2
#                 ),
#                 "image_tag_rate": (
#                     sum(1 for p in items if p.get("has_image_tag")) / len(items)
#                     if cat == "image" else None
#                 ),
#             }

#     # ── Assemble full results ─────────────────────────────────────────────────
#     full_results = {
#         "label":           label,
#         "timestamp":       datetime.now().isoformat(),
#         "question_count":  len(test_items),
#         "ragas_scores":    ragas_scores,
#         "custom_scores": {
#             "image_retrieval_score": round(image_score_avg, 4) if image_score_avg else None,
#             "web_fallback_score":    round(web_score_avg, 4)   if web_score_avg   else None,
#         },
#         "category_stats":  category_stats,
#         "per_question":    per_question,
#     }

#     # ── Save outputs ──────────────────────────────────────────────────────────
#     with open(SCORES_PATH, "w", encoding="utf-8") as f:
#         json.dump(full_results, f, indent=2, ensure_ascii=False)
#     print(f"\n  📊 Scores saved: {SCORES_PATH}")

#     # Append to history (one line per run for trend tracking)
#     history_entry = {
#         "label":     label,
#         "timestamp": full_results["timestamp"],
#         "scores":    ragas_scores,
#         "custom":    full_results["custom_scores"],
#         "n":         len(test_items),
#     }
#     with open(HISTORY_PATH, "a", encoding="utf-8") as f:
#         f.write(json.dumps(history_entry) + "\n")
#     print(f"  📈 History updated: {HISTORY_PATH}")

#     # Generate HTML report
#     _generate_html_report(full_results)
#     print(f"  🌐 Report: {REPORT_PATH}")

#     # ── Print summary ─────────────────────────────────────────────────────────
#     _print_summary(full_results)

#     return full_results


# # =============================================================================
# # HTML REPORT GENERATOR
# # =============================================================================

# def _generate_html_report(results: dict) -> None:
#     """Generate a visual HTML dashboard of evaluation results."""

#     scores  = results["ragas_scores"]
#     custom  = results["custom_scores"]
#     per_q   = results["per_question"]
#     label   = results["label"]
#     ts      = results["timestamp"]

#     def _bar(value: float, colour: str = "#2fa36b") -> str:
#         if value is None:
#             return "<span style='color:#999'>N/A</span>"
#         pct = int(value * 100)
#         bg  = "#e8f5e9" if value >= 0.7 else "#fff3e0" if value >= 0.5 else "#fce4ec"
#         return (
#             f"<div style='background:{bg};border-radius:4px;padding:2px 6px;"
#             f"display:inline-block;min-width:120px'>"
#             f"<div style='background:{colour};width:{pct}%;height:8px;"
#             f"border-radius:3px;margin-bottom:2px'></div>"
#             f"<span style='font-size:13px;font-weight:500'>{value:.3f}</span>"
#             f"</div>"
#         )

#     def _colour(value: float) -> str:
#         if value is None: return "#999"
#         if value >= 0.7:  return "#2fa36b"
#         if value >= 0.5:  return "#f59e0b"
#         return "#ef4444"

#     rows = ""
#     for i, item in enumerate(per_q, 1):
#         if "error" in item:
#             rows += (
#                 f"<tr><td>{i}</td>"
#                 f"<td><span style='background:#fee2e2;padding:2px 6px;"
#                 f"border-radius:3px;font-size:11px'>{item.get('category','?')}</span></td>"
#                 f"<td colspan='5' style='color:#ef4444'>"
#                 f"ERROR: {item.get('error','?')[:80]}</td></tr>"
#             )
#             continue

#         cat_colour = {
#             "graph":    "#dbeafe", "research": "#dcfce7",
#             "image":    "#fef3c7", "edge":     "#f3e8ff",
#         }.get(item.get("category", ""), "#f0f0f0")

#         image_badge = ""
#         if item.get("has_image_tag"):
#             image_badge = (
#                 "<span style='background:#dcfce7;color:#166534;"
#                 "padding:1px 6px;border-radius:3px;font-size:11px;"
#                 "margin-left:4px'>🖼️ image</span>"
#             )

#         web_badge = ""
#         if item.get("web_score") == 1.0:
#             web_badge = (
#                 "<span style='background:#dbeafe;color:#1e40af;"
#                 "padding:1px 6px;border-radius:3px;font-size:11px;"
#                 "margin-left:4px'>🌐 web</span>"
#             )

#         img_score_html = (
#             _bar(item.get("image_score"), "#8b5cf6")
#             if item.get("image_score") is not None else "—"
#         )

#         rows += f"""
#         <tr>
#           <td style='color:#6b7280;font-size:13px'>{i}</td>
#           <td>
#             <span style='background:{cat_colour};padding:2px 8px;
#             border-radius:3px;font-size:11px;font-weight:500'>
#             {item.get('category','?')}</span>
#           </td>
#           <td style='font-size:13px;max-width:280px'>
#             {item.get('question','')[:80]}{'...' if len(item.get('question',''))>80 else ''}
#             {image_badge}{web_badge}
#           </td>
#           <td style='font-size:12px;color:#6b7280'>
#             {item.get('query_mode','?')}
#           </td>
#           <td style='font-size:12px'>{item.get('elapsed_s','?')}s</td>
#           <td style='font-size:12px;color:#6b7280'>
#             {item.get('context_count','?')} chunks
#           </td>
#           <td>{img_score_html}</td>
#         </tr>"""

#     # Load history for trend chart
#     history_rows = ""
#     if HISTORY_PATH.exists():
#         with open(HISTORY_PATH) as f:
#             history = [json.loads(l) for l in f if l.strip()]
#         for h in history[-10:]:   # last 10 runs
#             sc = h.get("scores", {})
#             history_rows += f"""
#             <tr>
#               <td style='font-size:12px'>{h.get('timestamp','?')[:16]}</td>
#               <td style='font-size:12px'>{h.get('label','?')}</td>
#               <td>{_bar(sc.get('faithfulness'))}</td>
#               <td>{_bar(sc.get('answer_relevancy'))}</td>
#               <td>{_bar(sc.get('context_precision'))}</td>
#               <td>{_bar(sc.get('context_recall'))}</td>
#               <td>{_bar(sc.get('composite'), '#6366f1')}</td>
#             </tr>"""

#     html = f"""<!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <title>MedChat RAGAS Evaluation — {label}</title>
# <style>
#   body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
#          background:#f8fafc; color:#1e293b; margin:0; padding:24px; }}
#   .header {{ background:linear-gradient(135deg,#1f2a44,#2fa36b);
#              color:white; padding:24px 32px; border-radius:12px; margin-bottom:24px; }}
#   .header h1 {{ margin:0; font-size:24px; }}
#   .header p  {{ margin:6px 0 0; opacity:.8; font-size:14px; }}
#   .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
#            gap:16px; margin-bottom:24px; }}
#   .card {{ background:white; border-radius:10px; padding:20px;
#            box-shadow:0 1px 3px rgba(0,0,0,.08); text-align:center; }}
#   .card .val {{ font-size:36px; font-weight:700; margin:8px 0 4px; }}
#   .card .lbl {{ font-size:13px; color:#6b7280; }}
#   table {{ width:100%; border-collapse:collapse; background:white;
#            border-radius:10px; overflow:hidden;
#            box-shadow:0 1px 3px rgba(0,0,0,.08); }}
#   th {{ background:#f1f5f9; padding:10px 14px; text-align:left;
#         font-size:12px; color:#6b7280; font-weight:600;
#         text-transform:uppercase; letter-spacing:.04em; }}
#   td {{ padding:10px 14px; border-top:1px solid #f1f5f9; vertical-align:middle; }}
#   tr:hover td {{ background:#fafafa; }}
#   h2 {{ font-size:18px; font-weight:600; margin:24px 0 12px; }}
#   .section {{ background:white; border-radius:10px; padding:20px 24px;
#               box-shadow:0 1px 3px rgba(0,0,0,.08); margin-bottom:20px; }}
# </style>
# </head>
# <body>

# <div class="header">
#   <h1>MedChat RAGAS Evaluation</h1>
#   <p>Label: <strong>{label}</strong> &nbsp;|&nbsp;
#      Run: {ts[:16]} &nbsp;|&nbsp;
#      Questions: {results['question_count']}</p>
# </div>

# <div class="grid">
#   <div class="card">
#     <div class="lbl">Faithfulness</div>
#     <div class="val" style="color:{_colour(scores.get('faithfulness'))}">
#       {scores.get('faithfulness', 0):.3f}</div>
#     <div class="lbl">No hallucination</div>
#   </div>
#   <div class="card">
#     <div class="lbl">Answer Relevancy</div>
#     <div class="val" style="color:{_colour(scores.get('answer_relevancy'))}">
#       {scores.get('answer_relevancy', 0):.3f}</div>
#     <div class="lbl">On-topic answers</div>
#   </div>
#   <div class="card">
#     <div class="lbl">Context Precision</div>
#     <div class="val" style="color:{_colour(scores.get('context_precision'))}">
#       {scores.get('context_precision', 0):.3f}</div>
#     <div class="lbl">Retrieval quality</div>
#   </div>
#   <div class="card">
#     <div class="lbl">Context Recall</div>
#     <div class="val" style="color:{_colour(scores.get('context_recall'))}">
#       {scores.get('context_recall', 0):.3f}</div>
#     <div class="lbl">Coverage</div>
#   </div>
#   <div class="card" style="border:2px solid #6366f1">
#     <div class="lbl">Composite Score</div>
#     <div class="val" style="color:#6366f1">
#       {scores.get('composite', 0):.3f}</div>
#     <div class="lbl">Overall quality</div>
#   </div>
#   <div class="card">
#     <div class="lbl">Image Retrieval</div>
#     <div class="val" style="color:{_colour(custom.get('image_retrieval_score'))}">
#       {f"{custom.get('image_retrieval_score', 0):.3f}" if custom.get('image_retrieval_score') is not None else 'N/A'}
#     </div>
#     <div class="lbl">Image tag rate</div>
#   </div>
#   <div class="card">
#     <div class="lbl">Web Fallback</div>
#     <div class="val" style="color:{_colour(custom.get('web_fallback_score'))}">
#       {f"{custom.get('web_fallback_score', 0):.3f}" if custom.get('web_fallback_score') is not None else 'N/A'}
#     </div>
#     <div class="lbl">Correct fallback</div>
#   </div>
# </div>

# <h2>Per-question results</h2>
# <table>
#   <tr>
#     <th>#</th><th>Category</th><th>Question</th>
#     <th>Mode</th><th>Time</th><th>Contexts</th><th>Image score</th>
#   </tr>
#   {rows}
# </table>

# <h2>Run history (last 10 runs)</h2>
# <div class="section">
# <table>
#   <tr>
#     <th>Timestamp</th><th>Label</th><th>Faithfulness</th>
#     <th>Relevancy</th><th>Precision</th><th>Recall</th><th>Composite</th>
#   </tr>
#   {history_rows if history_rows else '<tr><td colspan="7" style="color:#6b7280;text-align:center">No history yet — this is the first run</td></tr>'}
# </table>
# </div>

# <p style="color:#9ca3af;font-size:12px;margin-top:24px">
#   Generated by cancer_evaluation.py &nbsp;|&nbsp;
#   MedChat Graph RAG &nbsp;|&nbsp; {ts}
# </p>
# </body>
# </html>"""

#     with open(REPORT_PATH, "w", encoding="utf-8") as f:
#         f.write(html)


# # =============================================================================
# # PRINT SUMMARY
# # =============================================================================

# def _print_summary(results: dict) -> None:
#     scores = results["ragas_scores"]
#     custom = results["custom_scores"]

#     def _grade(v: float) -> str:
#         if v is None: return "N/A "
#         if v >= 0.80: return "✅  "
#         if v >= 0.60: return "⚠️  "
#         return "❌  "

#     print(f"\n{'='*70}")
#     print(f"  RAGAS EVALUATION RESULTS — {results['label']}")
#     print(f"{'='*70}")
#     print(f"\n  Core RAGAS Metrics:")
#     print(f"  {_grade(scores.get('faithfulness')    )} Faithfulness      : {scores.get('faithfulness',      0):.3f}  (target ≥ 0.80)")
#     print(f"  {_grade(scores.get('answer_relevancy') )} Answer Relevancy  : {scores.get('answer_relevancy',  0):.3f}  (target ≥ 0.75)")
#     print(f"  {_grade(scores.get('context_precision'))} Context Precision : {scores.get('context_precision', 0):.3f}  (target ≥ 0.70)")
#     print(f"  {_grade(scores.get('context_recall')   )} Context Recall    : {scores.get('context_recall',    0):.3f}  (target ≥ 0.70)")
#     print(f"\n  {'─'*40}")
#     print(f"  {'  '} Composite Score   : {scores.get('composite', 0):.3f}")

#     print(f"\n  Custom Metrics:")
#     img = custom.get("image_retrieval_score")
#     web = custom.get("web_fallback_score")
#     print(f"  {_grade(img)} Image Retrieval   : {f'{img:.3f}' if img is not None else 'N/A ':>5}  (target = 1.00 for image questions)")
#     print(f"  {_grade(web)} Web Fallback      : {f'{web:.3f}' if web is not None else 'N/A ':>5}  (target = 1.00 for edge questions)")

#     # Category breakdown
#     print(f"\n  By category:")
#     for cat, stat in results.get("category_stats", {}).items():
#         img_rate = stat.get("image_tag_rate")
#         img_str  = f" | image_tag_rate={img_rate:.0%}" if img_rate is not None else ""
#         print(f"    {cat:<10} {stat['count']} questions | avg {stat['avg_elapsed_s']}s{img_str}")

#     print(f"\n  Output files:")
#     print(f"    Scores  : {SCORES_PATH}")
#     print(f"    Report  : {REPORT_PATH}")
#     print(f"    History : {HISTORY_PATH}")
#     print(f"\n  ➡️  Open {REPORT_PATH} in your browser to view the full dashboard")
#     print("=" * 70)


# # =============================================================================
# # COMPARISON UTILITY
# # Compare two runs from history to see if an enhancement helped
# # =============================================================================

# def compare_runs(label_a: str, label_b: str) -> None:
#     """
#     Compare two evaluation runs from history.
#     Usage: call from Python or add --compare flag.

#     Example:
#         compare_runs("baseline", "after_crossencoder")
#     """
#     if not HISTORY_PATH.exists():
#         print("No history file found. Run evaluation first.")
#         return

#     with open(HISTORY_PATH) as f:
#         history = [json.loads(l) for l in f if l.strip()]

#     run_a = next((h for h in history if h["label"] == label_a), None)
#     run_b = next((h for h in history if h["label"] == label_b), None)

#     if not run_a:
#         print(f"Label '{label_a}' not found in history.")
#         return
#     if not run_b:
#         print(f"Label '{label_b}' not found in history.")
#         return

#     print(f"\n{'='*60}")
#     print(f"  Comparison: {label_a}  →  {label_b}")
#     print(f"{'='*60}")

#     metrics = ["faithfulness", "answer_relevancy", "context_precision",
#                "context_recall", "composite"]

#     for metric in metrics:
#         a_val = run_a["scores"].get(metric, 0)
#         b_val = run_b["scores"].get(metric, 0)
#         delta = b_val - a_val
#         arrow = "▲" if delta > 0.01 else "▼" if delta < -0.01 else "→"
#         colour = "✅" if delta > 0.01 else "❌" if delta < -0.01 else "  "
#         print(f"  {colour} {metric:<20} {a_val:.3f}  {arrow}  {b_val:.3f}  "
#               f"({'+' if delta >= 0 else ''}{delta:.3f})")

#     print()


# # =============================================================================
# # CLI ENTRY POINT
# # =============================================================================

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="MedChat RAGAS Evaluation Pipeline"
#     )
#     parser.add_argument(
#         "--label",    default="baseline",
#         help="Run label for history tracking (default: baseline)"
#     )
#     parser.add_argument(
#         "--quick",    action="store_true",
#         help="Run only first 5 questions (smoke test)"
#     )
#     parser.add_argument(
#         "--category", default=None,
#         choices=["graph", "research", "image", "edge"],
#         help="Run only one category"
#     )
#     parser.add_argument(
#         "--compare",  nargs=2, metavar=("LABEL_A", "LABEL_B"),
#         help="Compare two runs from history"
#     )
#     args = parser.parse_args()

#     if args.compare:
#         compare_runs(args.compare[0], args.compare[1])
#     else:
#         run_evaluation(
#             label=args.label,
#             quick=args.quick,
#             category=args.category,
#         )

# Attempt 2

# # =============================================================================
# # cancer_evaluation.py  v3.0  FINAL
# #
# # REQUIRES (run these two commands first):
# #   pip uninstall ragas -y
# #   pip install "ragas==0.2.6"
# #
# # WHY: Your installed ragas 0.4.3 is the vibrantlabsai FORK, not the official
# # explodinggradients/ragas. The fork's Faithfulness class does NOT inherit from
# # ragas.metrics.base.Metric, so evaluate() always rejects it with:
# #   "All metrics must be initialised metric objects"
# # Official ragas 0.2.6 uses module-level singleton metrics that DO pass the
# # isinstance(m, Metric) check inside evaluate().
# #
# # CONFIRMED WORKING PATTERN (from ragas_test.py API inspection):
# #   from ragas.metrics import faithfulness, answer_relevancy, ...
# #   result = evaluate(dataset, metrics=[faithfulness,...], llm=ragas_llm, embeddings=ragas_emb)
# #
# # HOW TO RUN:
# #   python cancer_evaluation.py --quick           # 5 questions ~8 min
# #   python cancer_evaluation.py                   # 20 questions ~30 min
# #   python cancer_evaluation.py --category graph  # one category
# #   python cancer_evaluation.py --label v5_1      # labelled run
# #   python cancer_evaluation.py --compare baseline v5_1
# # =============================================================================

# from __future__ import annotations

# import json
# import math
# import time
# import warnings
# import argparse
# import traceback
# from datetime import datetime
# from pathlib import Path
# from typing import Optional

# warnings.filterwarnings("ignore", category=DeprecationWarning)

# # =============================================================================
# # RAGAS — official explodinggradients/ragas 0.2.x API
# # Metrics are MODULE-LEVEL SINGLETONS. LLM passed to evaluate(), not constructors.
# # =============================================================================

# _RAGAS_AVAILABLE = False
# _ragas_ver = "unknown"

# try:
#     import ragas as _rp
#     _ragas_ver = _rp.__version__

#     # Detect the vibrantlabsai fork and warn immediately
#     try:
#         import importlib.metadata as _m
#         _hp = _m.metadata("ragas").get("Home-page", "")
#         if "vibrantlabsai" in _hp:
#             print("=" * 60)
#             print("❌  WRONG RAGAS PACKAGE")
#             print("    You have the vibrantlabsai fork (incompatible).")
#             print("    Fix:")
#             print("      pip uninstall ragas -y")
#             print("      pip install 'ragas==0.2.6'")
#             print("=" * 60)
#             raise ImportError("vibrantlabsai fork detected")
#     except ImportError as _fork_err:
#         if "vibrantlabsai" in str(_fork_err):
#             raise

#     from ragas              import evaluate, RunConfig
#     from ragas.metrics      import (faithfulness, answer_relevancy,
#                                     context_precision, context_recall)
#     from ragas.llms         import LangchainLLMWrapper
#     from ragas.embeddings   import LangchainEmbeddingsWrapper
#     from datasets           import Dataset

#     # Confirm these are real Metric instances (not vibrantlabsai classes)
#     from ragas.metrics.base import Metric as _Metric
#     assert isinstance(faithfulness, _Metric), "faithfulness is not a Metric instance"

#     _RAGAS_AVAILABLE = True
#     print(f"✅  RAGAS {_ragas_ver} (official explodinggradients) ready")

# except (ImportError, AssertionError) as _e:
#     print(f"⚠️  RAGAS unavailable: {_e}")
#     print("    Run: pip uninstall ragas -y && pip install 'ragas==0.2.6'")

# from langchain_groq        import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings

# from config import (
#     GROQ_API_KEY, GROQ_MODEL_QUERY,
#     QUERY_MODE_RESEARCH, QUERY_MODE_GRAPH, QUERY_MODE_AUTO,
#     EMBEDDING_MODEL,
# )
# from cancer_retrieval import (
#     generate_answer,
#     _run_research_mode, _run_graph_mode, _run_auto_mode,
#     _retrieve_image_chunks,
# )

# # =============================================================================
# # CONFIG
# # =============================================================================

# # llama-3.3-70b-versatile: 32k context window handles context recall without truncation.
# # Better structured output than 8b — fixes the faithfulness NLI StringIO parsing error.
# RAGAS_JUDGE_MODEL = "llama-3.3-70b-versatile"

# EVAL_DIR     = Path(__file__).parent / "output" / "evaluation"
# SCORES_PATH  = EVAL_DIR / "ragas_scores.json"
# REPORT_PATH  = EVAL_DIR / "ragas_report.html"
# HISTORY_PATH = EVAL_DIR / "ragas_history.jsonl"
# EVAL_DIR.mkdir(parents=True, exist_ok=True)

# # =============================================================================
# # TEST SET — 20 gold-standard Q&A pairs
# # =============================================================================

# TEST_SET = [
#     # GRAPH (5)
#     {"question": "What foods should a patient on cisplatin avoid?",
#      "ground_truth": "Patients should avoid alcohol, fatty fried foods, spicy foods, and large meals. Small frequent bland meals are recommended. Adequate hydration prevents nephrotoxicity.",
#      "query_mode": QUERY_MODE_GRAPH, "category": "graph",
#      "description": "Food avoidance — core graph traversal"},
#     {"question": "What are the mandatory nutritional guidelines for pemetrexed?",
#      "ground_truth": "Pemetrexed requires folic acid 400-1000 mcg daily from 7 days before and vitamin B12 1000 mcg IM every 3 cycles to reduce haematological and GI toxicity.",
#      "query_mode": QUERY_MODE_GRAPH, "category": "graph",
#      "description": "Mandatory supplementation — NutritionGuideline node"},
#     {"question": "A breast cancer patient on AC-T is taking warfarin. What are the dietary risks?",
#      "ground_truth": "Warfarin interacts with capecitabine raising INR and bleeding risk. Diarrhoea reduces vitamin K absorption. Consistent vitamin K intake and INR monitoring are essential.",
#      "query_mode": QUERY_MODE_GRAPH, "category": "graph",
#      "description": "Drug-diet interaction"},
#     {"question": "What eating side effects does vincristine cause and how are they managed?",
#      "ground_truth": "Vincristine causes constipation from autonomic neuropathy. Increase dietary fibre, fluids, use stool softeners. Paralytic ileus risk increases with concurrent morphine.",
#      "query_mode": QUERY_MODE_GRAPH, "category": "graph",
#      "description": "EatingAdverseEffect node test"},
#     {"question": "What foods help manage nausea during chemotherapy?",
#      "ground_truth": "Dry crackers, plain toast, ginger tea, cold foods, and small frequent meals help. Avoid fatty, fried, spicy, or strong-smelling foods.",
#      "query_mode": QUERY_MODE_GRAPH, "category": "graph",
#      "description": "RELIEVED_BY edge test"},
#     # RESEARCH (5)
#     {"question": "What is the 5-year overall survival rate for osteosarcoma?",
#      "ground_truth": "5-year survival for localised osteosarcoma is 60-70%. Metastatic disease is 20-30%. Histological response >90% necrosis is the strongest prognostic factor.",
#      "query_mode": QUERY_MODE_RESEARCH, "category": "research",
#      "description": "Survival stats"},
#     {"question": "How do ALL and AML differ in treatment?",
#      "ground_truth": "ALL is common in children and uses multi-agent protocols. AML uses 7+3 induction. Ph+ ALL requires TKIs like imatinib.",
#      "query_mode": QUERY_MODE_RESEARCH, "category": "research",
#      "description": "Leukemia treatment differentiation"},
#     {"question": "What molecular targets guide NSCLC treatment?",
#      "ground_truth": "EGFR: erlotinib/osimertinib. ALK: crizotinib/alectinib. PD-L1 high: pembrolizumab. KRAS G12C: sotorasib.",
#      "query_mode": QUERY_MODE_RESEARCH, "category": "research",
#      "description": "Molecular targets"},
#     {"question": "What staging system is used for melanoma and what is stage IV?",
#      "ground_truth": "Melanoma uses AJCC TNM. Stage IV means distant metastasis. Survival has improved with checkpoint inhibitors.",
#      "query_mode": QUERY_MODE_RESEARCH, "category": "research",
#      "description": "Staging system"},
#     {"question": "What are the main risk factors for breast cancer?",
#      "ground_truth": "Age, BRCA1/2 mutations, family history, early menarche, late menopause, nulliparity, HRT, alcohol, and obesity.",
#      "query_mode": QUERY_MODE_RESEARCH, "category": "research",
#      "description": "Risk factors"},
#     # IMAGE (5)
#     {"question": "Show me the PRISMA flowchart from the systematic review.",
#      "ground_truth": "PRISMA flowchart shows records identified, screened, excluded, and included in the final synthesis.",
#      "query_mode": QUERY_MODE_RESEARCH, "category": "image",
#      "description": "PRISMA — must return [IMAGE:] tag"},
#     {"question": "Are there any survival curves or Kaplan-Meier plots?",
#      "ground_truth": "Kaplan-Meier curves show survival probability over time with median survival and log-rank p-values.",
#      "query_mode": QUERY_MODE_RESEARCH, "category": "image",
#      "description": "Survival curves"},
#     {"question": "What does Figure 1 show in the breast cancer paper?",
#      "ground_truth": "Figures typically show treatment algorithms, molecular subtype classifications, or drug delivery systems.",
#      "query_mode": QUERY_MODE_RESEARCH, "category": "image",
#      "description": "Specific figure reference"},
#     {"question": "Show me tables with chemotherapy dosing information.",
#      "ground_truth": "Dosing tables show drug name, dose in mg/m2, route, schedule, and toxicities.",
#      "query_mode": QUERY_MODE_RESEARCH, "category": "image",
#      "description": "Table request"},
#     {"question": "Are there flowcharts showing leukemia treatment pathways?",
#      "ground_truth": "Leukemia flowcharts show diagnosis through risk stratification, induction, response, and transplant decisions.",
#      "query_mode": QUERY_MODE_RESEARCH, "category": "image",
#      "description": "Flowchart request"},
#     # EDGE (5)
#     {"question": "What vaccine is approved for preventing osteosarcoma?",
#      "ground_truth": "No vaccine is approved for preventing osteosarcoma in humans. This is outside the scope of the clinical review papers.",
#      "query_mode": QUERY_MODE_AUTO, "category": "edge",
#      "description": "Out-of-corpus — must trigger web fallback"},
#     {"question": "What is the latest FDA-approved drug for osteosarcoma in 2024?",
#      "ground_truth": "This requires current regulatory information beyond the scope of the review papers.",
#      "query_mode": QUERY_MODE_AUTO, "category": "edge",
#      "description": "Recent regulatory"},
#     {"question": "What eating effects does paclitaxel cause and what foods to avoid?",
#      "ground_truth": "Paclitaxel causes nausea, taste changes, and mucositis. Avoid spicy and acidic foods. Small bland meals help.",
#      "query_mode": QUERY_MODE_AUTO, "category": "edge",
#      "description": "Auto mode hybrid"},
#     {"question": "Does chemotherapy affect fertility?",
#      "ground_truth": "Alkylating agents cause gonadal toxicity. Fertility preservation should be discussed before treatment.",
#      "query_mode": QUERY_MODE_AUTO, "category": "edge",
#      "description": "Partially in-corpus"},
#     {"question": "What is the standard treatment for stage III melanoma?",
#      "ground_truth": "Surgery plus adjuvant anti-PD-1 or BRAF/MEK inhibitors for BRAF V600E mutated tumours.",
#      "query_mode": QUERY_MODE_RESEARCH, "category": "edge",
#      "description": "Specific staging"},
# ]

# # =============================================================================
# # HELPERS
# # =============================================================================

# def _get_contexts(q: str, mode: str) -> list[str]:
#     try:
#         fn = {QUERY_MODE_RESEARCH: _run_research_mode,
#               QUERY_MODE_GRAPH:    _run_graph_mode}.get(mode, _run_auto_mode)
#         _, docs, _, _ = fn(q, "", [], "")
#         ctx = [d.page_content for d in docs if d.page_content.strip()]
#         for img in _retrieve_image_chunks(q):
#             if img.page_content.strip():
#                 ctx.append(img.page_content)
#         return ctx or ["No context retrieved"]
#     except Exception as e:
#         return [f"Error: {e}"]

# def _image_score(ans: str, cat: str) -> Optional[float]:
#     if cat != "image": return None
#     import re
#     if re.search(r'\[IMAGE:\s*[^\]]+\]', ans, re.IGNORECASE): return 1.0
#     if any(k in ans.lower() for k in ["figure","table","chart","flowchart","diagram"]): return 0.5
#     return 0.0

# def _web_score(ans: str, cat: str) -> Optional[float]:
#     if cat != "edge": return None
#     return 1.0 if ("🌐" in ans or "[W1]" in ans or "web search" in ans.lower()) else 0.0

# def _safe(v) -> Optional[float]:
#     try:
#         f = float(v)
#         return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
#     except (TypeError, ValueError):
#         return None

# def _mean_col(df, col: str) -> Optional[float]:
#     if col not in df.columns: return None
#     good = [_safe(v) for v in df[col].tolist() if _safe(v) is not None]
#     return round(sum(good)/len(good), 4) if good else None

# def _grade(v) -> str:
#     f = _safe(v)
#     if f is None or f == 0.0: return "❓  "
#     if f >= 0.80: return "✅  "
#     if f >= 0.60: return "⚠️  "
#     return "❌  "

# # =============================================================================
# # MAIN RUNNER
# # =============================================================================

# def run_evaluation(label="baseline", quick=False, category=None) -> dict:
#     if not _RAGAS_AVAILABLE:
#         print("❌  RAGAS unavailable.")
#         print("    pip uninstall ragas -y && pip install 'ragas==0.2.6'")
#         return {}

#     print("=" * 70)
#     print(f"  MedChat RAGAS Evaluation  v3.0")
#     print(f"  RAGAS    : {_ragas_ver} (official)")
#     print(f"  Pipeline : {GROQ_MODEL_QUERY}")
#     print(f"  Judge    : {RAGAS_JUDGE_MODEL}")
#     print(f"  Label    : {label}")
#     print(f"  Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print("=" * 70)

#     items = TEST_SET
#     if category:
#         items = [t for t in items if t["category"] == category]
#         print(f"  Category : {category} ({len(items)} questions)")
#     if quick:
#         items = items[:5]
#         print(f"  Mode     : quick (first 5)")
#     print(f"  Total    : {len(items)} questions\n")

#     # Build RAGAS LLM — LangchainLLMWrapper around ChatGroq
#     lc_llm    = ChatGroq(model=RAGAS_JUDGE_MODEL, temperature=0, api_key=GROQ_API_KEY, max_tokens=2048)
#     lc_emb    = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
#                                        model_kwargs={"device":"cpu"},
#                                        encode_kwargs={"normalize_embeddings":True})
#     ragas_llm = LangchainLLMWrapper(lc_llm)
#     ragas_emb = LangchainEmbeddingsWrapper(lc_emb)
#     print(f"  ✅  Judge LLM     : LangchainLLMWrapper({RAGAS_JUDGE_MODEL})")
#     print(f"  ✅  Embeddings    : LangchainEmbeddingsWrapper({EMBEDDING_MODEL})\n")

#     # CONFIRMED: ragas 0.2.6 singleton metrics pass isinstance(m, Metric)
#     METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

#     # ── Step 1: Generate answers ────────────────────────────────────────────
#     print("  Step 1 — Generating answers...")
#     questions, answers, contexts_list, ground_truths, per_question = [], [], [], [], []

#     for i, item in enumerate(items, 1):
#         q, gt, mode, cat = item["question"], item["ground_truth"], item["query_mode"], item["category"]
#         print(f"  [{i:2d}/{len(items)}] {cat.upper():10s} | {q[:58]}...")
#         try:
#             t0 = time.time()
#             ans, srcs = generate_answer(query=q, patient_report="", chat_history=[],
#                                          cancer_filter="", query_mode=mode)
#             el  = time.time() - t0
#             ctx = _get_contexts(q, mode)
#             ims = _image_score(ans, cat)
#             ws  = _web_score(ans, cat)
#             hi  = "[IMAGE:" in ans.upper()
#             hw  = "🌐" in ans or "[W1]" in ans
#             print(f"             ✅ {el:.1f}s | ctx={len(ctx)} | "
#                   f"{'🖼️' if hi else '  '} {'🌐' if hw else ''} chars={len(ans)}")
#             questions.append(q); answers.append(ans)
#             contexts_list.append(ctx); ground_truths.append(gt)
#             per_question.append({"question":q,"answer":ans,"ground_truth":gt,
#                                   "query_mode":mode,"category":cat,
#                                   "description":item["description"],
#                                   "elapsed_s":round(el,2),"context_count":len(ctx),
#                                   "has_image_tag":hi,"web_fired":hw,
#                                   "image_score":ims,"web_score":ws,
#                                   "sources":[s.get("label","") for s in srcs],
#                                   "answer_preview":ans[:300]})
#             time.sleep(2)
#         except Exception as e:
#             print(f"             ❌ {str(e)[:80]}")
#             questions.append(q); answers.append("Error.")
#             contexts_list.append(["Error"]); ground_truths.append(gt)
#             per_question.append({"question":q,"error":str(e),
#                                   "category":cat,"description":item["description"]})
#             time.sleep(2)

#     # ── Step 2: RAGAS scoring ───────────────────────────────────────────────
#     print(f"\n  Step 2 — RAGAS scoring ({RAGAS_JUDGE_MODEL})...")
#     print(f"  Est: {len(items)*20}–{len(items)*35}s\n")

#     ds = Dataset.from_dict({"question":questions,"answer":answers,
#                             "contexts":contexts_list,"ground_truth":ground_truths})
#     ragas_scores: dict = {}
#     try:
#         # CONFIRMED WORKING PATTERN for official ragas 0.2.6:
#         # - pass singleton metrics list
#         # - pass llm= and embeddings= to evaluate()
#         # - RunConfig max_workers=1 forces sequential (Groq rate limit safety)
#         run_cfg = RunConfig(max_workers=1, timeout=180)
#         result  = evaluate(dataset=ds, metrics=METRICS,
#                            llm=ragas_llm, embeddings=ragas_emb,
#                            run_config=run_cfg, raise_exceptions=False)
#         df = result.to_pandas()
#         print("  Raw per-question scores:")
#         cols = [c for c in ["faithfulness","answer_relevancy",
#                              "context_precision","context_recall"] if c in df.columns]
#         print(df[cols].to_string())
#         print()

#         fs = _mean_col(df,"faithfulness"); ars = _mean_col(df,"answer_relevancy")
#         cps = _mean_col(df,"context_precision"); crs = _mean_col(df,"context_recall")
#         valid = [s for s in [fs,ars,cps,crs] if s is not None]
#         ragas_scores = {
#             "faithfulness":      fs  or 0.0,
#             "answer_relevancy":  ars or 0.0,
#             "context_precision": cps or 0.0,
#             "context_recall":    crs or 0.0,
#             "composite": round(sum(valid)/len(valid),4) if valid else 0.0,
#         }
#         gi = 0
#         for pq in per_question:
#             if "error" not in pq and gi < len(df):
#                 row = df.iloc[gi]
#                 for c in ["faithfulness","answer_relevancy","context_precision","context_recall"]:
#                     if c in df.columns: pq[f"ragas_{c}"] = _safe(row[c])
#                 gi += 1
#     except Exception as e:
#         print(f"  ❌ RAGAS error: {e}"); traceback.print_exc()
#         ragas_scores = {"faithfulness":0.0,"answer_relevancy":0.0,
#                         "context_precision":0.0,"context_recall":0.0,
#                         "composite":0.0,"error":str(e)}

#     # ── Step 3: Custom + category stats ────────────────────────────────────
#     img_items  = [p for p in per_question if p.get("category")=="image" and "error" not in p]
#     edge_items = [p for p in per_question if p.get("category")=="edge"  and "error" not in p]
#     img_ss  = [p["image_score"] for p in img_items  if p.get("image_score") is not None]
#     web_ss  = [p["web_score"]   for p in edge_items if p.get("web_score")   is not None]
#     img_avg = round(sum(img_ss)/len(img_ss),4) if img_ss else None
#     web_avg = round(sum(web_ss)/len(web_ss),4) if web_ss else None

#     cat_stats: dict = {}
#     for cat in ["graph","research","image","edge"]:
#         its = [p for p in per_question if p.get("category")==cat]
#         if its:
#             ok = [p for p in its if "error" not in p]
#             cat_stats[cat] = {"count":len(its),"success":len(ok),
#                 "avg_elapsed_s": round(sum(p.get("elapsed_s",0) for p in ok)/len(ok),2) if ok else 0,
#                 "image_tag_rate": round(sum(1 for p in ok if p.get("has_image_tag"))/len(ok),3) if cat=="image" and ok else None,
#                 "web_rate": round(sum(1 for p in ok if p.get("web_fired"))/len(ok),3) if cat=="edge" and ok else None}

#     full = {"label":label,"timestamp":datetime.now().isoformat(),
#             "ragas_version":_ragas_ver,"judge_model":RAGAS_JUDGE_MODEL,
#             "pipeline_model":GROQ_MODEL_QUERY,"question_count":len(items),
#             "ragas_scores":ragas_scores,
#             "custom_scores":{"image_retrieval_score":img_avg,"web_fallback_score":web_avg},
#             "category_stats":cat_stats,"per_question":per_question}

#     with open(SCORES_PATH,"w",encoding="utf-8") as f: json.dump(full,f,indent=2,ensure_ascii=False)
#     with open(HISTORY_PATH,"a",encoding="utf-8") as f:
#         f.write(json.dumps({"label":label,"timestamp":full["timestamp"],
#                             "scores":ragas_scores,"custom":full["custom_scores"],
#                             "n":len(items)})+"\n")

#     _html_report(full)
#     _print_summary(full)
#     print(f"\n  📊 {SCORES_PATH}\n  🌐 {REPORT_PATH}\n  📈 {HISTORY_PATH}")
#     print("=" * 70)
#     return full

# # =============================================================================
# # PRINT SUMMARY
# # =============================================================================

# def _print_summary(r: dict) -> None:
#     s, c = r["ragas_scores"], r["custom_scores"]
#     print(f"\n{'='*70}")
#     print(f"  RAGAS RESULTS — {r['label']}  |  ragas {r.get('ragas_version','?')}")
#     print(f"{'='*70}\n  Scores:")
#     for k,lbl,t in [("faithfulness","Faithfulness     ","≥0.80"),
#                     ("answer_relevancy","Answer Relevancy ","≥0.75"),
#                     ("context_precision","Context Precision","≥0.70"),
#                     ("context_recall","Context Recall   ","≥0.70")]:
#         v = s.get(k, 0.0)
#         print(f"  {_grade(v)}{lbl}  {v:.3f}  (target {t})")
#     print(f"\n  ── Composite: {s.get('composite',0.0):.3f}\n")
#     img = c.get("image_retrieval_score"); web = c.get("web_fallback_score")
#     print(f"  {_grade(img)}Image Retrieval  {f'{img:.3f}' if img is not None else 'N/A':>5}  (target=1.00)")
#     print(f"  {_grade(web)}Web Fallback     {f'{web:.3f}' if web is not None else 'N/A':>5}  (target=1.00)")
#     print("\n  By category:")
#     for cat, st in r.get("category_stats",{}).items():
#         ex = ([f"img_tag={st['image_tag_rate']:.0%}"] if st.get("image_tag_rate") is not None else [])
#         ex += ([f"web={st['web_rate']:.0%}"]          if st.get("web_rate") is not None else [])
#         print(f"    {cat:<12} {st['success']}/{st['count']} ok | "
#               f"avg {st['avg_elapsed_s']}s" + (f"  {'  '.join(ex)}" if ex else ""))

# # =============================================================================
# # HTML REPORT
# # =============================================================================

# def _html_report(r: dict) -> None:
#     s, c, per_q = r["ragas_scores"], r["custom_scores"], r["per_question"]

#     def _col(v) -> str:
#         f = _safe(v)
#         if f is None or f == 0.0: return "#9ca3af"
#         return "#16a34a" if f >= 0.70 else "#d97706" if f >= 0.50 else "#dc2626"

#     def _bar(v, col="#16a34a") -> str:
#         f = _safe(v)
#         if f is None: return "<span style='color:#9ca3af'>N/A</span>"
#         p = int(max(0.0,min(1.0,f))*100)
#         bg = "#f0fdf4" if f>=0.70 else "#fffbeb" if f>=0.50 else "#fef2f2"
#         return (f"<div style='background:{bg};border-radius:4px;padding:3px 8px;"
#                 f"display:inline-block;min-width:110px'>"
#                 f"<div style='background:{col};width:{p}%;height:6px;"
#                 f"border-radius:3px;margin-bottom:2px'></div>"
#                 f"<span style='font-size:13px;font-weight:600'>{f:.3f}</span></div>")

#     rows = ""
#     for i, item in enumerate(per_q, 1):
#         if "error" in item:
#             rows += (f"<tr><td>{i}</td><td><span class='badge badge-{item.get('category','')}'>"
#                      f"{item.get('category','')}</span></td>"
#                      f"<td colspan='6' style='color:#dc2626'>❌ {item.get('error','')[:80]}</td></tr>")
#             continue
#         bd = ("" + ("<span class='bm bi'>🖼️</span>" if item.get("has_image_tag") else "")
#                  + ("<span class='bm bw'>🌐</span>" if item.get("web_fired") else ""))
#         rows += (f"<tr><td style='color:#6b7280;font-size:12px'>{i}</td>"
#                  f"<td><span class='badge badge-{item.get('category','')}' >"
#                  f"{item.get('category','')}</span></td>"
#                  f"<td style='font-size:13px;max-width:250px'>{item.get('question','')[:72]}{'...' if len(item.get('question',''))>72 else ''}{bd}</td>"
#                  f"<td style='font-size:11px;color:#6b7280'>{item.get('query_mode','')}</td>"
#                  f"<td style='font-size:12px'>{item.get('elapsed_s','?')}s</td>"
#                  f"<td style='font-size:12px;color:#6b7280'>{item.get('context_count','?')}</td>"
#                  f"<td>{_bar(item.get('ragas_faithfulness'))}</td>"
#                  f"<td>{_bar(item.get('ragas_answer_relevancy'),'#6366f1')}</td></tr>")

#     hist = ""
#     if HISTORY_PATH.exists():
#         with open(HISTORY_PATH) as hf:
#             for h in [json.loads(l) for l in hf if l.strip()][-10:]:
#                 sc = h.get("scores",{})
#                 hist += (f"<tr><td style='font-size:12px'>{h.get('timestamp','')[:16]}</td>"
#                          f"<td style='font-size:12px;font-weight:500'>{h.get('label','')}</td>"
#                          f"<td>{_bar(sc.get('faithfulness'))}</td>"
#                          f"<td>{_bar(sc.get('answer_relevancy'),'#6366f1')}</td>"
#                          f"<td>{_bar(sc.get('context_precision'),'#0891b2')}</td>"
#                          f"<td>{_bar(sc.get('context_recall'),'#7c3aed')}</td>"
#                          f"<td>{_bar(sc.get('composite'),'#1f2a44')}</td></tr>")

#     ts, label = r["timestamp"], r["label"]
#     img_v = c.get("image_retrieval_score"); web_v = c.get("web_fallback_score")

#     html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
# <title>MedChat RAGAS — {label}</title><style>
# *{{box-sizing:border-box;margin:0;padding:0}}
# body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f1f5f9;color:#1e293b;padding:24px}}
# .hdr{{background:linear-gradient(135deg,#1f2a44,#2fa36b);color:white;padding:24px 32px;border-radius:14px;margin-bottom:24px}}
# .hdr h1{{font-size:22px;margin-bottom:4px}}.hdr p{{font-size:13px;opacity:.8}}
# .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:14px;margin-bottom:24px}}
# .card{{background:white;border-radius:10px;padding:18px;box-shadow:0 1px 3px rgba(0,0,0,.07);text-align:center}}
# .card .val{{font-size:34px;font-weight:700;margin:8px 0 4px}}.card .lbl{{font-size:12px;color:#64748b}}
# .card.hl{{border:2px solid #6366f1}}
# table{{width:100%;border-collapse:collapse;background:white;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.07);margin-bottom:24px}}
# th{{background:#f8fafc;padding:10px 12px;text-align:left;font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.06em}}
# td{{padding:10px 12px;border-top:1px solid #f1f5f9}}tr:hover td{{background:#fafafa}}
# h2{{font-size:16px;font-weight:600;margin-bottom:12px}}
# .badge{{padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}}
# .badge-graph{{background:#dbeafe;color:#1e40af}}.badge-research{{background:#dcfce7;color:#166534}}
# .badge-image{{background:#fef3c7;color:#92400e}}.badge-edge{{background:#f3e8ff;color:#6b21a8}}
# .bm{{padding:1px 6px;border-radius:3px;font-size:11px;margin-left:4px}}
# .bi{{background:#dcfce7;color:#166534}}.bw{{background:#dbeafe;color:#1e40af}}
# </style></head><body>
# <div class="hdr"><h1>MedChat RAGAS Evaluation</h1>
# <p>Label: <strong>{label}</strong> | {ts[:16]} | {r['question_count']}q | RAGAS {r.get('ragas_version','?')} | Judge: <strong>{r['judge_model']}</strong></p></div>
# <div class="grid">
# <div class="card"><div class="lbl">Faithfulness</div><div class="val" style="color:{_col(s.get('faithfulness'))}">{s.get('faithfulness',0.0):.3f}</div><div class="lbl">No hallucination</div></div>
# <div class="card"><div class="lbl">Answer Relevancy</div><div class="val" style="color:{_col(s.get('answer_relevancy'))}">{s.get('answer_relevancy',0.0):.3f}</div><div class="lbl">On-topic</div></div>
# <div class="card"><div class="lbl">Context Precision</div><div class="val" style="color:{_col(s.get('context_precision'))}">{s.get('context_precision',0.0):.3f}</div><div class="lbl">Retrieval quality</div></div>
# <div class="card"><div class="lbl">Context Recall</div><div class="val" style="color:{_col(s.get('context_recall'))}">{s.get('context_recall',0.0):.3f}</div><div class="lbl">Coverage</div></div>
# <div class="card hl"><div class="lbl">Composite</div><div class="val" style="color:#6366f1">{s.get('composite',0.0):.3f}</div><div class="lbl">Overall</div></div>
# <div class="card"><div class="lbl">Image Retrieval</div><div class="val" style="color:{_col(img_v)}">{f"{img_v:.3f}" if img_v is not None else "N/A"}</div><div class="lbl">[IMAGE:] rate</div></div>
# <div class="card"><div class="lbl">Web Fallback</div><div class="val" style="color:{_col(web_v)}">{f"{web_v:.3f}" if web_v is not None else "N/A"}</div><div class="lbl">Fallback rate</div></div>
# </div>
# <h2>Per-question results</h2>
# <table><tr><th>#</th><th>Cat</th><th>Question</th><th>Mode</th><th>Time</th><th>Ctx</th><th>Faithfulness</th><th>Relevancy</th></tr>{rows}</table>
# <h2>Run history</h2>
# <table><tr><th>Timestamp</th><th>Label</th><th>Faithfulness</th><th>Relevancy</th><th>Precision</th><th>Recall</th><th>Composite</th></tr>
# {hist or '<tr><td colspan="7" style="color:#94a3b8;text-align:center;padding:16px">First run</td></tr>'}
# </table>
# <p style="color:#94a3b8;font-size:12px;margin-top:16px">MedChat RAGAS v3.0 | {ts}</p>
# </body></html>"""

#     with open(REPORT_PATH,"w",encoding="utf-8") as f: f.write(html)

# # =============================================================================
# # COMPARE + CLI
# # =============================================================================

# def compare_runs(a: str, b: str) -> None:
#     if not HISTORY_PATH.exists(): print("No history."); return
#     with open(HISTORY_PATH) as f:
#         hist = [json.loads(l) for l in f if l.strip()]
#     ra = next((h for h in reversed(hist) if h["label"]==a), None)
#     rb = next((h for h in reversed(hist) if h["label"]==b), None)
#     if not ra: print(f"'{a}' not found"); return
#     if not rb: print(f"'{b}' not found"); return
#     print(f"\n{'='*60}\n  {a} → {b}\n{'='*60}")
#     for m in ["faithfulness","answer_relevancy","context_precision","context_recall","composite"]:
#         va,vb = ra["scores"].get(m,0.0),rb["scores"].get(m,0.0); d=vb-va
#         print(f"  {'✅' if d>0.005 else '❌' if d<-0.005 else '  '} "
#               f"{m:<22} {va:.3f} → {vb:.3f}  ({'+' if d>=0 else ''}{d:.3f})")

# if __name__ == "__main__":
#     p = argparse.ArgumentParser(description="MedChat RAGAS Evaluation v3.0")
#     p.add_argument("--label",    default="baseline")
#     p.add_argument("--quick",    action="store_true")
#     p.add_argument("--category", choices=["graph","research","image","edge"])
#     p.add_argument("--compare",  nargs=2, metavar=("A","B"))
#     args = p.parse_args()
#     if args.compare: compare_runs(args.compare[0], args.compare[1])
#     else: run_evaluation(label=args.label, quick=args.quick, category=args.category)


# Attempt 3:

# =============================================================================
# cancer_evaluation.py  v3.1  (Graph Context Fix)
#
# REQUIRES:
#   pip uninstall ragas -y
#   pip install "ragas==0.2.6"
#
# HOW TO RUN:
#   python cancer_evaluation.py --quick           # 5 questions ~8 min
#   python cancer_evaluation.py                   # 20 questions ~30 min
#   python cancer_evaluation.py --category graph  # one category
# =============================================================================

# from __future__ import annotations

# import os
# os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

# import json
# import math
# import time
# import warnings
# import argparse
# import traceback
# from datetime import datetime
# from pathlib import Path
# from typing import Optional

# warnings.filterwarnings("ignore", category=DeprecationWarning)

# # =============================================================================
# # RAGAS — official explodinggradients/ragas 0.2.x API
# # =============================================================================

# _RAGAS_AVAILABLE = False
# _ragas_ver = "unknown"

# try:
#     import ragas as _rp
#     _ragas_ver = _rp.__version__

#     try:
#         import importlib.metadata as _m
#         _hp = _m.metadata("ragas").get("Home-page", "")
#         if "vibrantlabsai" in _hp:
#             print("=" * 60)
#             print("❌  WRONG RAGAS PACKAGE")
#             print("    You have the vibrantlabsai fork (incompatible).")
#             print("    Fix:")
#             print("      pip uninstall ragas -y")
#             print("      pip install 'ragas==0.2.6'")
#             print("=" * 60)
#             raise ImportError("vibrantlabsai fork detected")
#     except ImportError as _fork_err:
#         if "vibrantlabsai" in str(_fork_err):
#             raise

#     from ragas              import evaluate, RunConfig
#     from ragas.metrics      import (faithfulness, answer_relevancy,
#                                     context_precision, context_recall)
#     from ragas.llms         import LangchainLLMWrapper
#     from ragas.embeddings   import LangchainEmbeddingsWrapper
#     from datasets           import Dataset

#     from ragas.metrics.base import Metric as _Metric
#     assert isinstance(faithfulness, _Metric), "faithfulness is not a Metric instance"

#     _RAGAS_AVAILABLE = True
#     print(f"✅  RAGAS {_ragas_ver} (official explodinggradients) ready")

# except (ImportError, AssertionError) as _e:
#     print(f"⚠️  RAGAS unavailable: {_e}")
#     print("    Run: pip uninstall ragas -y && pip install 'ragas==0.2.6'")

# from langchain_groq        import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings

# from config import (
#     GROQ_API_KEY, GROQ_MODEL_QUERY,
#     QUERY_MODE_RESEARCH, QUERY_MODE_GRAPH, QUERY_MODE_AUTO,
#     EMBEDDING_MODEL,
# )
# from cancer_retrieval import (
#     generate_answer,
#     _run_research_mode, _run_graph_mode, _run_auto_mode,
#     _retrieve_image_chunks,
# )

# # =============================================================================
# # CONFIG
# # =============================================================================

# RAGAS_JUDGE_MODEL = "llama-3.1-8b-instant"

# EVAL_DIR     = Path(__file__).parent / "output" / "evaluation"
# SCORES_PATH  = EVAL_DIR / "ragas_scores.json"
# REPORT_PATH  = EVAL_DIR / "ragas_report.html"
# HISTORY_PATH = EVAL_DIR / "ragas_history.jsonl"
# EVAL_DIR.mkdir(parents=True, exist_ok=True)

# # =============================================================================
# # TEST SET — 20 gold-standard Q&A pairs
# # =============================================================================

# # TEST_SET = [
# #     # GRAPH (5)
# #     {"question": "What foods should a patient on cisplatin avoid?",
# #      "ground_truth": "Patients should avoid alcohol, fatty fried foods, spicy foods, and large meals. Small frequent bland meals are recommended. Adequate hydration prevents nephrotoxicity.",
# #      "query_mode": QUERY_MODE_GRAPH, "category": "graph",
# #      "description": "Food avoidance — core graph traversal"},
# #     {"question": "What are the mandatory nutritional guidelines for pemetrexed?",
# #      "ground_truth": "Pemetrexed requires folic acid 400-1000 mcg daily from 7 days before and vitamin B12 1000 mcg IM every 3 cycles to reduce haematological and GI toxicity.",
# #      "query_mode": QUERY_MODE_GRAPH, "category": "graph",
# #      "description": "Mandatory supplementation — NutritionGuideline node"},
# #     {"question": "A breast cancer patient on AC-T is taking warfarin. What are the dietary risks?",
# #      "ground_truth": "Warfarin interacts with capecitabine raising INR and bleeding risk. Diarrhoea reduces vitamin K absorption. Consistent vitamin K intake and INR monitoring are essential.",
# #      "query_mode": QUERY_MODE_GRAPH, "category": "graph",
# #      "description": "Drug-diet interaction"},
# #     {"question": "What eating side effects does vincristine cause and how are they managed?",
# #      "ground_truth": "Vincristine causes constipation from autonomic neuropathy. Increase dietary fibre, fluids, use stool softeners. Paralytic ileus risk increases with concurrent morphine.",
# #      "query_mode": QUERY_MODE_GRAPH, "category": "graph",
# #      "description": "EatingAdverseEffect node test"},
# #     {"question": "What foods help manage nausea during chemotherapy?",
# #      "ground_truth": "Dry crackers, plain toast, ginger tea, cold foods, and small frequent meals help. Avoid fatty, fried, spicy, or strong-smelling foods.",
# #      "query_mode": QUERY_MODE_GRAPH, "category": "graph",
# #      "description": "RELIEVED_BY edge test"},
# #     # RESEARCH (5)
# #     {"question": "What is the 5-year overall survival rate for osteosarcoma?",
# #      "ground_truth": "5-year survival for localised osteosarcoma is 60-70%. Metastatic disease is 20-30%. Histological response >90% necrosis is the strongest prognostic factor.",
# #      "query_mode": QUERY_MODE_RESEARCH, "category": "research",
# #      "description": "Survival stats"},
# #     {"question": "How do ALL and AML differ in treatment?",
# #      "ground_truth": "ALL is common in children and uses multi-agent protocols. AML uses 7+3 induction. Ph+ ALL requires TKIs like imatinib.",
# #      "query_mode": QUERY_MODE_RESEARCH, "category": "research",
# #      "description": "Leukemia treatment differentiation"},
# #     {"question": "What molecular targets guide NSCLC treatment?",
# #      "ground_truth": "EGFR: erlotinib/osimertinib. ALK: crizotinib/alectinib. PD-L1 high: pembrolizumab. KRAS G12C: sotorasib.",
# #      "query_mode": QUERY_MODE_RESEARCH, "category": "research",
# #      "description": "Molecular targets"},
# #     {"question": "What staging system is used for melanoma and what is stage IV?",
# #      "ground_truth": "Melanoma uses AJCC TNM. Stage IV means distant metastasis. Survival has improved with checkpoint inhibitors.",
# #      "query_mode": QUERY_MODE_RESEARCH, "category": "research",
# #      "description": "Staging system"},
# #     {"question": "What are the main risk factors for breast cancer?",
# #      "ground_truth": "Age, BRCA1/2 mutations, family history, early menarche, late menopause, nulliparity, HRT, alcohol, and obesity.",
# #      "query_mode": QUERY_MODE_RESEARCH, "category": "research",
# #      "description": "Risk factors"},
# #     # IMAGE (5)
# #     {"question": "Show me the PRISMA flowchart from the systematic review.",
# #      "ground_truth": "PRISMA flowchart shows records identified, screened, excluded, and included in the final synthesis.",
# #      "query_mode": QUERY_MODE_RESEARCH, "category": "image",
# #      "description": "PRISMA — must return [IMAGE:] tag"},
# #     {"question": "Are there any survival curves or Kaplan-Meier plots?",
# #      "ground_truth": "Kaplan-Meier curves show survival probability over time with median survival and log-rank p-values.",
# #      "query_mode": QUERY_MODE_RESEARCH, "category": "image",
# #      "description": "Survival curves"},
# #     {"question": "What does Figure 1 show in the breast cancer paper?",
# #      "ground_truth": "Figures typically show treatment algorithms, molecular subtype classifications, or drug delivery systems.",
# #      "query_mode": QUERY_MODE_RESEARCH, "category": "image",
# #      "description": "Specific figure reference"},
# #     {"question": "Show me tables with chemotherapy dosing information.",
# #      "ground_truth": "Dosing tables show drug name, dose in mg/m2, route, schedule, and toxicities.",
# #      "query_mode": QUERY_MODE_RESEARCH, "category": "image",
# #      "description": "Table request"},
# #     {"question": "Are there flowcharts showing leukemia treatment pathways?",
# #      "ground_truth": "Leukemia flowcharts show diagnosis through risk stratification, induction, response, and transplant decisions.",
# #      "query_mode": QUERY_MODE_RESEARCH, "category": "image",
# #      "description": "Flowchart request"},
# #     # EDGE (5)
# #     {"question": "What vaccine is approved for preventing osteosarcoma?",
# #      "ground_truth": "No vaccine is approved for preventing osteosarcoma in humans. This is outside the scope of the clinical review papers.",
# #      "query_mode": QUERY_MODE_AUTO, "category": "edge",
# #      "description": "Out-of-corpus — must trigger web fallback"},
# #     {"question": "What is the latest FDA-approved drug for osteosarcoma in 2024?",
# #      "ground_truth": "This requires current regulatory information beyond the scope of the review papers.",
# #      "query_mode": QUERY_MODE_AUTO, "category": "edge",
# #      "description": "Recent regulatory"},
# #     {"question": "What eating effects does paclitaxel cause and what foods to avoid?",
# #      "ground_truth": "Paclitaxel causes nausea, taste changes, and mucositis. Avoid spicy and acidic foods. Small bland meals help.",
# #      "query_mode": QUERY_MODE_AUTO, "category": "edge",
# #      "description": "Auto mode hybrid"},
# #     {"question": "Does chemotherapy affect fertility?",
# #      "ground_truth": "Alkylating agents cause gonadal toxicity. Fertility preservation should be discussed before treatment.",
# #      "query_mode": QUERY_MODE_AUTO, "category": "edge",
# #      "description": "Partially in-corpus"},
# #     {"question": "What is the standard treatment for stage III melanoma?",
# #      "ground_truth": "Surgery plus adjuvant anti-PD-1 or BRAF/MEK inhibitors for BRAF V600E mutated tumours.",
# #      "query_mode": QUERY_MODE_RESEARCH, "category": "edge",
# #      "description": "Specific staging"},
# # ]

# # =============================================================================
# # GOLDEN DATASET LOADER
# # =============================================================================
# # import json
# # from pathlib import Path

# # def load_golden_dataset(json_path="D:\\Desktop\\Neo_4J\\Medchat_Graph_RAG\\data\\corrected_golden_dataset.json"):
# #     """
# #     Loads the external JSON golden dataset and maps its metadata
# #     to the internal query modes expected by the RAG pipeline.
# #     """
# #     dataset_path = Path(json_path)
    
# #     # 1. Validation: Ensure the file exists before running the heavy pipeline
# #     if not dataset_path.exists():
# #         print(f"❌ Error: Could not find {json_path} in the root directory.")
# #         return []

# #     with open(dataset_path, "r", encoding="utf-8") as f:
# #         raw_data = json.load(f)

# #     formatted_dataset = []

# #     # 2. Parsing and Mapping: Loop through the JSON and translate the metadata
# #     for item in raw_data:
# #         meta = item.get("metadata", {})
# #         q_type = meta.get("type", "vector")
        
# #         # We must map the JSON "type" to your pipeline's internal QUERY_MODE constants.
# #         # This tells your router whether to hit Neo4j, BM25/LangChain, or both.
# #         if q_type == "graph":
# #             mode = QUERY_MODE_GRAPH
# #             cat = "graph"
# #         elif q_type == "vector":
# #             mode = QUERY_MODE_RESEARCH
# #             cat = "research"
# #         elif q_type == "image":
# #             mode = QUERY_MODE_RESEARCH
# #             cat = "image"
# #         elif q_type == "hybrid":
# #             # Hybrid needs both Graph and Vector data to answer multi-hop questions
# #             mode = QUERY_MODE_AUTO 
# #             cat = "hybrid"
# #         elif q_type == "fallback":
# #             # Edge cases expected to trigger the DuckDuckGo web search
# #             mode = QUERY_MODE_AUTO
# #             cat = "edge"
# #         else:
# #             mode = QUERY_MODE_AUTO
# #             cat = "auto"

# #         # 3. Formatting: Reconstruct the dictionary so the RAGAS runner understands it
# #         formatted_dataset.append({
# #             "question": item["question"],
# #             "ground_truth": item["ground_truth"],
# #             "query_mode": mode,
# #             "category": cat,
# #             "description": f"{meta.get('difficulty', 'unknown').title()} - {meta.get('reasoning_type', 'unknown')}",
# #             # We pass the expected contexts through to help debug Context Recall failures later
# #             "expected_contexts": item.get("contexts", []) 
# #         })

# #     print(f"✅ Successfully loaded {len(formatted_dataset)} questions from the Golden Dataset")
# #     return formatted_dataset

# # # 4. Initialization: Call the function so the rest of the script can use the data
# # TEST_SET = load_golden_dataset()

# # =============================================================================
# # GOLDEN DATASET LOADER
# # =============================================================================
# import json
# from pathlib import Path

# def load_golden_dataset(json_path="D:\\Desktop\\Neo_4J\\Medchat_Graph_RAG\\data\\corrected_golden_dataset.json"):
#     """
#     Loads the external JSON golden dataset and maps its metadata
#     to the internal query modes expected by the RAG pipeline.
#     """
#     dataset_path = Path(json_path)
    
#     # 1. Validation: Ensure the file exists before running the heavy pipeline
#     if not dataset_path.exists():
#         print(f"❌ Error: Could not find {json_path}")
#         return []

#     with open(dataset_path, "r", encoding="utf-8") as f:
#         raw_data = json.load(f)

#     # Buckets for our mingled strategy
#     graph_qs, vector_qs, hybrid_qs, edge_qs = [], [], [], []

#     # 2. Parsing and Mapping: Loop through the JSON and translate the metadata
#     for item in raw_data:
#         meta = item.get("metadata", {})
#         q_type = meta.get("type", "vector")
        
#         mode = {
#             "graph": QUERY_MODE_GRAPH,
#             "vector": QUERY_MODE_RESEARCH,
#             "image": QUERY_MODE_RESEARCH,
#             "hybrid": QUERY_MODE_AUTO,
#             "fallback": QUERY_MODE_AUTO
#         }.get(q_type, QUERY_MODE_AUTO)

#         formatted = {
#             "question": item["question"],
#             "ground_truth": item["ground_truth"],
#             "query_mode": mode,
#             "category": q_type,
#             "description": f"{meta.get('difficulty', 'unknown')} - {meta.get('reasoning_type', '')}"
#         }

#         # Sort questions into their respective buckets
#         if q_type == "graph": graph_qs.append(formatted)
#         elif q_type in ["vector", "image"]: vector_qs.append(formatted)
#         elif q_type == "hybrid": hybrid_qs.append(formatted)
#         elif q_type == "fallback": edge_qs.append(formatted)

#     # 🟢 THE MINGLED 10 STRATEGY: Grab a mix of 10 questions
#     mingled_10 = graph_qs[:3] + vector_qs[:3] + hybrid_qs[:2] + edge_qs[:2]
    
#     print(f"✅ Loaded {len(mingled_10)} questions for the Mingled 10 Benchmark!")
#     return mingled_10

# # 4. Initialization: Call the function so the rest of the script can use the data
# TEST_SET = load_golden_dataset()

# # =============================================================================
# # HELPERS
# # =============================================================================

# def _get_contexts(q: str, mode: str) -> list[str]:
#     """
#     Retrieves contexts exactly as the pipeline does, but formatted
#     for the RAGAS judge. FIX: Includes Graph Cypher context now.
#     """
#     try:
#         fn = {QUERY_MODE_RESEARCH: _run_research_mode,
#               QUERY_MODE_GRAPH:    _run_graph_mode}.get(mode, _run_auto_mode)
        
#         # Capture the full formatted text which includes the graph data
#         context_text, docs, _, _ = fn(q, "", [], "")
        
#         ctx = []
        
#         # 1. Recover the Graph Context (Cypher results)
#         if "## Graph Knowledge Base" in context_text:
#             # Isolate the graph text before the vector chunks begin
#             graph_part = context_text.split("[1] Source:")[0].strip()
#             if graph_part:
#                 ctx.append(graph_part)
                
#         # 2. Add the standard Vector PDF chunks
#         ctx.extend([d.page_content for d in docs if d.page_content.strip()])
        
#         # 3. Add the Visual/Image chunks
#         for img in _retrieve_image_chunks(q):
#             if img.page_content.strip():
#                 ctx.append(img.page_content)
#         # 🟢 THE TPM FIX: Keep only the Top 3 contexts to avoid 6000 TPM Groq limits
#         ctx = ctx[:3]
                
#         return ctx or ["No context retrieved"]
    
#     except Exception as e:
#         return [f"Error: {e}"]

# def _image_score(ans: str, cat: str) -> Optional[float]:
#     if cat != "image": return None
#     import re
#     if re.search(r'\[IMAGE:\s*[^\]]+\]', ans, re.IGNORECASE): return 1.0
#     if any(k in ans.lower() for k in ["figure","table","chart","flowchart","diagram"]): return 0.5
#     return 0.0

# def _web_score(ans: str, cat: str) -> Optional[float]:
#     if cat != "edge": return None
#     return 1.0 if ("🌐" in ans or "[W1]" in ans or "web search" in ans.lower()) else 0.0

# def _safe(v) -> Optional[float]:
#     try:
#         f = float(v)
#         return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
#     except (TypeError, ValueError):
#         return None

# def _mean_col(df, col: str) -> Optional[float]:
#     if col not in df.columns: return None
#     good = [_safe(v) for v in df[col].tolist() if _safe(v) is not None]
#     return round(sum(good)/len(good), 4) if good else None

# def _grade(v) -> str:
#     f = _safe(v)
#     if f is None or f == 0.0: return "❓  "
#     if f >= 0.80: return "✅  "
#     if f >= 0.60: return "⚠️  "
#     return "❌  "

# # =============================================================================
# # MAIN RUNNER
# # =============================================================================

# def run_evaluation(label="baseline", quick=False, category=None) -> dict:
#     if not _RAGAS_AVAILABLE:
#         print("❌  RAGAS unavailable.")
#         print("    pip uninstall ragas -y && pip install 'ragas==0.2.6'")
#         return {}

#     print("=" * 70)
#     print(f"  MedChat RAGAS Evaluation  v3.1 (Graph Fix Applied)")
#     print(f"  RAGAS    : {_ragas_ver} (official)")
#     print(f"  Pipeline : {GROQ_MODEL_QUERY}")
#     print(f"  Judge    : {RAGAS_JUDGE_MODEL}")
#     print(f"  Label    : {label}")
#     print(f"  Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print("=" * 70)

#     items = TEST_SET
#     if category:
#         items = [t for t in items if t["category"] == category]
#         print(f"  Category : {category} ({len(items)} questions)")
#     if quick:
#         items = items[:5]
#         print(f"  Mode     : quick (first 5)")
#     print(f"  Total    : {len(items)} questions\n")

#     # ── Increased max_tokens to 4096 to prevent truncation ──
#     lc_llm    = ChatGroq(model=RAGAS_JUDGE_MODEL, temperature=0, api_key=GROQ_API_KEY, max_tokens=4096)
#     # lc_llm    = ChatGroq(model=RAGAS_JUDGE_MODEL, temperature=0, api_key=GROQ_API_KEY, max_tokens=2048)
#     lc_emb    = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
#                                        model_kwargs={"device":"cpu"},
#                                        encode_kwargs={"normalize_embeddings":True})
#     ragas_llm = LangchainLLMWrapper(lc_llm)
#     ragas_emb = LangchainEmbeddingsWrapper(lc_emb)
#     print(f"  ✅  Judge LLM     : LangchainLLMWrapper({RAGAS_JUDGE_MODEL})")
#     print(f"  ✅  Embeddings    : LangchainEmbeddingsWrapper({EMBEDDING_MODEL})\n")

#     METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

#     # ── Step 1: Generate answers ────────────────────────────────────────────
#     print("  Step 1 — Generating answers...")
#     questions, answers, contexts_list, ground_truths, per_question = [], [], [], [], []

#     for i, item in enumerate(items, 1):
#         q, gt, mode, cat = item["question"], item["ground_truth"], item["query_mode"], item["category"]
#         print(f"  [{i:2d}/{len(items)}] {cat.upper():10s} | {q[:58]}...")
#         try:
#             t0 = time.time()
#             ans, srcs = generate_answer(query=q, patient_report="", chat_history=[],
#                                          cancer_filter="", query_mode=mode)
#             el  = time.time() - t0
#             ctx = _get_contexts(q, mode)
#             ims = _image_score(ans, cat)
#             ws  = _web_score(ans, cat)
#             hi  = "[IMAGE:" in ans.upper()
#             hw  = "🌐" in ans or "[W1]" in ans
#             print(f"             ✅ {el:.1f}s | ctx={len(ctx)} | "
#                   f"{'🖼️' if hi else '  '} {'🌐' if hw else ''} chars={len(ans)}")
#             questions.append(q); answers.append(ans)
#             contexts_list.append(ctx); ground_truths.append(gt)
#             per_question.append({"question":q,"answer":ans,"ground_truth":gt,
#                                   "query_mode":mode,"category":cat,
#                                   "description":item["description"],
#                                   "elapsed_s":round(el,2),"context_count":len(ctx),
#                                   "has_image_tag":hi,"web_fired":hw,
#                                   "image_score":ims,"web_score":ws,
#                                   "sources":[s.get("label","") for s in srcs],
#                                   "answer_preview":ans[:300]})
#             time.sleep(2)
#         except Exception as e:
#             print(f"             ❌ {str(e)[:80]}")
#             questions.append(q); answers.append("Error.")
#             contexts_list.append(["Error"]); ground_truths.append(gt)
#             per_question.append({"question":q,"error":str(e),
#                                   "category":cat,"description":item["description"]})
#             time.sleep(2)

#     # ── Step 2: RAGAS scoring ───────────────────────────────────────────────
#     print(f"\n  Step 2 — RAGAS scoring ({RAGAS_JUDGE_MODEL})...")
#     print(f"  Est: {len(items)*20}–{len(items)*35}s\n")

#     ds = Dataset.from_dict({"question":questions,"answer":answers,
#                             "contexts":contexts_list,"ground_truth":ground_truths})
#     ragas_scores: dict = {}
#     try:
#         run_cfg = RunConfig(max_workers=1, timeout=180)
#         result  = evaluate(dataset=ds, metrics=METRICS,
#                            llm=ragas_llm, embeddings=ragas_emb,
#                            run_config=run_cfg, raise_exceptions=False)
#         df = result.to_pandas()
#         print("  Raw per-question scores:")
#         cols = [c for c in ["faithfulness","answer_relevancy",
#                              "context_precision","context_recall"] if c in df.columns]
#         print(df[cols].to_string())
#         print()

#         fs = _mean_col(df,"faithfulness"); ars = _mean_col(df,"answer_relevancy")
#         cps = _mean_col(df,"context_precision"); crs = _mean_col(df,"context_recall")
#         valid = [s for s in [fs,ars,cps,crs] if s is not None]
#         ragas_scores = {
#             "faithfulness":      fs  or 0.0,
#             "answer_relevancy":  ars or 0.0,
#             "context_precision": cps or 0.0,
#             "context_recall":    crs or 0.0,
#             "composite": round(sum(valid)/len(valid),4) if valid else 0.0,
#         }
#         gi = 0
#         for pq in per_question:
#             if "error" not in pq and gi < len(df):
#                 row = df.iloc[gi]
#                 for c in ["faithfulness","answer_relevancy","context_precision","context_recall"]:
#                     if c in df.columns: pq[f"ragas_{c}"] = _safe(row[c])
#                 gi += 1
#     except Exception as e:
#         print(f"  ❌ RAGAS error: {e}"); traceback.print_exc()
#         ragas_scores = {"faithfulness":0.0,"answer_relevancy":0.0,
#                         "context_precision":0.0,"context_recall":0.0,
#                         "composite":0.0,"error":str(e)}

#     # ── Step 3: Custom + category stats ────────────────────────────────────
#     img_items  = [p for p in per_question if p.get("category")=="image" and "error" not in p]
#     edge_items = [p for p in per_question if p.get("category")=="edge"  and "error" not in p]
#     img_ss  = [p["image_score"] for p in img_items  if p.get("image_score") is not None]
#     web_ss  = [p["web_score"]   for p in edge_items if p.get("web_score")   is not None]
#     img_avg = round(sum(img_ss)/len(img_ss),4) if img_ss else None
#     web_avg = round(sum(web_ss)/len(web_ss),4) if web_ss else None

#     cat_stats: dict = {}
#     for cat in ["graph","research","image","edge"]:
#         its = [p for p in per_question if p.get("category")==cat]
#         if its:
#             ok = [p for p in its if "error" not in p]
#             cat_stats[cat] = {"count":len(its),"success":len(ok),
#                 "avg_elapsed_s": round(sum(p.get("elapsed_s",0) for p in ok)/len(ok),2) if ok else 0,
#                 "image_tag_rate": round(sum(1 for p in ok if p.get("has_image_tag"))/len(ok),3) if cat=="image" and ok else None,
#                 "web_rate": round(sum(1 for p in ok if p.get("web_fired"))/len(ok),3) if cat=="edge" and ok else None}

#     full = {"label":label,"timestamp":datetime.now().isoformat(),
#             "ragas_version":_ragas_ver,"judge_model":RAGAS_JUDGE_MODEL,
#             "pipeline_model":GROQ_MODEL_QUERY,"question_count":len(items),
#             "ragas_scores":ragas_scores,
#             "custom_scores":{"image_retrieval_score":img_avg,"web_fallback_score":web_avg},
#             "category_stats":cat_stats,"per_question":per_question}

#     with open(SCORES_PATH,"w",encoding="utf-8") as f: json.dump(full,f,indent=2,ensure_ascii=False)
#     with open(HISTORY_PATH,"a",encoding="utf-8") as f:
#         f.write(json.dumps({"label":label,"timestamp":full["timestamp"],
#                             "scores":ragas_scores,"custom":full["custom_scores"],
#                             "n":len(items)})+"\n")

#     _html_report(full)
#     _print_summary(full)
#     print(f"\n  📊 {SCORES_PATH}\n  🌐 {REPORT_PATH}\n  📈 {HISTORY_PATH}")
#     print("=" * 70)
#     return full

# # =============================================================================
# # PRINT SUMMARY
# # =============================================================================

# def _print_summary(r: dict) -> None:
#     s, c = r["ragas_scores"], r["custom_scores"]
#     print(f"\n{'='*70}")
#     print(f"  RAGAS RESULTS — {r['label']}  |  ragas {r.get('ragas_version','?')}")
#     print(f"{'='*70}\n  Scores:")
#     for k,lbl,t in [("faithfulness","Faithfulness     ","≥0.80"),
#                     ("answer_relevancy","Answer Relevancy ","≥0.75"),
#                     ("context_precision","Context Precision","≥0.70"),
#                     ("context_recall","Context Recall   ","≥0.70")]:
#         v = s.get(k, 0.0)
#         print(f"  {_grade(v)}{lbl}  {v:.3f}  (target {t})")
#     print(f"\n  ── Composite: {s.get('composite',0.0):.3f}\n")
#     img = c.get("image_retrieval_score"); web = c.get("web_fallback_score")
#     print(f"  {_grade(img)}Image Retrieval  {f'{img:.3f}' if img is not None else 'N/A':>5}  (target=1.00)")
#     print(f"  {_grade(web)}Web Fallback     {f'{web:.3f}' if web is not None else 'N/A':>5}  (target=1.00)")
#     print("\n  By category:")
#     for cat, st in r.get("category_stats",{}).items():
#         ex = ([f"img_tag={st['image_tag_rate']:.0%}"] if st.get("image_tag_rate") is not None else [])
#         ex += ([f"web={st['web_rate']:.0%}"]          if st.get("web_rate") is not None else [])
#         print(f"    {cat:<12} {st['success']}/{st['count']} ok | "
#               f"avg {st['avg_elapsed_s']}s" + (f"  {'  '.join(ex)}" if ex else ""))

# # =============================================================================
# # HTML REPORT
# # =============================================================================

# def _html_report(r: dict) -> None:
#     s, c, per_q = r["ragas_scores"], r["custom_scores"], r["per_question"]

#     def _col(v) -> str:
#         f = _safe(v)
#         if f is None or f == 0.0: return "#9ca3af"
#         return "#16a34a" if f >= 0.70 else "#d97706" if f >= 0.50 else "#dc2626"

#     def _bar(v, col="#16a34a") -> str:
#         f = _safe(v)
#         if f is None: return "<span style='color:#9ca3af'>N/A</span>"
#         p = int(max(0.0,min(1.0,f))*100)
#         bg = "#f0fdf4" if f>=0.70 else "#fffbeb" if f>=0.50 else "#fef2f2"
#         return (f"<div style='background:{bg};border-radius:4px;padding:3px 8px;"
#                 f"display:inline-block;min-width:110px'>"
#                 f"<div style='background:{col};width:{p}%;height:6px;"
#                 f"border-radius:3px;margin-bottom:2px'></div>"
#                 f"<span style='font-size:13px;font-weight:600'>{f:.3f}</span></div>")

#     rows = ""
#     for i, item in enumerate(per_q, 1):
#         if "error" in item:
#             rows += (f"<tr><td>{i}</td><td><span class='badge badge-{item.get('category','')}'>"
#                      f"{item.get('category','')}</span></td>"
#                      f"<td colspan='6' style='color:#dc2626'>❌ {item.get('error','')[:80]}</td></tr>")
#             continue
#         bd = ("" + ("<span class='bm bi'>🖼️</span>" if item.get("has_image_tag") else "")
#                  + ("<span class='bm bw'>🌐</span>" if item.get("web_fired") else ""))
#         rows += (f"<tr><td style='color:#6b7280;font-size:12px'>{i}</td>"
#                  f"<td><span class='badge badge-{item.get('category','')}' >"
#                  f"{item.get('category','')}</span></td>"
#                  f"<td style='font-size:13px;max-width:250px'>{item.get('question','')[:72]}{'...' if len(item.get('question',''))>72 else ''}{bd}</td>"
#                  f"<td style='font-size:11px;color:#6b7280'>{item.get('query_mode','')}</td>"
#                  f"<td style='font-size:12px'>{item.get('elapsed_s','?')}s</td>"
#                  f"<td style='font-size:12px;color:#6b7280'>{item.get('context_count','?')}</td>"
#                  f"<td>{_bar(item.get('ragas_faithfulness'))}</td>"
#                  f"<td>{_bar(item.get('ragas_answer_relevancy'),'#6366f1')}</td></tr>")

#     hist = ""
#     if HISTORY_PATH.exists():
#         with open(HISTORY_PATH) as hf:
#             for h in [json.loads(l) for l in hf if l.strip()][-10:]:
#                 sc = h.get("scores",{})
#                 hist += (f"<tr><td style='font-size:12px'>{h.get('timestamp','')[:16]}</td>"
#                          f"<td style='font-size:12px;font-weight:500'>{h.get('label','')}</td>"
#                          f"<td>{_bar(sc.get('faithfulness'))}</td>"
#                          f"<td>{_bar(sc.get('answer_relevancy'),'#6366f1')}</td>"
#                          f"<td>{_bar(sc.get('context_precision'),'#0891b2')}</td>"
#                          f"<td>{_bar(sc.get('context_recall'),'#7c3aed')}</td>"
#                          f"<td>{_bar(sc.get('composite'),'#1f2a44')}</td></tr>")

#     ts, label = r["timestamp"], r["label"]
#     img_v = c.get("image_retrieval_score"); web_v = c.get("web_fallback_score")

#     html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
# <title>MedChat RAGAS — {label}</title><style>
# *{{box-sizing:border-box;margin:0;padding:0}}
# body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f1f5f9;color:#1e293b;padding:24px}}
# .hdr{{background:linear-gradient(135deg,#1f2a44,#2fa36b);color:white;padding:24px 32px;border-radius:14px;margin-bottom:24px}}
# .hdr h1{{font-size:22px;margin-bottom:4px}}.hdr p{{font-size:13px;opacity:.8}}
# .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:14px;margin-bottom:24px}}
# .card{{background:white;border-radius:10px;padding:18px;box-shadow:0 1px 3px rgba(0,0,0,.07);text-align:center}}
# .card .val{{font-size:34px;font-weight:700;margin:8px 0 4px}}.card .lbl{{font-size:12px;color:#64748b}}
# .card.hl{{border:2px solid #6366f1}}
# table{{width:100%;border-collapse:collapse;background:white;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.07);margin-bottom:24px}}
# th{{background:#f8fafc;padding:10px 12px;text-align:left;font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.06em}}
# td{{padding:10px 12px;border-top:1px solid #f1f5f9}}tr:hover td{{background:#fafafa}}
# h2{{font-size:16px;font-weight:600;margin-bottom:12px}}
# .badge{{padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}}
# .badge-graph{{background:#dbeafe;color:#1e40af}}.badge-research{{background:#dcfce7;color:#166534}}
# .badge-image{{background:#fef3c7;color:#92400e}}.badge-edge{{background:#f3e8ff;color:#6b21a8}}
# .bm{{padding:1px 6px;border-radius:3px;font-size:11px;margin-left:4px}}
# .bi{{background:#dcfce7;color:#166534}}.bw{{background:#dbeafe;color:#1e40af}}
# </style></head><body>
# <div class="hdr"><h1>MedChat RAGAS Evaluation</h1>
# <p>Label: <strong>{label}</strong> | {ts[:16]} | {r['question_count']}q | RAGAS {r.get('ragas_version','?')} | Judge: <strong>{r['judge_model']}</strong></p></div>
# <div class="grid">
# <div class="card"><div class="lbl">Faithfulness</div><div class="val" style="color:{_col(s.get('faithfulness'))}">{s.get('faithfulness',0.0):.3f}</div><div class="lbl">No hallucination</div></div>
# <div class="card"><div class="lbl">Answer Relevancy</div><div class="val" style="color:{_col(s.get('answer_relevancy'))}">{s.get('answer_relevancy',0.0):.3f}</div><div class="lbl">On-topic</div></div>
# <div class="card"><div class="lbl">Context Precision</div><div class="val" style="color:{_col(s.get('context_precision'))}">{s.get('context_precision',0.0):.3f}</div><div class="lbl">Retrieval quality</div></div>
# <div class="card"><div class="lbl">Context Recall</div><div class="val" style="color:{_col(s.get('context_recall'))}">{s.get('context_recall',0.0):.3f}</div><div class="lbl">Coverage</div></div>
# <div class="card hl"><div class="lbl">Composite</div><div class="val" style="color:#6366f1">{s.get('composite',0.0):.3f}</div><div class="lbl">Overall</div></div>
# <div class="card"><div class="lbl">Image Retrieval</div><div class="val" style="color:{_col(img_v)}">{f"{img_v:.3f}" if img_v is not None else "N/A"}</div><div class="lbl">[IMAGE:] rate</div></div>
# <div class="card"><div class="lbl">Web Fallback</div><div class="val" style="color:{_col(web_v)}">{f"{web_v:.3f}" if web_v is not None else "N/A"}</div><div class="lbl">Fallback rate</div></div>
# </div>
# <h2>Per-question results</h2>
# <table><tr><th>#</th><th>Cat</th><th>Question</th><th>Mode</th><th>Time</th><th>Ctx</th><th>Faithfulness</th><th>Relevancy</th></tr>{rows}</table>
# <h2>Run history</h2>
# <table><tr><th>Timestamp</th><th>Label</th><th>Faithfulness</th><th>Relevancy</th><th>Precision</th><th>Recall</th><th>Composite</th></tr>
# {hist or '<tr><td colspan="7" style="color:#94a3b8;text-align:center;padding:16px">First run</td></tr>'}
# </table>
# <p style="color:#94a3b8;font-size:12px;margin-top:16px">MedChat RAGAS v3.1 | {ts}</p>
# </body></html>"""

#     with open(REPORT_PATH,"w",encoding="utf-8") as f: f.write(html)

# # =============================================================================
# # COMPARE + CLI
# # =============================================================================

# def compare_runs(a: str, b: str) -> None:
#     if not HISTORY_PATH.exists(): print("No history."); return
#     with open(HISTORY_PATH) as f:
#         hist = [json.loads(l) for l in f if l.strip()]
#     ra = next((h for h in reversed(hist) if h["label"]==a), None)
#     rb = next((h for h in reversed(hist) if h["label"]==b), None)
#     if not ra: print(f"'{a}' not found"); return
#     if not rb: print(f"'{b}' not found"); return
#     print(f"\n{'='*60}\n  {a} → {b}\n{'='*60}")
#     for m in ["faithfulness","answer_relevancy","context_precision","context_recall","composite"]:
#         va,vb = ra["scores"].get(m,0.0),rb["scores"].get(m,0.0); d=vb-va
#         print(f"  {'✅' if d>0.005 else '❌' if d<-0.005 else '  '} "
#               f"{m:<22} {va:.3f} → {vb:.3f}  ({'+' if d>=0 else ''}{d:.3f})")

# if __name__ == "__main__":
#     p = argparse.ArgumentParser(description="MedChat RAGAS Evaluation v3.1")
#     p.add_argument("--label",    default="baseline_with_graph_fix")
#     p.add_argument("--quick",    action="store_true")
#     p.add_argument("--category", choices=["graph","research","image","edge"])
#     p.add_argument("--compare",  nargs=2, metavar=("A","B"))
#     args = p.parse_args()
#     if args.compare: compare_runs(args.compare[0], args.compare[1])
#     else: run_evaluation(label=args.label, quick=args.quick, category=args.category)


# Attempt 4: The Ultimate Fix (with better context retrieval and category handling)

# =============================================================================
# cancer_evaluation.py  v4.0  (Pluggable LLM Architecture)
#
# RUN: python cancer_evaluation.py
# =============================================================================

from __future__ import annotations

import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

import json
import math
import time
import warnings
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# RAGAS INIT
# =============================================================================

_RAGAS_AVAILABLE = False
_ragas_ver = "unknown"

try:
    import ragas as _rp
    _ragas_ver = _rp.__version__
    
    from ragas import evaluate, RunConfig
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets import Dataset

    _RAGAS_AVAILABLE = True
    print(f"✅  RAGAS {_ragas_ver} ready")

except ImportError as _e:
    print(f"⚠️  RAGAS unavailable: {_e}")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel

from config import (
    QUERY_MODE_RESEARCH, QUERY_MODE_GRAPH, QUERY_MODE_AUTO,
    EMBEDDING_MODEL, JUDGE_LLM_PROVIDER, RAGAS_JUDGE_MODEL,
    GROQ_MODEL_QUERY
)
from cancer_retrieval import (
    generate_answer,
    _run_research_mode, _run_graph_mode, _run_auto_mode,
    _retrieve_image_chunks,
)

# =============================================================================
# LLM FACTORY (PLUGGABLE ARCHITECTURE)
# =============================================================================

def get_eval_llm(provider: str, model_name: str, **kwargs) -> BaseChatModel:
    """
    Dynamically loads the appropriate LangChain ChatModel based on provider.
    Supported: 'groq', 'google', 'openai'.
    """
    provider = provider.lower()
    temp = kwargs.get("temperature", 0.0)
    max_tokens = kwargs.get("max_tokens", 4096)

    if provider == "groq":
        from langchain_groq import ChatGroq
        from config import GROQ_API_KEY
        return ChatGroq(model=model_name, temperature=temp, api_key=GROQ_API_KEY, max_tokens=max_tokens)
        
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        from config import GOOGLE_API_KEY
        return ChatGoogleGenerativeAI(
            model=model_name, temperature=temp, google_api_key=GOOGLE_API_KEY, 
            max_output_tokens=max_tokens, timeout=120, max_retries=15
        )
        
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        from config import OPENAI_API_KEY
        return ChatOpenAI(model=model_name, temperature=temp, api_key=OPENAI_API_KEY, max_tokens=max_tokens)
        
    else:
        raise ValueError(f"❌ Unsupported LLM provider defined in config: {provider}")

# =============================================================================
# CONFIG
# =============================================================================

EVAL_DIR     = Path(__file__).parent / "output" / "evaluation"
SCORES_PATH  = EVAL_DIR / "ragas_scores.json"
REPORT_PATH  = EVAL_DIR / "ragas_report.html"
HISTORY_PATH = EVAL_DIR / "ragas_history.jsonl"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# GOLDEN DATASET LOADER
# =============================================================================

def load_golden_dataset(json_path="D:\\Desktop\\Neo_4J\\Medchat_Graph_RAG\\data\\corrected_golden_dataset.json"):
    dataset_path = Path(json_path)
    if not dataset_path.exists():
        print(f"❌ Error: Could not find {json_path}")
        return []

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    graph_qs, vector_qs, hybrid_qs, edge_qs = [], [], [], []

    for item in raw_data:
        meta = item.get("metadata", {})
        q_type = meta.get("type", "vector")
        
        mode = {
            "graph": QUERY_MODE_GRAPH,
            "vector": QUERY_MODE_RESEARCH,
            "image": QUERY_MODE_RESEARCH,
            "hybrid": QUERY_MODE_AUTO,
            "fallback": QUERY_MODE_AUTO
        }.get(q_type, QUERY_MODE_AUTO)

        formatted = {
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "query_mode": mode,
            "category": q_type,
            "description": f"{meta.get('difficulty', 'unknown')} - {meta.get('reasoning_type', '')}"
        }

        if q_type == "graph": graph_qs.append(formatted)
        elif q_type in ["vector", "image"]: vector_qs.append(formatted)
        elif q_type == "hybrid": hybrid_qs.append(formatted)
        elif q_type == "fallback": edge_qs.append(formatted)

    # 🟢 THE MINGLED 10 STRATEGY
    mingled_10 = graph_qs[:3] + vector_qs[:3] + hybrid_qs[:2] + edge_qs[:2]
    
    print(f"✅ Loaded {len(mingled_10)} questions for the Mingled Benchmark!")
    return mingled_10

TEST_SET = load_golden_dataset()

# =============================================================================
# HELPERS
# =============================================================================

def _get_contexts(q: str, mode: str) -> list[str]:
    try:
        fn = {QUERY_MODE_RESEARCH: _run_research_mode,
              QUERY_MODE_GRAPH:    _run_graph_mode}.get(mode, _run_auto_mode)
        
        context_text, docs, _, _ = fn(q, "", [], "")
        ctx = []
        
        if "## Graph Knowledge Base" in context_text:
            graph_part = context_text.split("[1] Source:")[0].strip()
            if graph_part:
                ctx.append(graph_part)
                
        ctx.extend([d.page_content for d in docs if d.page_content.strip()])
        
        for img in _retrieve_image_chunks(q):
            if img.page_content.strip():
                ctx.append(img.page_content)

        # 🟢 TPM FIX: Keep only top 2 contexts
        ctx = ctx[:2]
                
        return ctx or ["No context retrieved"]
    
    except Exception as e:
        return [f"Error: {e}"]

def _image_score(ans: str, cat: str) -> Optional[float]:
    if cat != "image": return None
    import re
    if re.search(r'\[IMAGE:\s*[^\]]+\]', ans, re.IGNORECASE): return 1.0
    if any(k in ans.lower() for k in ["figure","table","chart","flowchart","diagram"]): return 0.5
    return 0.0

def _web_score(ans: str, cat: str) -> Optional[float]:
    if cat != "edge": return None
    return 1.0 if ("🌐" in ans or "[W1]" in ans or "web search" in ans.lower()) else 0.0

def _safe(v) -> Optional[float]:
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None

def _mean_col(df, col: str) -> Optional[float]:
    if col not in df.columns: return None
    good = [_safe(v) for v in df[col].tolist() if _safe(v) is not None]
    return round(sum(good)/len(good), 4) if good else None

def _grade(v) -> str:
    f = _safe(v)
    if f is None or f == 0.0: return "❓  "
    if f >= 0.80: return "✅  "
    if f >= 0.60: return "⚠️  "
    return "❌  "

# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_evaluation(label="baseline", quick=False, category=None) -> dict:
    if not _RAGAS_AVAILABLE: return {}

    print("=" * 70)
    print(f"  MedChat RAGAS Evaluation  v4.0 (Pluggable Architecture)")
    print(f"  Pipeline : {GROQ_MODEL_QUERY}")
    print(f"  Judge    : {JUDGE_LLM_PROVIDER} / {RAGAS_JUDGE_MODEL}")
    print(f"  Label    : {label}")
    print(f"  Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    items = TEST_SET
    if category:
        items = [t for t in items if t["category"] == category]
    if quick:
        items = items[:5]
    print(f"  Total    : {len(items)} questions\n")

    # --- INITIALIZE THE PLUGGABLE JUDGE LLM ---
    lc_llm = get_eval_llm(JUDGE_LLM_PROVIDER, RAGAS_JUDGE_MODEL, temperature=0, max_tokens=4096)
    
    lc_emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
                                       model_kwargs={"device":"cpu"},
                                       encode_kwargs={"normalize_embeddings":True})
    ragas_llm = LangchainLLMWrapper(lc_llm)
    ragas_emb = LangchainEmbeddingsWrapper(lc_emb)
    print(f"  ✅  Judge Instantiated : {type(lc_llm).__name__}")
    print(f"  ✅  Embeddings         : {EMBEDDING_MODEL}\n")

    METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

    # ── Step 1: Generate answers ────────────────────────────────────────────
    print("  Step 1 — Generating answers...")
    questions, answers, contexts_list, ground_truths, per_question = [], [], [], [], []

    for i, item in enumerate(items, 1):
        q, gt, mode, cat = item["question"], item["ground_truth"], item["query_mode"], item["category"]
        print(f"  [{i:2d}/{len(items)}] {cat.upper():10s} | {q[:58]}...")
        try:
            t0 = time.time()
            ans, srcs = generate_answer(query=q, patient_report="", chat_history=[],
                                         cancer_filter="", query_mode=mode)
            el  = time.time() - t0
            ctx = _get_contexts(q, mode)
            ims = _image_score(ans, cat)
            ws  = _web_score(ans, cat)
            hi  = "[IMAGE:" in ans.upper()
            hw  = "🌐" in ans or "[W1]" in ans
            print(f"             ✅ {el:.1f}s | ctx={len(ctx)} | "
                  f"{'🖼️' if hi else '  '} {'🌐' if hw else ''} chars={len(ans)}")
            questions.append(q); answers.append(ans)
            contexts_list.append(ctx); ground_truths.append(gt)
            per_question.append({"question":q,"answer":ans,"ground_truth":gt,
                                  "query_mode":mode,"category":cat,
                                  "description":item["description"],
                                  "elapsed_s":round(el,2),"context_count":len(ctx),
                                  "has_image_tag":hi,"web_fired":hw,
                                  "image_score":ims,"web_score":ws,
                                  "sources":[s.get("label","") for s in srcs],
                                  "answer_preview":ans[:300]})
            time.sleep(2)
        except Exception as e:
            print(f"             ❌ {str(e)[:80]}")
            questions.append(q); answers.append("Error.")
            contexts_list.append(["Error"]); ground_truths.append(gt)
            per_question.append({"question":q,"error":str(e),
                                  "category":cat,"description":item["description"]})
            time.sleep(2)

    # ── Step 2: RAGAS scoring ───────────────────────────────────────────────
    # print(f"\n  Step 2 — RAGAS scoring ({RAGAS_JUDGE_MODEL})...")
    
    # ds = Dataset.from_dict({"question":questions,"answer":answers,
    #                         "contexts":contexts_list,"ground_truth":ground_truths})
    # ragas_scores: dict = {}
    # try:
    #     run_cfg = RunConfig(max_workers=1, timeout=800)
    #     result  = evaluate(dataset=ds, metrics=METRICS,
    #                        llm=ragas_llm, embeddings=ragas_emb,
    #                        run_config=run_cfg, raise_exceptions=False)
    #     df = result.to_pandas()
        
    #     fs = _mean_col(df,"faithfulness"); ars = _mean_col(df,"answer_relevancy")
    #     cps = _mean_col(df,"context_precision"); crs = _mean_col(df,"context_recall")
    #     valid = [s for s in [fs,ars,cps,crs] if s is not None]
    #     ragas_scores = {
    #         "faithfulness":      fs  or 0.0,
    #         "answer_relevancy":  ars or 0.0,
    #         "context_precision": cps or 0.0,
    #         "context_recall":    crs or 0.0,
    #         "composite": round(sum(valid)/len(valid),4) if valid else 0.0,
    #     }
    #     gi = 0
    #     for pq in per_question:
    #         if "error" not in pq and gi < len(df):
    #             row = df.iloc[gi]
    #             for c in ["faithfulness","answer_relevancy","context_precision","context_recall"]:
    #                 if c in df.columns: pq[f"ragas_{c}"] = _safe(row[c])
    #             gi += 1
    # except Exception as e:
    #     print(f"  ❌ RAGAS error: {e}"); traceback.print_exc()
    #     ragas_scores = {"composite":0.0,"error":str(e)}

    # ── Step 2: RAGAS scoring (One-by-One to bypass rate limits) ────────────
    print(f"\n  Step 2 — RAGAS scoring ({RAGAS_JUDGE_MODEL})...")
    
    ragas_score_lists = {"faithfulness": [], "answer_relevancy": [], 
                         "context_precision": [], "context_recall": []}
    
    for i, pq in enumerate(per_question):
        if "error" in pq:
            continue
            
        print(f"  Grading question {i+1}/{len(per_question)}...")
        
        # Create a tiny dataset of just ONE question
        single_ds = Dataset.from_dict({
            "question": [pq["question"]],
            "answer": [pq["answer"]],
            "contexts": [contexts_list[i]],
            "ground_truth": [ground_truths[i]]
        })
        
        try:
            # Short timeout, no deadlocks
            run_cfg = RunConfig(max_workers=1, timeout=300) 
            result  = evaluate(dataset=single_ds, metrics=METRICS,
                               llm=ragas_llm, embeddings=ragas_emb,
                               run_config=run_cfg, raise_exceptions=False)
            df = result.to_pandas()
            
            if not df.empty:
                row = df.iloc[0]
                for c in ["faithfulness","answer_relevancy","context_precision","context_recall"]:
                    if c in df.columns: 
                        val = _safe(row[c])
                        pq[f"ragas_{c}"] = val
                        if val is not None:
                            ragas_score_lists[c].append(val)
        except Exception as e:
            print(f"  ❌ RAGAS error on Q{i+1}: {e}")

        # 🟢 THE SILVER BULLET: Sleep for 15 seconds to reset the Per-Minute limits!
        if i < len(per_question) - 1:
            print("  Waiting 15s for API limits to cool down...")
            time.sleep(15)

    # Calculate final averages
    ragas_scores = {}
    for k, v_list in ragas_score_lists.items():
        ragas_scores[k] = sum(v_list) / len(v_list) if v_list else 0.0
        
    valid_means = [v for v in ragas_scores.values() if v > 0]
    ragas_scores["composite"] = sum(valid_means)/len(valid_means) if valid_means else 0.0

    # ── Step 3: Save Data ──────────────────────────────────────────────────
    img_items  = [p for p in per_question if p.get("category")=="image" and "error" not in p]
    edge_items = [p for p in per_question if p.get("category")=="edge"  and "error" not in p]
    img_ss  = [p["image_score"] for p in img_items  if p.get("image_score") is not None]
    web_ss  = [p["web_score"]   for p in edge_items if p.get("web_score")   is not None]
    
    cat_stats: dict = {}
    for cat in ["graph","research","image","edge"]:
        its = [p for p in per_question if p.get("category")==cat]
        if its:
            ok = [p for p in its if "error" not in p]
            cat_stats[cat] = {"count":len(its),"success":len(ok)}

    full = {"label":label,"timestamp":datetime.now().isoformat(),
            "judge_model":RAGAS_JUDGE_MODEL, "pipeline_model":GROQ_MODEL_QUERY,
            "ragas_scores":ragas_scores,
            "custom_scores":{"image_retrieval_score": round(sum(img_ss)/len(img_ss),4) if img_ss else None,
                             "web_fallback_score": round(sum(web_ss)/len(web_ss),4) if web_ss else None},
            "category_stats":cat_stats,"per_question":per_question}

    with open(SCORES_PATH,"w",encoding="utf-8") as f: json.dump(full,f,indent=2,ensure_ascii=False)
    with open(HISTORY_PATH,"a",encoding="utf-8") as f:
        f.write(json.dumps({"label":label,"timestamp":full["timestamp"], "scores":ragas_scores})+"\n")

    _html_report(full)
    _print_summary(full)
    return full

def _print_summary(r: dict) -> None:
    s = r["ragas_scores"]
    print(f"\n{'='*70}\n  RAGAS RESULTS — {r['label']} \n{'='*70}\n  Scores:")
    for k,lbl in [("faithfulness","Faithfulness     "),("answer_relevancy","Answer Relevancy "),
                  ("context_precision","Context Precision"),("context_recall","Context Recall   ")]:
        v = s.get(k, 0.0)
        print(f"  {_grade(v)}{lbl}  {v:.3f}")
    print(f"\n  ── Composite: {s.get('composite',0.0):.3f}\n{'='*70}")

def _html_report(r: dict) -> None:
    pass # Leaving existing HTML untouched, just minimized for snippet length. (Keep your existing HTML function here!)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MedChat RAGAS Evaluation v4.0")
    p.add_argument("--label",    default="baseline_pluggable")
    p.add_argument("--quick",    action="store_true")
    p.add_argument("--category", choices=["graph","research","image","edge"])
    args = p.parse_args()
    run_evaluation(label=args.label, quick=args.quick, category=args.category)