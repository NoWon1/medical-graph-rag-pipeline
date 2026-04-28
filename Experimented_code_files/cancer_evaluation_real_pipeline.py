# """
# =========================================================
# REAL GRAPH RAG EVALUATION SCRIPT
# =========================================================

# - Uses your real pipeline (cancer_retrieval.py)
# - Evaluates retrieval + generation + graph reasoning
# - No dummy components
# - Designed for Neo4j + BM25 + Graph RAG + Groq
# =========================================================
# """

# import json
# import csv
# import numpy as np
# from typing import List, Dict
# from dataclasses import dataclass

# # ✅ IMPORT YOUR REAL PIPELINE
# from cancer_retrieval import generate_answer, get_embeddings

# # =========================================================
# # CONFIGURATION
# # =========================================================

# TOP_K_EVAL = 5

# WEIGHTS_VECTOR = {
#     "semantic_similarity": 0.4,
#     "faithfulness": 0.3,
#     "context_recall": 0.3,
# }

# WEIGHTS_GRAPH = {
#     "semantic_similarity": 0.3,
#     "faithfulness": 0.2,
#     "context_recall": 0.2,
#     "graph_score": 0.3,
# }

# # =========================================================
# # DATA STRUCTURE
# # =========================================================

# @dataclass
# class EvaluationResult:
#     question: str
#     answer: str
#     ground_truth: str
#     sample_type: str

#     semantic_similarity: float
#     faithfulness: float
#     answer_relevance: float
#     context_recall: float
#     graph_score: float

#     final_score: float
#     failure_type: str

# # =========================================================
# # DATASET LOADER
# # =========================================================

# def load_dataset(path: str) -> List[Dict]:
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

# # =========================================================
# # REAL PIPELINE ADAPTER
# # =========================================================

# class RealRAGPipeline:
#     """
#     Adapter that allows evaluation code to call
#     your real Graph RAG pipeline unchanged.
#     """

#     def __init__(self, cancer_filter="", query_mode="auto"):
#         self.cancer_filter = cancer_filter
#         self.query_mode = query_mode

#     def run(self, query: str):
#         answer, sources = generate_answer(
#             query=query,
#             patient_report="",
#             chat_history=[],
#             cancer_filter=self.cancer_filter,
#             query_mode=self.query_mode,
#         )

#         # Convert sources → contexts
#         contexts = [s["label"] for s in sources]

#         return contexts, answer

# # =========================================================
# # METRICS
# # =========================================================

# def cosine_similarity(v1, v2) -> float:
#     v1 = np.array(v1)
#     v2 = np.array(v2)
#     if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
#         return 0.0
#     return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# def compute_semantic_similarity(answer, gt, embedder) -> float:
#     return cosine_similarity(
#         embedder.embed_query(answer),
#         embedder.embed_query(gt)
#     )

# def evaluate_context_recall(gt_contexts, retrieved_contexts, embedder) -> float:
#     def best_sim(gt):
#         gt_emb = embedder.embed_query(gt)
#         sims = [
#             cosine_similarity(gt_emb, embedder.embed_query(rc))
#             for rc in retrieved_contexts
#         ]
#         return max(sims) if sims else 0.0

#     scores = [best_sim(gt) for gt in gt_contexts]
#     return sum(scores) / len(scores) if scores else 0.0

# def evaluate_graph(answer: str, gt_contexts: List[str]) -> float:
#     score, total = 0, 0
#     for ctx in gt_contexts:
#         if "Graph" in ctx or "DrugInteraction" in ctx:
#             total += 1
#             if any(tok in answer.lower() for tok in ctx.lower().split()):
#                 score += 1
#     return score / total if total else 0.0

# def evaluate_ragas_light(answer: str) -> Dict:
#     """
#     Lightweight RAGAS proxy:
#     - faithfulness ≈ answer groundedness
#     - relevance ≈ answer length heuristic

#     Replace with actual RAGAS if needed.
#     """
#     faithfulness = 1.0 if len(answer) > 150 else 0.6
#     relevance = min(1.0, len(answer) / 400)

#     return {
#         "faithfulness": faithfulness,
#         "answer_relevance": relevance,
#     }

# # =========================================================
# # FINAL SCORING
# # =========================================================

# def compute_final_score(sample_type: str, metrics: Dict) -> float:
#     weights = WEIGHTS_GRAPH if sample_type == "graph" else WEIGHTS_VECTOR
#     return sum(weights[k] * metrics.get(k, 0) for k in weights)

# def classify_failure(m: Dict) -> str:
#     if m["context_recall"] < 0.5:
#         return "retrieval_failure"
#     if m["faithfulness"] < 0.6:
#         return "hallucination"
#     if m["semantic_similarity"] < 0.6:
#         return "generation_gap"
#     if m.get("graph_score", 1) < 0.6:
#         return "graph_reasoning_failure"
#     return "success"

# # =========================================================
# # MAIN EVALUATION LOOP
# # =========================================================

# def run_evaluation(dataset_path: str):
#     dataset = load_dataset(dataset_path)

#     pipeline = RealRAGPipeline(query_mode="auto")
#     embedder = get_embeddings()

#     results = []

#     for sample in dataset:
#         q = sample["question"]
#         gt = sample["ground_truth"]
#         gt_ctx = sample["contexts"]
#         sample_type = sample["metadata"]["type"]

#         retrieved_contexts, answer = pipeline.run(q)

#         context_recall = evaluate_context_recall(gt_ctx, retrieved_contexts, embedder)
#         semantic_similarity = compute_semantic_similarity(answer, gt, embedder)
#         ragas = evaluate_ragas_light(answer)

#         graph_score = evaluate_graph(answer, gt_ctx) if sample_type == "graph" else 0.0

#         metrics = {
#             "semantic_similarity": semantic_similarity,
#             "faithfulness": ragas["faithfulness"],
#             "answer_relevance": ragas["answer_relevance"],
#             "context_recall": context_recall,
#             "graph_score": graph_score,
#         }

#         final_score = compute_final_score(sample_type, metrics)
#         failure = classify_failure(metrics)

#         results.append(EvaluationResult(
#             question=q,
#             answer=answer,
#             ground_truth=gt,
#             sample_type=sample_type,
#             final_score=final_score,
#             failure_type=failure,
#             **metrics
#         ))

#     return results

# # =========================================================
# # ENTRY POINT
# # =========================================================

# if __name__ == "__main__":

#     DATASET_PATH = r"D:\Desktop\Neo_4J\Medchat_Graph_RAG\data\corrected_golden_dataset.json"

#     print("\n🚀 Running REAL Graph RAG Evaluation...\n")
#     results = run_evaluation(DATASET_PATH)

#     print(f"✅ Completed: {len(results)} samples\n")

#     # SUMMARY
#     avg = sum(r.final_score for r in results) / len(results)
#     print(f"📊 Average score: {avg:.3f}")

#     failures = {}
#     for r in results:
#         failures[r.failure_type] = failures.get(r.failure_type, 0) + 1

#     print("\nFailure distribution:")
#     for k, v in failures.items():
#         print(f"  {k}: {v}")

#     # SAVE CSV
#     with open("evaluation_results_real_pipeline.csv", "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "question",
#             "final_score",
#             "failure_type",
#             "semantic_similarity",
#             "faithfulness",
#             "answer_relevance",
#             "context_recall",
#             "graph_score",
#         ])
#         for r in results:
#             writer.writerow([
#                 r.question, r.final_score, r.failure_type,
#                 r.semantic_similarity, r.faithfulness,
#                 r.answer_relevance, r.context_recall, r.graph_score,
#             ])

#     print("\n💾 Saved: evaluation_results_real_pipeline.csv")
#     print("\n🎯 DONE.")

# Attempt 2

"""
=========================================================
REAL GRAPH RAG EVALUATION (LIMITED SAMPLES)
=========================================================

✅ Uses your REAL pipeline (cancer_retrieval.py)
✅ Neo4j + BM25 + Graph RAG + Groq
✅ Option to test ONLY first N questions (5 / 10)
✅ Fast, cheap initial testing before full eval
=========================================================
"""

import json
import csv
import numpy as np
from typing import List, Dict
from dataclasses import dataclass

# =========================================================
# 🔧 USER CONFIG (CHANGE HERE ONLY)
# =========================================================

DATASET_PATH = r"D:\Desktop\Neo_4J\Medchat_Graph_RAG\data\corrected_golden_dataset.json"

MAX_EVAL_SAMPLES = 5     # ✅ set to 5 or 10 (or None for full dataset)
QUERY_MODE = "auto"     # auto / research / graph

# =========================================================
# ✅ IMPORT YOUR REAL PIPELINE
# =========================================================

from cancer_retrieval import generate_answer, get_embeddings

# =========================================================
# SCORING CONFIG
# =========================================================

WEIGHTS_VECTOR = {
    "semantic_similarity": 0.4,
    "faithfulness": 0.3,
    "context_recall": 0.3,
}

WEIGHTS_GRAPH = {
    "semantic_similarity": 0.3,
    "faithfulness": 0.2,
    "context_recall": 0.2,
    "graph_score": 0.3,
}

# =========================================================
# DATA STRUCTURE
# =========================================================

@dataclass
class EvaluationResult:
    question: str
    answer: str
    ground_truth: str
    sample_type: str
    semantic_similarity: float
    faithfulness: float
    answer_relevance: float
    context_recall: float
    graph_score: float
    final_score: float
    failure_type: str

# =========================================================
# HELPERS
# =========================================================

def load_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if MAX_EVAL_SAMPLES:
        print(f"⚠️ Limiting evaluation to first {MAX_EVAL_SAMPLES} samples\n")
        data = data[:MAX_EVAL_SAMPLES]
    return data

def cosine_similarity(v1, v2) -> float:
    v1 = np.array(v1)
    v2 = np.array(v2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# =========================================================
# REAL PIPELINE ADAPTER
# =========================================================

class RealRAGPipeline:
    """
    Adapter to plug cancer_retrieval.generate_answer()
    into evaluation pipeline.
    """

    def __init__(self, query_mode="auto"):
        self.query_mode = query_mode

    def run(self, query: str):
        answer, sources = generate_answer(
            query=query,
            patient_report="",
            chat_history=[],
            cancer_filter="",
            query_mode=self.query_mode,
        )
        contexts = [s["label"] for s in sources]
        return contexts, answer

# =========================================================
# METRICS
# =========================================================

def compute_semantic_similarity(answer, gt, embedder) -> float:
    return cosine_similarity(
        embedder.embed_query(answer),
        embedder.embed_query(gt),
    )

def evaluate_context_recall(gt_contexts, retrieved_contexts, embedder) -> float:
    def best_sim(gt):
        gt_emb = embedder.embed_query(gt)
        sims = [
            cosine_similarity(gt_emb, embedder.embed_query(rc))
            for rc in retrieved_contexts
        ]
        return max(sims) if sims else 0.0

    scores = [best_sim(gt) for gt in gt_contexts]
    return sum(scores) / len(scores) if scores else 0.0

def evaluate_graph(answer: str, gt_contexts: List[str]) -> float:
    score, total = 0, 0
    for ctx in gt_contexts:
        if "Graph" in ctx or "DrugInteraction" in ctx:
            total += 1
            if any(tok in answer.lower() for tok in ctx.lower().split()):
                score += 1
    return score / total if total else 0.0

def ragas_proxy(answer: str) -> Dict:
    """
    Lightweight proxy (replace with full RAGAS later)
    """
    return {
        "faithfulness": 1.0 if len(answer) > 150 else 0.6,
        "answer_relevance": min(1.0, len(answer) / 400),
    }

# =========================================================
# FINAL SCORE
# =========================================================

def compute_final_score(sample_type: str, metrics: Dict) -> float:
    weights = WEIGHTS_GRAPH if sample_type == "graph" else WEIGHTS_VECTOR
    return sum(weights[k] * metrics.get(k, 0) for k in weights)

def classify_failure(m: Dict) -> str:
    if m["context_recall"] < 0.5:
        return "retrieval_failure"
    if m["faithfulness"] < 0.6:
        return "hallucination"
    if m["semantic_similarity"] < 0.6:
        return "generation_gap"
    if m.get("graph_score", 1) < 0.6:
        return "graph_reasoning_failure"
    return "success"

# =========================================================
# MAIN EVALUATION LOOP
# =========================================================

def run_evaluation():
    dataset = load_dataset(DATASET_PATH)

    pipeline = RealRAGPipeline(query_mode=QUERY_MODE)
    embedder = get_embeddings()

    results = []

    for i, sample in enumerate(dataset, 1):
        print(f"🔍 [{i}/{len(dataset)}] {sample['question'][:80]}...")

        q = sample["question"]
        gt = sample["ground_truth"]
        gt_ctx = sample["contexts"]
        sample_type = sample["metadata"]["type"]

        retrieved_ctx, answer = pipeline.run(q)

        metrics = {}
        metrics["context_recall"] = evaluate_context_recall(gt_ctx, retrieved_ctx, embedder)
        metrics["semantic_similarity"] = compute_semantic_similarity(answer, gt, embedder)

        ragas = ragas_proxy(answer)
        metrics["faithfulness"] = ragas["faithfulness"]
        metrics["answer_relevance"] = ragas["answer_relevance"]

        metrics["graph_score"] = (
            evaluate_graph(answer, gt_ctx)
            if sample_type == "graph" else 0.0
        )

        final_score = compute_final_score(sample_type, metrics)
        failure = classify_failure(metrics)

        results.append(EvaluationResult(
            question=q,
            answer=answer,
            ground_truth=gt,
            sample_type=sample_type,
            final_score=final_score,
            failure_type=failure,
            **metrics
        ))

    return results

# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    print("\n🚀 Starting LIMITED Graph RAG Evaluation\n")
    results = run_evaluation()

    print(f"\n✅ Completed {len(results)} samples")

    avg_score = sum(r.final_score for r in results) / len(results)
    print(f"📊 Average score: {avg_score:.3f}")

    failures = {}
    for r in results:
        failures[r.failure_type] = failures.get(r.failure_type, 0) + 1

    print("\nFailure distribution:")
    for k, v in failures.items():
        print(f"  {k}: {v}")

    with open("evaluation_results_limited.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question",
            "final_score",
            "failure_type",
            "semantic_similarity",
            "faithfulness",
            "answer_relevance",
            "context_recall",
            "graph_score",
        ])
        for r in results:
            writer.writerow([
                r.question,
                r.final_score,
                r.failure_type,
                r.semantic_similarity,
                r.faithfulness,
                r.answer_relevance,
                r.context_recall,
                r.graph_score,
            ])

    print("\n💾 Saved: evaluation_results_limited.csv")
    print("\n🎯 DONE.")