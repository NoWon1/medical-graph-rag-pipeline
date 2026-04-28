import json
from typing import List, Dict, Any
from dataclasses import dataclass

# ================================
# CONFIGURATION
# ================================

TOP_K_EVAL = 5  # Limit context for RAGAS to avoid truncation

WEIGHTS_VECTOR = {
    "semantic_similarity": 0.4,
    "faithfulness": 0.3,
    "context_recall": 0.3
}

WEIGHTS_GRAPH = {
    "semantic_similarity": 0.3,
    "faithfulness": 0.2,
    "context_recall": 0.2,
    "graph_score": 0.3
}


# ================================
# DATA STRUCTURE
# ================================

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


# ================================
# DATASET LOADER
# ================================

def load_dataset(path: str) -> List[Dict]:
    """
    Load golden dataset JSON file

    Expected format:
    - question
    - ground_truth
    - contexts
    - metadata.type (vector / graph)
    """
    with open(path, "r") as f:
        return json.load(f)


# ================================
# RETRIEVAL PIPELINE WRAPPER
# ================================

def run_retrieval(pipeline, query: str):
    """
    Wrapper over your existing retrieval pipeline.

    Should return:
    - retrieved_docs: list of documents
    - answer: generated answer
    """
    response = pipeline.run(query)

    return response["documents"], response["answer"]


# ================================
# RETRIEVAL EVALUATION
# ================================

def evaluate_context_recall(gt_contexts: List[str], retrieved_contexts: List[str], embedder) -> float:
    """
    Compare GT contexts vs retrieved contexts using embedding similarity

    Measures:
    - Did retriever fetch relevant information?
    """

    def max_similarity(gt, retrieved_list):
        gt_emb = embedder.embed(gt)
        sims = []
        for ctx in retrieved_list:
            ctx_emb = embedder.embed(ctx)
            sims.append(cosine_similarity(gt_emb, ctx_emb))
        return max(sims) if sims else 0

    scores = [max_similarity(gt, retrieved_contexts) for gt in gt_contexts]

    return sum(scores) / len(scores)


# ================================
# SEMANTIC SIMILARITY
# ================================

def compute_semantic_similarity(answer: str, ground_truth: str, embedder) -> float:
    """
    Compare answer with ground truth

    This is your GOLDEN DATASET anchor metric
    """
    a_emb = embedder.embed(answer)
    gt_emb = embedder.embed(ground_truth)

    return cosine_similarity(a_emb, gt_emb)


# ================================
# RAGAS EVALUATION
# ================================

def evaluate_ragas_metrics(question: str, answer: str, contexts: List[str], ragas_evaluator) -> Dict:
    """
    Run RAGAS evaluation

    Returns:
    - faithfulness
    - answer_relevance

    IMPORTANT:
    Only pass TOP-K contexts to avoid token overflow
    """

    contexts = contexts[:TOP_K_EVAL]

    result = ragas_evaluator.evaluate(
        question=question,
        answer=answer,
        contexts=contexts
    )

    return {
        "faithfulness": result["faithfulness"],
        "answer_relevance": result["answer_relevance"]
    }


# ================================
# GRAPH EVALUATION (CUSTOM)
# ================================

def evaluate_graph(answer: str, gt_contexts: List[str]) -> float:
    """
    Custom evaluation for graph-based samples

    Strategy:
    - Extract entities and relations from answer
    - Match against GT graph facts
    """

    # SIMPLE VERSION (you can improve later)

    score = 0
    total = 0

    for ctx in gt_contexts:
        if "Graph" in ctx or "DrugInteraction" in ctx:
            total += 1
            if any(token in answer.lower() for token in ctx.lower().split()):
                score += 1

    return score / total if total > 0 else 0


# ================================
# FINAL SCORE COMPUTATION
# ================================

def compute_final_score(sample_type: str, metrics: Dict) -> float:
    """
    Weighted scoring depending on sample type
    """

    if sample_type == "vector":
        weights = WEIGHTS_VECTOR
    else:
        weights = WEIGHTS_GRAPH

    score = 0
    for key, weight in weights.items():
        score += weight * metrics.get(key, 0)

    return score


# ================================
# FAILURE CLASSIFICATION
# ================================

def classify_failure(metrics: Dict) -> str:
    """
    Identify root cause of failure

    This is the MOST IMPORTANT part for debugging
    """

    if metrics["context_recall"] < 0.5:
        return "retrieval_failure"

    elif metrics["faithfulness"] < 0.6:
        return "hallucination"

    elif metrics["semantic_similarity"] < 0.6:
        return "generation_gap"

    elif metrics.get("graph_score", 1) < 0.6:
        return "graph_reasoning_failure"

    else:
        return "success"


# ================================
# MAIN EVALUATION LOOP
# ================================

def run_evaluation(dataset_path: str, pipeline, embedder, ragas_evaluator):
    """
    Main evaluation pipeline

    Flow:
    1. Load dataset
    2. Run retrieval + generation
    3. Evaluate all metrics
    4. Classify failure
    5. Store results
    """

    dataset = load_dataset(dataset_path)

    results = []

    for sample in dataset:

        question = sample["question"]
        ground_truth = sample["ground_truth"]
        gt_contexts = sample["contexts"]
        sample_type = sample["metadata"]["type"]

        # -------------------------------
        # RUN PIPELINE
        # -------------------------------
        retrieved_docs, answer = run_retrieval(pipeline, question)

        retrieved_contexts = [doc.page_content for doc in retrieved_docs]

        # -------------------------------
        # RETRIEVAL EVALUATION
        # -------------------------------
        context_recall = evaluate_context_recall(
            gt_contexts,
            retrieved_contexts,
            embedder
        )

        # -------------------------------
        # GENERATION EVALUATION
        # -------------------------------
        semantic_similarity = compute_semantic_similarity(
            answer,
            ground_truth,
            embedder
        )

        ragas_scores = evaluate_ragas_metrics(
            question,
            answer,
            retrieved_contexts,
            ragas_evaluator
        )

        # -------------------------------
        # GRAPH EVALUATION (if needed)
        # -------------------------------
        graph_score = 0
        if sample_type == "graph":
            graph_score = evaluate_graph(answer, gt_contexts)

        # -------------------------------
        # COMBINE METRICS
        # -------------------------------
        metrics = {
            "semantic_similarity": semantic_similarity,
            "faithfulness": ragas_scores["faithfulness"],
            "answer_relevance": ragas_scores["answer_relevance"],
            "context_recall": context_recall,
            "graph_score": graph_score
        }

        final_score = compute_final_score(sample_type, metrics)

        failure_type = classify_failure(metrics)

        # -------------------------------
        # STORE RESULT
        # -------------------------------
        result = EvaluationResult(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            sample_type=sample_type,
            semantic_similarity=semantic_similarity,
            faithfulness=ragas_scores["faithfulness"],
            answer_relevance=ragas_scores["answer_relevance"],
            context_recall=context_recall,
            graph_score=graph_score,
            final_score=final_score,
            failure_type=failure_type
        )

        results.append(result)

    return results


# ================================
# HELPER: COSINE SIMILARITY
# ================================

def cosine_similarity(vec1, vec2):
    """
    Basic cosine similarity function
    """
    import numpy as np

    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# ================================
# MAIN EXECUTION BLOCK
# ================================

if __name__ == "__main__":
    """
    Entry point to run full evaluation

    You need to plug:
    - Your RAG pipeline object
    - Your embedding model
    - Your RAGAS evaluator
    """

    print("\n🚀 Starting RAG Evaluation Pipeline...\n")

    # =========================================
    # 🔌 STEP 1: LOAD YOUR COMPONENTS
    # =========================================

    # ---- Replace these with your actual implementations ----

    class DummyEmbedder:
        """
        Replace with your real embedding model
        Example: SentenceTransformer, OpenAIEmbeddings, etc.
        """
        def embed(self, text):
            import numpy as np
            return np.random.rand(384)  # placeholder


    class DummyRAGASEvaluator:
        """
        Replace with actual RAGAS evaluator
        """
        def evaluate(self, question, answer, contexts):
            import random
            return {
                "faithfulness": random.uniform(0.5, 1.0),
                "answer_relevance": random.uniform(0.5, 1.0)
            }


    class DummyPipeline:
        """
        Replace with your actual pipeline:
        pipeline.run(query) → {"documents": [...], "answer": "..."}
        """
        def run(self, query):
            class Doc:
                def __init__(self, content):
                    self.page_content = content

            docs = [Doc(f"Dummy context for: {query}") for _ in range(5)]

            return {
                "documents": docs,
                "answer": f"Dummy answer for: {query}"
            }


    # Initialize components
    pipeline = DummyPipeline()
    embedder = DummyEmbedder()
    ragas_evaluator = DummyRAGASEvaluator()

    # Dataset path (UPDATE THIS)
    dataset_path = "ragas_dataset.json"

    # =========================================
    # 🧪 STEP 2: RUN EVALUATION
    # =========================================

    results = run_evaluation(
        dataset_path=r"D:\Desktop\Neo_4J\Medchat_Graph_RAG\data\corrected_golden_dataset.json",
        pipeline=pipeline,
        embedder=embedder,
        ragas_evaluator=ragas_evaluator
    )

    print("\n✅ Evaluation Completed!\n")

    # =========================================
    # 📊 STEP 3: PRINT SAMPLE RESULTS
    # =========================================

    for i, res in enumerate(results[:5]):  # show first 5
        print(f"\n🔹 Sample {i+1}")
        print(f"Question: {res.question}")
        print(f"Answer: {res.answer[:100]}...")
        print(f"Final Score: {round(res.final_score, 3)}")
        print(f"Failure Type: {res.failure_type}")
        print("-" * 50)

    # =========================================
    # 📈 STEP 4: SUMMARY METRICS
    # =========================================

    total = len(results)

    avg_score = sum(r.final_score for r in results) / total

    failure_counts = {}
    for r in results:
        failure_counts[r.failure_type] = failure_counts.get(r.failure_type, 0) + 1

    print("\n📊 SUMMARY")
    print("=" * 50)
    print(f"Total Samples: {total}")
    print(f"Average Score: {round(avg_score, 3)}")

    print("\nFailure Distribution:")
    for k, v in failure_counts.items():
        print(f"  {k}: {v}")

    # =========================================
    # 💾 STEP 5: SAVE RESULTS
    # =========================================

    import csv

    output_file = "evaluation_results.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "question",
            "answer",
            "final_score",
            "failure_type",
            "semantic_similarity",
            "faithfulness",
            "answer_relevance",
            "context_recall",
            "graph_score"
        ])

        # Rows
        for r in results:
            writer.writerow([
                r.question,
                r.answer,
                r.final_score,
                r.failure_type,
                r.semantic_similarity,
                r.faithfulness,
                r.answer_relevance,
                r.context_recall,
                r.graph_score
            ])

    print(f"\n💾 Results saved to: {output_file}")

    print("\n🎯 DONE.")