"""
ragas_test.py — Run this FIRST before cancer_evaluation.py
Finds the exact RAGAS API pattern that works on your machine.

Usage:
    python ragas_test.py

This takes ~2 minutes and tells you exactly which strategy works.
"""

import os, sys, warnings, importlib
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
JUDGE_MODEL  = "llama-3.1-8b-instant"
EMBED_MODEL  = "BAAI/bge-base-en-v1.5"

print("=" * 60)
print("  RAGAS API Compatibility Test")
print("=" * 60)

# Step 1: Check RAGAS version
try:
    import ragas
    print(f"\n✅ RAGAS version: {ragas.__version__}")
except ImportError:
    print("❌ RAGAS not installed. Run: pip install ragas datasets openai")
    sys.exit(1)

# Step 2: Import evaluate
from ragas import evaluate
from datasets import Dataset

# Step 3: Build a tiny test dataset (1 question)
ds = Dataset.from_dict({
    "question":    ["What causes nausea during cisplatin chemotherapy?"],
    "answer":      ["Cisplatin commonly causes nausea and vomiting due to its "
                    "effect on the gastrointestinal tract and the chemoreceptor "
                    "trigger zone. Anti-emetic premedication is required."],
    "contexts":    [["Cisplatin causes nausea and vomiting. It affects the "
                     "chemoreceptor trigger zone. Anti-emetics are required."]],
    "ground_truth":["Cisplatin causes nausea by stimulating the chemoreceptor "
                    "trigger zone. Anti-emetic premedication is mandatory."],
})
print(f"✅ Test dataset: 1 question ready")

# Step 4: Try building the LLM
ragas_llm = None
ragas_emb  = None

print(f"\n--- Building RAGAS LLM ({JUDGE_MODEL}) ---")

# Method A: llm_factory with Groq OpenAI-compatible endpoint
try:
    from ragas.llms import llm_factory
    from openai import OpenAI
    client = OpenAI(api_key=GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1")
    ragas_llm = llm_factory(JUDGE_MODEL, client=client)
    print(f"✅ Method A: llm_factory → {type(ragas_llm).__name__}")
except Exception as e:
    print(f"❌ Method A failed: {e}")

# Method B: LangchainLLMWrapper (deprecated but may still work)
if ragas_llm is None:
    try:
        from ragas.llms import LangchainLLMWrapper
        from langchain_groq import ChatGroq
        _lc = ChatGroq(model=JUDGE_MODEL, temperature=0, api_key=GROQ_API_KEY)
        ragas_llm = LangchainLLMWrapper(_lc)
        print(f"✅ Method B: LangchainLLMWrapper → {type(ragas_llm).__name__}")
    except Exception as e:
        print(f"❌ Method B failed: {e}")

if ragas_llm is None:
    print("❌ Could not build RAGAS LLM. Check GROQ_API_KEY in .env")
    sys.exit(1)

# Step 5: Try building embeddings
print(f"\n--- Building RAGAS Embeddings ({EMBED_MODEL}) ---")

# Method A: RAGAS native HuggingFace (requires sentence-transformers)
try:
    from ragas.embeddings import HuggingFaceEmbeddings as RagasHFE
    ragas_emb = RagasHFE(model=EMBED_MODEL)
    print(f"✅ Method A: RagasHFEmbeddings → {type(ragas_emb).__name__}")
except Exception as e:
    print(f"❌ Method A failed: {e}")
    if "sentence" in str(e).lower():
        print("   → Fix: pip install sentence-transformers")

# Method B: LangchainEmbeddingsWrapper
if ragas_emb is None:
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_huggingface import HuggingFaceEmbeddings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        _hf = HuggingFaceEmbeddings(model_name=EMBED_MODEL,
                                     model_kwargs={"device": "cpu"},
                                     encode_kwargs={"normalize_embeddings": True})
        ragas_emb = LangchainEmbeddingsWrapper(_hf)
        print(f"✅ Method B: LangchainEmbeddingsWrapper → {type(ragas_emb).__name__}")
    except Exception as e:
        print(f"❌ Method B failed: {e}")

if ragas_emb is None:
    print("❌ Could not build RAGAS embeddings")
    sys.exit(1)

# Step 6: Try building metrics — prefer the API pinned in requirements.txt
print(f"\n--- Building Metrics ---")
working_metrics = None

# Strategy A: old-style singleton metrics (ragas==0.2.6)
try:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from ragas.metrics import (
        faithfulness, answer_relevancy,
        context_precision, context_recall,
    )
    working_metrics = [
        faithfulness, answer_relevancy,
        context_precision, context_recall,
    ]
    print(f"✅ Strategy A: old-style singleton metrics")
    print(f"   Types: {[type(m).__name__ for m in working_metrics]}")
except Exception as e:
    print(f"❌ Strategy A: {e}")

# Strategy B: class-based metrics (for newer RAGAS versions)
if working_metrics is None:
    try:
        metrics_mod = importlib.import_module("ragas.metrics.collections")
        Faithfulness = getattr(metrics_mod, "Faithfulness")
        AnswerRelevancy = getattr(metrics_mod, "AnswerRelevancy")
        ContextPrecision = getattr(metrics_mod, "ContextPrecision")
        ContextRecall = getattr(metrics_mod, "ContextRecall")
        f  = Faithfulness(llm=ragas_llm)
        ar = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb)
        cp = ContextPrecision(llm=ragas_llm)
        cr = ContextRecall(llm=ragas_llm)
        working_metrics = [f, ar, cp, cr]
        print(f"✅ Strategy B: collections metrics with llm in constructor")
    except Exception as e:
        print(f"❌ Strategy B: {e}")

if working_metrics is None:
    print("❌ Cannot build any metrics")
    sys.exit(1)

# Step 7: Check isinstance(metric, Metric) — this is what evaluate() checks
print(f"\n--- Checking isinstance(metric, Metric) ---")
try:
    from ragas.metrics.base import Metric
    for m in working_metrics:
        is_ok = isinstance(m, Metric)
        print(f"  {type(m).__name__}: isinstance(Metric) = {is_ok}")
except Exception as e:
    print(f"⚠️  Could not check: {e}")

# Step 8: Run evaluate() — try all call patterns
print(f"\n--- Testing evaluate() call ---")
from ragas import RunConfig
run_cfg = RunConfig(max_workers=1, timeout=60)

# Pattern 1: metrics only (no llm/embeddings in evaluate call)
try:
    print("  Pattern 1: evaluate(dataset, metrics) — no llm/embeddings kwargs...")
    result = evaluate(
        dataset=ds,
        metrics=working_metrics,
        run_config=run_cfg,
        raise_exceptions=False,
    )
    df = result.to_pandas()
    print(f"  ✅ Pattern 1 WORKS!")
    print(f"  Scores: {df.to_dict(orient='records')[0]}")
    WORKING_PATTERN = 1
except Exception as e:
    print(f"  ❌ Pattern 1 failed: {e}")
    WORKING_PATTERN = None

# Pattern 2: pass llm/embeddings to evaluate() too
if WORKING_PATTERN is None:
    try:
        print("  Pattern 2: evaluate(dataset, metrics, llm=, embeddings=)...")
        result = evaluate(
            dataset=ds,
            metrics=working_metrics,
            llm=ragas_llm,
            embeddings=ragas_emb,
            run_config=run_cfg,
            raise_exceptions=False,
        )
        df = result.to_pandas()
        print(f"  ✅ Pattern 2 WORKS!")
        print(f"  Scores: {df.to_dict(orient='records')[0]}")
        WORKING_PATTERN = 2
    except Exception as e:
        print(f"  ❌ Pattern 2 failed: {e}")

print(f"\n{'='*60}")
if WORKING_PATTERN:
    print(f"  ✅ WORKING PATTERN: {WORKING_PATTERN}")
    print(f"  Update cancer_evaluation.py to use this pattern.")
else:
    print(f"  ❌ No working pattern found.")
    print(f"  Check your RAGAS version: pip show ragas")
print("=" * 60)
