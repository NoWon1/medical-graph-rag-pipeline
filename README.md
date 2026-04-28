# MedChat Graph RAG

MedChat Graph RAG is a cancer-focused assistant that combines:

- Neo4j vector retrieval over paper chunks
- a structured treatment and nutrition knowledge graph
- image-aware retrieval using `[IMAGE: ...]` tags
- a Streamlit chat UI
- optional RAGAS-based evaluation

This repository's active, production-facing code lives in the root of `Medchat_Graph_RAG`.

## Active files

- [config.py](/d:/Desktop/Neo_4J/Medchat_Graph_RAG/config.py)  
  Single source of truth for paths, API keys, Neo4j settings, models, query modes, and graph constants.

- [cancer_ingestion.py](/d:/Desktop/Neo_4J/Medchat_Graph_RAG/cancer_ingestion.py)  
  Processes PDFs, extracts text and images, creates chunks, and writes embeddings to Neo4j.

- [cancer_retrieval.py](/d:/Desktop/Neo_4J/Medchat_Graph_RAG/cancer_retrieval.py)  
  Main retrieval and answer-generation pipeline for research, graph, and auto modes.

- [cancer_app.py](/d:/Desktop/Neo_4J/Medchat_Graph_RAG/cancer_app.py)  
  Streamlit UI for asking questions and uploading patient reports.

- [Neo4j_clear_db.py](/d:/Desktop/Neo_4J/Medchat_Graph_RAG/Neo4j_clear_db.py)  
  Utility to clear the Neo4j database or drop the vector index before a fresh run.

- [cancer_evaluation.py](/d:/Desktop/Neo_4J/Medchat_Graph_RAG/cancer_evaluation.py)  
  Main evaluation script for your pipeline.

- [Ragas_test.py](/d:/Desktop/Neo_4J/Medchat_Graph_RAG/Ragas_test.py)  
  Compatibility tester for local `ragas` setup before running full evaluation.

## Repository workflow

### 1. Install dependencies

```powershell
py -3 -m pip install -r requirements.txt
```

### 2. Configure environment

Create or update `.env` with the keys used by [config.py](/d:/Desktop/Neo_4J/Medchat_Graph_RAG/config.py):

```env
NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=
NEO4J_DATABASE=neo4j

GROQ_API_KEY=
GOOGLE_API_KEY=
OPENAI_API_KEY=

PIPELINE_LLM_PROVIDER=groq
PIPELINE_LLM_MODEL=llama-3.1-8b-instant
JUDGE_LLM_PROVIDER=google
JUDGE_LLM_MODEL=gemini-2.5-pro
```

### 3. Add source PDFs

Place your medical review PDFs in:

```text
data/
```

### 4. Clear Neo4j if needed

Use this before a fresh ingestion if old data may contaminate the run:

```powershell
py -3 Neo4j_clear_db.py --option 1
```

Options:

- `--option 1` clears MedChat data only
- `--option 2` wipes the full database
- `--option 3` drops only the vector index

### 5. Run ingestion

```powershell
py -3 cancer_ingestion.py
```

This step:

- extracts and cleans PDF text
- extracts caption-linked visual assets
- injects `[IMAGE: filename.png]` tags
- creates chunk JSON files under `output/`
- builds the Neo4j vector index

### 6. Launch the app

```powershell
streamlit run cancer_app.py
```

Available modes in the UI:

- `Auto`: combines graph and literature retrieval
- `Research & Literature`: focuses on clinical papers
- `Treatment & Nutrition`: focuses on graph-based food, side-effect, and interaction guidance

### 7. Run evaluation

First verify `ragas` compatibility:

```powershell
py -3 Ragas_test.py
```

Then run the main evaluation:

```powershell
py -3 cancer_evaluation.py
```

Useful variants:

```powershell
py -3 cancer_evaluation.py --quick
py -3 cancer_evaluation.py --category graph
```

## Outputs

The pipeline writes artifacts under `output/`, including:

- `output/markdown/`
- `output/chunks/`
- `output/caption_chunks/`
- `output/images/`
- `output/embedding_export/`
- `output/logs/`
- `output/evaluation/`

## Important repository rule

Do not treat `Experimented_code_files/` as the active codebase.

That folder contains experimental and archival files only. Production work, fixes, and future changes should target the root files in `Medchat_Graph_RAG` unless you intentionally decide to recover a specific idea from the archive.

See:

- [Experimented_code_files/README.md](/d:/Desktop/Neo_4J/Medchat_Graph_RAG/Experimented_code_files/README.md)

## Safe development rule for future work

When using code from `Experimented_code_files/`, do it selectively:

- compare logic before copying anything
- do not import the archived files directly into the active pipeline by default
- port only the exact function or idea you want
- re-test after every manual merge

This keeps the production path stable and avoids accidental regression from older experiments.
