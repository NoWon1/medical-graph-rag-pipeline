# Experimented Code Files

This folder is an archive of experimental work.

It is not the active source of truth for the MedChat pipeline.

## Rule for this folder

- Do not use these files as the default implementation.
- Do not import from this folder in new production changes unless there is a deliberate reason.
- Prefer the root files in `Medchat_Graph_RAG/` for all active development.

## What this folder is for

Use this folder only when you want to:

- review an older idea
- recover a specific logic change
- compare an experiment against the current production file

## Safe way to use this folder

If you want to reuse something from here, use it carefully:

1. Identify the exact function or block worth keeping.
2. Compare it with the current root version.
3. Copy only the needed logic into the active file in the repository root.
4. Re-test the active pipeline after merging.

## What not to do

- Do not point new scripts at this folder by default.
- Do not assume code here is newer or better than the root version.
- Do not treat this directory as production-ready.

## Active code lives here instead

Use the root directory files for normal work:

- [../config.py](/d:/Desktop/Neo_4J/Medchat_Graph_RAG/config.py)
- [../cancer_ingestion.py](/d:/Desktop/Neo_4J/Medchat_Graph_RAG/cancer_ingestion.py)
- [../cancer_retrieval.py](/d:/Desktop/Neo_4J/Medchat_Graph_RAG/cancer_retrieval.py)
- [../cancer_app.py](/d:/Desktop/Neo_4J/Medchat_Graph_RAG/cancer_app.py)
- [../cancer_evaluation.py](/d:/Desktop/Neo_4J/Medchat_Graph_RAG/cancer_evaluation.py)

In short: this folder is for reference, not for default use.
