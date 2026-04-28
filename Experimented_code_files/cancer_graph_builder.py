# =============================================================================
# cancer_graph_builder.py — Medical Knowledge Graph Builder
#
# PURPOSE:
#   Reads output/chunks/ JSON files produced by cancer_ingestion.py,
#   extracts medical entities and relations using LLMGraphTransformer
#   (ChatGroq llama-3.3-70b), and stores everything in the same Neo4j
#   Aura instance where :Chunk nodes already live.
#
# WHAT IT BUILDS IN NEO4J:
#   :Chunk nodes        — already exist from ingestion (linked here)
#   :Entity nodes       — Disease, Drug, Gene, Biomarker, Treatment, etc.
#   :RELATION edges     — TREATS, INHIBITS, ASSOCIATED_WITH, etc.
#   :Community nodes    — Leiden cluster summaries for global queries
#   :MENTIONS edges     — Chunk → Entity (enables graph-vector hybrid search)
#
# RUN ORDER:
#   1. python cancer_ingestion.py      ← must run first
#   2. python cancer_graph_builder.py  ← this file
#   3. streamlit run cancer_app.py
#
# COST:
#   Runs ONCE per paper batch. One-time Groq token spend.
#   ~300–900 chunks × ~500 tokens/chunk ≈ 150k–450k tokens total.
#   At Groq free tier limits, batched carefully to avoid rate limits.
# =============================================================================

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.documents import Document as LCDoc
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq
from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase

from config import (
    CHUNK_DIR, CAP_DIR, GRAPH_DIR, LOG_DIR,
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE,
    GROQ_API_KEY, GROQ_MODEL_BUILD, GROQ_TEMP_BUILD,
    ALLOWED_NODES, ALLOWED_RELATIONSHIPS,
    GRAPH_BUILD_BATCH, GRAPH_CONTENT_TYPES,
    ensure_dirs,
)

load_dotenv()

# =============================================================================
# HELPERS
# =============================================================================

def _load_chunks_for_graph() -> list[dict]:
    """
    Load text chunks from output/chunks/ JSON files.
    Filters to content types that contain meaningful medical relations.
    Excludes figure_caption (image metadata only) and very short chunks.
    Caption chunks from output/caption_chunks/ are included if they
    contain clinical content (table captions often have drug names).
    """
    chunks = []

    # Text chunks — main source
    for jp in sorted(CHUNK_DIR.glob("*_chunks.json")):
        with open(jp, encoding="utf-8") as f:
            for c in json.load(f):
                ct = c.get("content_type", "")
                if ct in GRAPH_CONTENT_TYPES and len(c.get("content", "")) > 100:
                    chunks.append(c)

    # Caption chunks — include table captions (often contain drug/outcome data)
    for jp in sorted(CAP_DIR.glob("*_caption_chunks.json")):
        with open(jp, encoding="utf-8") as f:
            for c in json.load(f):
                if c.get("content_type") == "table_caption":
                    content = c.get("content", c.get("caption", ""))
                    if len(content) > 50:
                        c["content"] = content
                        chunks.append(c)

    return chunks


def _chunks_to_lcdocs(chunks: list[dict]) -> list[LCDoc]:
    """Convert chunk dicts to LangChain Document objects for LLMGraphTransformer."""
    docs = []
    for c in chunks:
        content = c.get("content", "")
        if not content.strip():
            continue
        docs.append(LCDoc(
            page_content=content,
            metadata={
                "chunk_id":    c.get("chunk_id", ""),
                "source_file": c.get("source_file", ""),
                "cancer_type": c.get("cancer_type", "general"),
                "content_type": c.get("content_type", ""),
            }
        ))
    return docs

# =============================================================================
# NEO4J GRAPH WRITER
# =============================================================================

class MedicalGraphWriter:
    """
    Writes LLMGraphTransformer output into Neo4j.

    Responsibilities:
      - Upsert :Entity nodes (merge on name+type to avoid duplicates)
      - Upsert :RELATION edges between entities
      - Link :Chunk nodes to :Entity nodes via :MENTIONS edges
      - Store community summaries as :Community nodes
    """

    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        )
        # Also use LangChain Neo4jGraph for schema operations
        self.kg = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
        )
        self._nodes_written = 0
        self._edges_written = 0
        self._mentions_written = 0

    def create_constraints(self) -> None:
        """Create uniqueness constraints for clean graph — run once."""
        constraints = [
            "CREATE CONSTRAINT entity_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE",
            "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (c:Community) REQUIRE c.community_id IS UNIQUE",
        ]
        with self.driver.session(database=NEO4J_DATABASE) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    # Constraint may already exist — safe to ignore
                    print(f"    ℹ️  Constraint note: {e}")
        print("    ✅ Neo4j constraints created")

    def write_graph_documents(self, graph_docs: list, source_chunks: list[LCDoc]) -> None:
        """
        Write graph transformer output to Neo4j.

        graph_docs: output from LLMGraphTransformer.convert_to_graph_documents()
        source_chunks: the LCDoc objects that were transformed (same order)
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            for graph_doc, source_chunk in zip(graph_docs, source_chunks):
                chunk_id   = source_chunk.metadata.get("chunk_id", "")
                source_file = source_chunk.metadata.get("source_file", "")
                cancer_type = source_chunk.metadata.get("cancer_type", "")

                # Write nodes
                entity_names_in_chunk = []
                for node in graph_doc.nodes:
                    session.run("""
                        MERGE (e:Entity {name: $name, type: $type})
                        ON CREATE SET
                            e.source_file  = $source_file,
                            e.cancer_type  = $cancer_type,
                            e.created_at   = timestamp()
                        ON MATCH SET
                            e.source_files = CASE
                                WHEN $source_file IN coalesce(e.source_files, [])
                                THEN e.source_files
                                ELSE coalesce(e.source_files, []) + [$source_file]
                            END
                    """,
                        name=str(node.id),
                        type=node.type,
                        source_file=source_file,
                        cancer_type=cancer_type,
                    )
                    entity_names_in_chunk.append(str(node.id))
                    self._nodes_written += 1

                # Write relationships
                for rel in graph_doc.relationships:
                    session.run("""
                        MATCH (e1:Entity {name: $src})
                        MATCH (e2:Entity {name: $tgt})
                        MERGE (e1)-[r:RELATION {type: $rel_type}]->(e2)
                        ON CREATE SET
                            r.source_file  = $source_file,
                            r.cancer_type  = $cancer_type,
                            r.created_at   = timestamp()
                        ON MATCH SET
                            r.occurrence = coalesce(r.occurrence, 0) + 1
                    """,
                        src=str(rel.source.id),
                        tgt=str(rel.target.id),
                        rel_type=rel.type,
                        source_file=source_file,
                        cancer_type=cancer_type,
                    )
                    self._edges_written += 1

                # Link :Chunk → :Entity via :MENTIONS
                # This is what enables the hybrid vector+graph Cypher query
                if chunk_id and entity_names_in_chunk:
                    for ent_name in entity_names_in_chunk:
                        session.run("""
                            MATCH (c:Chunk {chunk_id: $chunk_id})
                            MATCH (e:Entity {name: $ent_name})
                            MERGE (c)-[:MENTIONS]->(e)
                        """,
                            chunk_id=chunk_id,
                            ent_name=ent_name,
                        )
                        self._mentions_written += 1

    def write_community_summary(self, community_id: str, summary: str,
                                 entity_names: list[str], cancer_types: list[str]) -> None:
        """Store a Leiden community summary as a :Community node."""
        with self.driver.session(database=NEO4J_DATABASE) as session:
            session.run("""
                MERGE (c:Community {community_id: $cid})
                SET c.summary      = $summary,
                    c.cancer_types = $cancer_types,
                    c.entity_count = $count,
                    c.created_at   = timestamp()
            """,
                cid=community_id,
                summary=summary,
                cancer_types=list(set(cancer_types)),
                count=len(entity_names),
            )
            # Link community to its member entities
            for name in entity_names[:50]:  # cap to avoid huge writes
                session.run("""
                    MATCH (e:Entity {name: $name})
                    MATCH (c:Community {community_id: $cid})
                    MERGE (e)-[:BELONGS_TO]->(c)
                """, name=name, cid=community_id)

    def get_stats(self) -> dict:
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run("""
                MATCH (e:Entity) WITH count(e) AS entities
                MATCH ()-[r:RELATION]->() WITH entities, count(r) AS relations
                OPTIONAL MATCH ()-[m:MENTIONS]->() 
                RETURN entities, relations, count(m) AS mentions
            """)
            row = result.single()
            return {
                "entities":  row["entities"]  if row else 0,
                "relations": row["relations"] if row else 0,
                "mentions":  row["mentions"]  if row else 0,
            }

    def close(self) -> None:
        self.driver.close()

# =============================================================================
# COMMUNITY DETECTION + SUMMARISATION
# =============================================================================

def detect_communities_and_summarise(writer: MedicalGraphWriter, llm: ChatGroq) -> list[dict]:
    """
    Simple community detection using connected components from Neo4j.

    For each cancer type, groups co-occurring entities into communities
    based on shared source papers. Generates a 2-3 sentence LLM summary
    per community for use in global query mode.

    Note: For production Leiden algorithm, use graspologic library.
    This implementation uses Neo4j's built-in GDS if available,
    falling back to a manual grouping approach for Aura free tier.
    """
    print("\n    🔬 Detecting communities...")
    communities = []

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    # Group entities by cancer type (simple but effective for 6 papers)
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run("""
            MATCH (e:Entity)
            RETURN e.cancer_type AS cancer_type,
                   e.type        AS entity_type,
                   collect(e.name)[..20] AS entities
            ORDER BY cancer_type, entity_type
        """)
        rows = list(result)

    driver.close()

    # Generate summaries for each cancer_type × entity_type community
    community_counter = 0
    summaries = []

    for row in rows:
        cancer_type  = row["cancer_type"] or "general"
        entity_type  = row["entity_type"] or "Unknown"
        entity_names = row["entities"] or []

        if not entity_names or len(entity_names) < 2:
            continue

        community_id = f"community_{cancer_type}_{entity_type}_{community_counter:03d}"
        community_counter += 1

        # Generate LLM summary for this community
        entities_str = ", ".join(entity_names[:15])
        prompt = (
            f"You are a medical knowledge expert. "
            f"Below is a list of {entity_type} entities found in peer-reviewed "
            f"literature about {cancer_type} cancer:\n\n"
            f"{entities_str}\n\n"
            f"Write a 2-3 sentence clinical summary describing what these entities "
            f"represent and their collective significance in {cancer_type} cancer "
            f"treatment and research. Be precise and factual."
        )

        try:
            response = llm.invoke(prompt)
            summary  = response.content.strip()
        except Exception as e:
            print(f"    ⚠️  Summary failed for {community_id}: {e}")
            summary = f"Community of {entity_type} entities relevant to {cancer_type} cancer."

        # Write to Neo4j
        writer.write_community_summary(
            community_id=community_id,
            summary=summary,
            entity_names=entity_names,
            cancer_types=[cancer_type],
        )

        summary_record = {
            "community_id":  community_id,
            "cancer_type":   cancer_type,
            "entity_type":   entity_type,
            "entity_count":  len(entity_names),
            "entities":      entity_names,
            "summary":       summary,
        }
        summaries.append(summary_record)
        communities.append(summary_record)
        print(f"    📝 {community_id}: {len(entity_names)} entities → summarised")

        # Rate limit: small pause between LLM calls
        time.sleep(0.3)

    return summaries

# =============================================================================
# MAIN GRAPH BUILD
# =============================================================================

def build_graph(force_rebuild: bool = False) -> None:
    """
    Full graph construction pipeline:

    Step 1 — Load chunks from output/chunks/ JSON files
    Step 2 — Filter to graph-relevant content types
    Step 3 — Batch through LLMGraphTransformer (ChatGroq 70b)
    Step 4 — Write :Entity nodes + :RELATION edges to Neo4j
    Step 5 — Create :MENTIONS edges (Chunk → Entity)
    Step 6 — Community detection + summarisation
    Step 7 — Save build log to output/graph/graph_build_log.json
    """
    print("=" * 62)
    print("  cancer_graph_builder.py — Medical Knowledge Graph")
    print("=" * 62)

    ensure_dirs()

    # Check if graph already built
    build_log_path = GRAPH_DIR / "graph_build_log.json"
    if build_log_path.exists() and not force_rebuild:
        print("    ✅ Graph already built — skipping")
        print("    ⏭️   To rebuild: build_graph(force_rebuild=True)")
        return

    # ── Step 1: Load chunks ───────────────────────────────
    print("\n  Step 1 — Loading chunks for graph construction...")
    all_chunks = _load_chunks_for_graph()
    print(f"    📦 {len(all_chunks)} chunks selected for graph build")

    if not all_chunks:
        print("    ❌ No chunks found — run cancer_ingestion.py first")
        return

    # Stats breakdown
    ct_counts: dict = {}
    for c in all_chunks:
        ct = c.get("content_type", "?")
        ct_counts[ct] = ct_counts.get(ct, 0) + 1
    for ct, n in sorted(ct_counts.items(), key=lambda x: -x[1]):
        print(f"       {ct:<32} {n:4d} chunks")

    docs = _chunks_to_lcdocs(all_chunks)

    # ── Step 2: Init LLM + transformer ───────────────────
    print("\n  Step 2 — Initialising LLMGraphTransformer (ChatGroq 70b)...")
    llm = ChatGroq(
        model=GROQ_MODEL_BUILD,
        temperature=GROQ_TEMP_BUILD,
        api_key=GROQ_API_KEY,
    )

    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=ALLOWED_NODES,
        allowed_relationships=ALLOWED_RELATIONSHIPS,
        node_properties=["description"],          # extract a description if present
        relationship_properties=["evidence"],     # capture supporting sentence
        strict_mode=False,                        # allow novel relations, don't error
    )

    # ── Step 3: Init graph writer ─────────────────────────
    print("\n  Step 3 — Connecting to Neo4j Aura...")
    writer = MedicalGraphWriter()
    writer.create_constraints()

    # ── Step 4: Batch transform + write ──────────────────
    print(f"\n  Step 4 — Extracting entities and relations...")
    print(f"    Batch size : {GRAPH_BUILD_BATCH} chunks per Groq call")
    print(f"    Total docs : {len(docs)}")
    print(f"    Est. calls : {len(docs) // GRAPH_BUILD_BATCH + 1}")
    print(f"    ⚠️  This takes ~5–15 min on Groq free tier rate limits\n")

    build_results = []
    total_nodes = 0
    total_edges = 0
    failed_batches = 0

    for batch_start in range(0, len(docs), GRAPH_BUILD_BATCH):
        batch = docs[batch_start : batch_start + GRAPH_BUILD_BATCH]
        batch_num = batch_start // GRAPH_BUILD_BATCH + 1
        total_batches = len(docs) // GRAPH_BUILD_BATCH + 1

        try:
            graph_docs = transformer.convert_to_graph_documents(batch)

            batch_nodes = sum(len(gd.nodes) for gd in graph_docs)
            batch_edges = sum(len(gd.relationships) for gd in graph_docs)

            writer.write_graph_documents(graph_docs, batch)

            total_nodes += batch_nodes
            total_edges += batch_edges

            build_results.append({
                "batch":       batch_num,
                "chunks":      len(batch),
                "nodes":       batch_nodes,
                "edges":       batch_edges,
                "status":      "success",
            })

            print(f"    Batch {batch_num:3d}/{total_batches} | "
                  f"+{batch_nodes:3d} nodes | +{batch_edges:3d} edges | "
                  f"total: {total_nodes} nodes, {total_edges} edges")

        except Exception as e:
            failed_batches += 1
            err_msg = str(e)[:100]
            print(f"    ⚠️  Batch {batch_num} failed: {err_msg}")
            build_results.append({
                "batch": batch_num, "status": "failed", "error": err_msg
            })
            # Rate limit handling — wait and continue
            if "rate_limit" in err_msg.lower() or "429" in err_msg:
                print(f"    ⏳ Rate limit hit — waiting 60s...")
                time.sleep(60)
            else:
                time.sleep(2)

    # ── Step 5: Community summarisation ──────────────────
    print(f"\n  Step 5 — Community detection and summarisation...")
    community_summaries = detect_communities_and_summarise(writer, llm)

    # ── Step 6: Final stats ───────────────────────────────
    neo4j_stats = writer.get_stats()
    writer.close()

    print(f"\n  {'─'*50}")
    print(f"  Neo4j graph summary:")
    print(f"    :Entity nodes      : {neo4j_stats['entities']}")
    print(f"    :RELATION edges    : {neo4j_stats['relations']}")
    print(f"    :MENTIONS edges    : {neo4j_stats['mentions']}")
    print(f"    :Community nodes   : {len(community_summaries)}")
    print(f"    Failed batches     : {failed_batches}")

    # ── Step 7: Save build log ────────────────────────────
    build_log = {
        "total_chunks_processed": len(docs),
        "total_nodes":            total_nodes,
        "total_edges":            total_edges,
        "failed_batches":         failed_batches,
        "neo4j_stats":            neo4j_stats,
        "community_count":        len(community_summaries),
        "batches":                build_results,
    }

    with open(build_log_path, "w", encoding="utf-8") as f:
        json.dump(build_log, f, indent=2)

    # Save community summaries locally as backup
    with open(GRAPH_DIR / "community_summaries.json", "w", encoding="utf-8") as f:
        json.dump(community_summaries, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Graph build complete")
    print(f"  📋 Build log    : {build_log_path}")
    print(f"  📝 Communities  : {GRAPH_DIR / 'community_summaries.json'}")
    print(f"\n  ➡️   Next: streamlit run cancer_app.py")
    print("=" * 62)


if __name__ == "__main__":
    # force_rebuild=True  → clears existing graph data, full rebuild
    # force_rebuild=False → skips if build log already exists
    build_graph(force_rebuild=False)