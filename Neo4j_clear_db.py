# =============================================================================
# neo4j_clear_db.py — Clear Neo4j Aura before fresh pipeline run
#
# THREE OPTIONS depending on what you want to wipe:
#
#   Option 1 — Clear only MedChat nodes (safe, targeted)
#              Removes: :Chunk, :Entity, :Community, :RELATION, :MENTIONS
#              Keeps:   Any other experiments you ran (Roman Empire, healthcare)
#
#   Option 2 — Wipe EVERYTHING (full clean slate)
#              MATCH (n) DETACH DELETE n
#              Use this if you want zero contamination from old experiments
#
#   Option 3 — Drop vector index only (keeps nodes, resets index)
#              Use this if ingestion failed midway and index is corrupted
#
# RUN BEFORE:
#   python cancer_ingestion.py
#
# USAGE:
#   python neo4j_clear_db.py --option 1    ← targeted MedChat wipe
#   python neo4j_clear_db.py --option 2    ← full wipe
#   python neo4j_clear_db.py --option 3    ← index only
#   python neo4j_clear_db.py              ← interactive menu
# =============================================================================

import sys
import argparse
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

import os
NEO4J_URI      = os.getenv("NEO4J_URI",      "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME",  "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD",  "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE",  "neo4j")

# Vector index name — must match config.py
CHUNK_INDEX_NAME = "chunk_vector_index"


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def show_current_state(driver):
    """Print a count of everything currently in the database."""
    print("\n📊 Current database state:")
    with driver.session(database=NEO4J_DATABASE) as session:

        # Node counts by label
        result = session.run("""
            CALL db.labels() YIELD label
            CALL apoc.cypher.run(
                'MATCH (n:' + label + ') RETURN count(n) as count', {}
            ) YIELD value
            RETURN label, value.count AS count
            ORDER BY count DESC
        """)
        rows = list(result)

        if not rows:
            # Fallback without APOC
            result2 = session.run("MATCH (n) RETURN labels(n) AS lbl, count(*) AS cnt")
            label_counts: dict = {}
            for r in result2:
                for l in r["lbl"]:
                    label_counts[l] = label_counts.get(l, 0) + r["cnt"]
            if label_counts:
                for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
                    print(f"    :{lbl:<25} {cnt:>6} nodes")
            else:
                print("    Database is already empty.")
        else:
            for row in rows:
                print(f"    :{row['label']:<25} {row['count']:>6} nodes")

        # Relationship counts
        rel_result = session.run("MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt ORDER BY cnt DESC")
        rel_rows = list(rel_result)
        if rel_rows:
            print()
            for r in rel_rows:
                print(f"    :{r['rel']:<25} {r['cnt']:>6} relationships")

        # Vector indexes
        idx_result = session.run("SHOW INDEXES WHERE type = 'VECTOR'")
        idx_rows = list(idx_result)
        if idx_rows:
            print()
            for idx in idx_rows:
                print(f"    Vector index: {idx.get('name', '?')} on :{idx.get('labelsOrTypes', ['?'])[0]}")
        else:
            print("\n    No vector indexes found.")

    print()


def option1_clear_medchat_nodes(driver):
    """
    TARGETED: Remove only MedChat-specific node types.
    Safe if you have other experiments in the same DB.
    """
    print("\n🎯 Option 1 — Targeted MedChat wipe")
    print("   Removing: :Chunk, :Entity, :Community nodes and all their relationships")
    print("   Keeping:  Everything else in the database\n")

    with driver.session(database=NEO4J_DATABASE) as session:

        # Step 1: Remove :MENTIONS relationships
        result = session.run("MATCH ()-[r:MENTIONS]->() DELETE r RETURN count(r) AS deleted")
        print(f"   ✅ :MENTIONS relationships deleted : {result.single()['deleted']}")

        # Step 2: Remove :BELONGS_TO relationships (entity → community)
        result = session.run("MATCH ()-[r:BELONGS_TO]->() DELETE r RETURN count(r) AS deleted")
        print(f"   ✅ :BELONGS_TO relationships deleted: {result.single()['deleted']}")

        # Step 3: Remove :RELATION relationships (entity → entity)
        result = session.run("MATCH ()-[r:RELATION]->() DELETE r RETURN count(r) AS deleted")
        print(f"   ✅ :RELATION relationships deleted  : {result.single()['deleted']}")

        # Step 4: Delete :Community nodes
        result = session.run("MATCH (n:Community) DELETE n RETURN count(n) AS deleted")
        print(f"   ✅ :Community nodes deleted         : {result.single()['deleted']}")

        # Step 5: Delete :Entity nodes
        result = session.run("MATCH (n:Entity) DELETE n RETURN count(n) AS deleted")
        print(f"   ✅ :Entity nodes deleted            : {result.single()['deleted']}")

        # Step 6: Delete :Chunk nodes and their relationships
        # Do in batches — Aura free tier has memory limits
        total_chunks = 0
        while True:
            result = session.run("""
                MATCH (n:Chunk)
                WITH n LIMIT 500
                DETACH DELETE n
                RETURN count(n) AS deleted
            """)
            deleted = result.single()["deleted"]
            total_chunks += deleted
            if deleted == 0:
                break
        print(f"   ✅ :Chunk nodes deleted             : {total_chunks}")

        # Step 7: Drop vector index
        try:
            session.run(f"DROP INDEX {CHUNK_INDEX_NAME} IF EXISTS")
            print(f"   ✅ Vector index '{CHUNK_INDEX_NAME}' dropped")
        except Exception as e:
            print(f"   ℹ️  Index drop note: {e}")

        # Step 8: Drop constraints created by graph builder
        constraints_to_drop = ["entity_unique", "community_id"]
        for constraint in constraints_to_drop:
            try:
                session.run(f"DROP CONSTRAINT {constraint} IF EXISTS")
                print(f"   ✅ Constraint '{constraint}' dropped")
            except Exception as e:
                print(f"   ℹ️  Constraint '{constraint}' note: {e}")

    print("\n   ✅ MedChat nodes cleared. Database ready for fresh ingestion.")


def option2_full_wipe(driver):
    """
    NUCLEAR: Wipe every single node and relationship.
    Use when you want a completely clean database.
    """
    print("\n💣 Option 2 — Full database wipe")
    print("   This removes EVERYTHING including Roman Empire, healthcare, etc.\n")

    with driver.session(database=NEO4J_DATABASE) as session:

        # Drop all vector indexes first
        try:
            indexes = session.run("SHOW INDEXES WHERE type = 'VECTOR'")
            for idx in indexes:
                idx_name = idx.get("name", "")
                if idx_name:
                    session.run(f"DROP INDEX {idx_name} IF EXISTS")
                    print(f"   ✅ Dropped vector index: {idx_name}")
        except Exception as e:
            print(f"   ℹ️  Index drop note: {e}")

        # Drop all constraints
        try:
            constraints = session.run("SHOW CONSTRAINTS")
            for c in constraints:
                c_name = c.get("name", "")
                if c_name:
                    session.run(f"DROP CONSTRAINT {c_name} IF EXISTS")
                    print(f"   ✅ Dropped constraint: {c_name}")
        except Exception as e:
            print(f"   ℹ️  Constraint drop note: {e}")

        # Delete all nodes in batches (Aura free tier memory limit)
        print("\n   Deleting all nodes in batches of 1000...")
        total_deleted = 0
        while True:
            result = session.run("""
                MATCH (n)
                WITH n LIMIT 1000
                DETACH DELETE n
                RETURN count(n) AS deleted
            """)
            deleted = result.single()["deleted"]
            total_deleted += deleted
            if deleted == 0:
                break
            print(f"   ↳ {total_deleted} nodes deleted so far...")

    print(f"\n   ✅ Full wipe complete. {total_deleted} nodes removed.")
    print("   Database is a clean slate.")


def option3_index_only(driver):
    """
    INDEX ONLY: Drop the vector index but keep all nodes.
    Use when ingestion failed midway and you need to rebuild the index.
    """
    print(f"\n🔧 Option 3 — Drop vector index '{CHUNK_INDEX_NAME}' only")
    print("   Nodes and relationships are preserved.\n")

    with driver.session(database=NEO4J_DATABASE) as session:
        try:
            session.run(f"DROP INDEX {CHUNK_INDEX_NAME} IF EXISTS")
            print(f"   ✅ Vector index '{CHUNK_INDEX_NAME}' dropped")
            print("   Run cancer_ingestion.py to rebuild it.")
        except Exception as e:
            print(f"   ❌ Could not drop index: {e}")


def interactive_menu(driver):
    """Show current state and prompt for choice."""
    show_current_state(driver)

    print("=" * 60)
    print("  Neo4j Clear Options")
    print("=" * 60)
    print("  1 — Clear MedChat nodes only (safe, targeted)")
    print("      Removes: :Chunk, :Entity, :Community + all edges")
    print("      Keeps  : Other experiment data (Roman Empire, etc.)")
    print()
    print("  2 — Wipe EVERYTHING (full clean slate)")
    print("      Removes: Every node and relationship in the database")
    print("      Use    : When you want zero contamination from all past runs")
    print()
    print("  3 — Drop vector index only")
    print("      Removes: chunk_vector_index")
    print("      Keeps  : All nodes — use when index is corrupted")
    print()
    print("  0 — Exit without changes")
    print("=" * 60)

    choice = input("\nEnter choice (0/1/2/3): ").strip()

    if choice == "0":
        print("   No changes made.")
    elif choice == "1":
        confirm = input("   Confirm targeted MedChat wipe? (yes/no): ").strip().lower()
        if confirm == "yes":
            option1_clear_medchat_nodes(driver)
        else:
            print("   Cancelled.")
    elif choice == "2":
        confirm = input("   Confirm FULL WIPE of all data? Type 'WIPE' to confirm: ").strip()
        if confirm == "WIPE":
            option2_full_wipe(driver)
        else:
            print("   Cancelled.")
    elif choice == "3":
        option3_index_only(driver)
    else:
        print("   Invalid choice.")


def main():
    parser = argparse.ArgumentParser(description="Clear Neo4j Aura database before pipeline run")
    parser.add_argument("--option", type=int, choices=[1, 2, 3],
                        help="1=targeted, 2=full wipe, 3=index only")
    parser.add_argument("--confirm", action="store_true",
                        help="Skip confirmation prompt (use in scripts)")
    args = parser.parse_args()

    print("=" * 60)
    print("  neo4j_clear_db.py — Database Cleanup Utility")
    print(f"  URI: {NEO4J_URI[:40]}...")
    print("=" * 60)

    driver = get_driver()

    try:
        if args.option is None:
            interactive_menu(driver)
        else:
            show_current_state(driver)
            if args.option == 1:
                if not args.confirm:
                    c = input("Confirm targeted MedChat wipe? (yes/no): ").strip().lower()
                    if c != "yes":
                        print("Cancelled."); return
                option1_clear_medchat_nodes(driver)
            elif args.option == 2:
                if not args.confirm:
                    c = input("Type 'WIPE' to confirm full wipe: ").strip()
                    if c != "WIPE":
                        print("Cancelled."); return
                option2_full_wipe(driver)
            elif args.option == 3:
                option3_index_only(driver)

        # Show final state
        print("\n📊 Database state after cleanup:")
        show_current_state(driver)

    finally:
        driver.close()


if __name__ == "__main__":
    main()