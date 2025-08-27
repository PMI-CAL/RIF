#!/usr/bin/env python3
"""
Setup consolidated knowledge database for RIF
This creates a single source of truth for all knowledge data
"""

import duckdb
import json
import sys
from pathlib import Path
from datetime import datetime

def create_database():
    """Create the consolidated knowledge database with all necessary tables"""
    
    # Use a single consolidated database file
    db_path = "/Users/cal/DEV/RIF/knowledge/knowledge.duckdb"
    
    print(f"Creating consolidated knowledge database at: {db_path}")
    
    # Connect to database (creates if doesn't exist)
    conn = duckdb.connect(db_path)
    
    # Create entities table (for code entities and knowledge items)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            type VARCHAR NOT NULL,  -- pattern, decision, issue, capability, limitation, tool, etc.
            name VARCHAR NOT NULL,
            description TEXT,
            content TEXT,
            file_path VARCHAR,
            line_start INTEGER,
            line_end INTEGER,
            ast_hash VARCHAR(64),
            embedding FLOAT[],
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create relationships table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            source_id UUID NOT NULL,
            target_id UUID NOT NULL,
            relationship_type VARCHAR NOT NULL,
            metadata JSON,
            confidence FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_id) REFERENCES entities(id),
            FOREIGN KEY (target_id) REFERENCES entities(id)
        )
    """)
    
    # Create knowledge_items table (for patterns, decisions, issues)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_items (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            item_type VARCHAR NOT NULL,  -- pattern, decision, issue_resolution
            item_id VARCHAR UNIQUE,  -- pattern_id, decision_id, issue_id
            title VARCHAR NOT NULL,
            category VARCHAR,
            description TEXT,
            content JSON,  -- Full JSON content
            tags JSON,  -- Store tags as JSON array
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create agent_memory table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_memory (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_name VARCHAR NOT NULL,
            memory_type VARCHAR NOT NULL,
            content TEXT,
            context JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create orchestration tables (migrate from orchestration.duckdb)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orchestration_decisions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            issue_id INTEGER,
            decision_type VARCHAR,
            decision_details JSON,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orchestration_state (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            issue_id INTEGER,
            state VARCHAR,
            metadata JSON,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    print("✅ Database schema created successfully")
    
    # Verify tables
    tables = conn.execute("SHOW TABLES").fetchall()
    print(f"Created tables: {[t[0] for t in tables]}")
    
    conn.close()
    return db_path

def migrate_orchestration_data():
    """Migrate data from orchestration.duckdb to consolidated database"""
    
    orch_db = "/Users/cal/DEV/RIF/knowledge/orchestration.duckdb"
    main_db = "/Users/cal/DEV/RIF/knowledge/knowledge.duckdb"
    
    if not Path(orch_db).exists():
        print("⚠️ orchestration.duckdb not found, skipping migration")
        return
    
    print("Migrating orchestration data...")
    
    # Connect to both databases
    orch_conn = duckdb.connect(orch_db, read_only=True)
    main_conn = duckdb.connect(main_db)
    
    try:
        # Copy orchestration_decisions
        decisions = orch_conn.execute("SELECT * FROM orchestration_decisions").fetchall()
        if decisions:
            for row in decisions:
                main_conn.execute("""
                    INSERT INTO orchestration_decisions (issue_id, decision_type, decision_details, timestamp)
                    VALUES (?, ?, ?, ?)
                """, row)
            print(f"  Migrated {len(decisions)} orchestration decisions")
        
        # Copy orchestration_state
        states = orch_conn.execute("SELECT * FROM orchestration_state").fetchall()
        if states:
            for row in states:
                main_conn.execute("""
                    INSERT INTO orchestration_state (issue_id, state, metadata, timestamp)
                    VALUES (?, ?, ?, ?)
                """, row)
            print(f"  Migrated {len(states)} orchestration states")
    
    except Exception as e:
        print(f"  Warning during migration: {e}")
    
    orch_conn.close()
    main_conn.close()

def verify_database():
    """Verify the database is properly set up"""
    
    db_path = "/Users/cal/DEV/RIF/knowledge/knowledge.duckdb"
    conn = duckdb.connect(db_path, read_only=True)
    
    print("\n📊 Database Statistics:")
    
    # Check each table
    tables = ['entities', 'relationships', 'knowledge_items', 'agent_memory', 
              'orchestration_decisions', 'orchestration_state']
    
    for table in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {count} rows")
        except:
            print(f"  {table}: ❌ Not found")
    
    conn.close()

if __name__ == "__main__":
    # Create the consolidated database
    db_path = create_database()
    
    # Migrate existing data
    migrate_orchestration_data()
    
    # Verify setup
    verify_database()
    
    print(f"\n✅ Knowledge database ready at: {db_path}")
    print("Next step: Run knowledge ingestion to populate with patterns and issues")