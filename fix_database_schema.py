#!/usr/bin/env python3
"""
Fix database schema constraints for Issue #39 migration
Update the entities table to support all knowledge migration types
"""

import duckdb
import os
import sys

def fix_database_schema():
    """Fix the database schema to support all knowledge types."""
    db_path = 'knowledge/hybrid_knowledge.duckdb'
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return False
    
    try:
        conn = duckdb.connect(db_path)
        
        print("🔧 Fixing database schema constraints...")
        
        # Drop the old constraint and add the new one
        # DuckDB doesn't support ALTER TABLE DROP CONSTRAINT directly
        # So we need to recreate the table with the correct constraint
        
        # First, backup existing data
        print("📦 Backing up existing data...")
        backup_sql = """
        CREATE TABLE entities_backup AS 
        SELECT * FROM entities
        """
        conn.execute(backup_sql)
        
        # Drop the original table
        print("🗑️  Dropping old table...")
        conn.execute("DROP TABLE entities")
        
        # Recreate table with updated constraints
        print("🏗️  Creating updated table...")
        create_sql = """
        CREATE TABLE entities (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            type VARCHAR(50) NOT NULL,
            name VARCHAR(255) NOT NULL,
            file_path VARCHAR(500) NOT NULL,
            line_start INTEGER,
            line_end INTEGER,
            ast_hash VARCHAR(64),
            embedding FLOAT[768],
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- Updated CHECK constraint with all knowledge migration types
            CHECK (type IN (
                'function', 'class', 'module', 'variable', 'constant', 'interface', 'enum',
                'pattern', 'decision', 'learning', 'metric', 'issue_resolution', 'checkpoint', 
                'knowledge_item'
            )),
            CHECK (line_start IS NULL OR line_start >= 1),
            CHECK (line_end IS NULL OR line_end >= line_start),
            CHECK (name != ''),
            CHECK (file_path != '')
        )
        """
        conn.execute(create_sql)
        
        # Restore data
        print("♻️  Restoring data...")
        restore_sql = """
        INSERT INTO entities 
        SELECT * FROM entities_backup
        """
        conn.execute(restore_sql)
        
        # Drop backup table
        conn.execute("DROP TABLE entities_backup")
        
        # Recreate indexes
        print("🔍 Recreating indexes...")
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_entities_type_name ON entities(type, name)",
            "CREATE INDEX IF NOT EXISTS idx_entities_file_path ON entities(file_path)",
            "CREATE INDEX IF NOT EXISTS idx_entities_hash ON entities(ast_hash)",
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)",
            "CREATE INDEX IF NOT EXISTS idx_entities_created_at ON entities(created_at)"
        ]
        
        for index_sql in indexes:
            conn.execute(index_sql)
        
        # Verify the fix
        print("✅ Verifying schema fix...")
        result = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='entities'").fetchall()
        
        if result and 'pattern' in result[0][0] and 'decision' in result[0][0]:
            print("✅ Schema successfully updated!")
            print("📊 New constraint supports all knowledge migration types:")
            print("   - Original: function, class, module, variable, constant, interface, enum")
            print("   - Added: pattern, decision, learning, metric, issue_resolution, checkpoint, knowledge_item")
            success = True
        else:
            print("❌ Schema update verification failed")
            success = False
        
        conn.close()
        return success
        
    except Exception as e:
        print(f"❌ Schema fix failed: {e}")
        return False

if __name__ == '__main__':
    success = fix_database_schema()
    if success:
        print("\n🎉 Database schema fixed successfully!")
        print("Migration can now proceed with all knowledge types.")
    else:
        print("\n💥 Database schema fix failed!")
        print("Manual intervention may be required.")
    
    sys.exit(0 if success else 1)