#!/usr/bin/env python3
"""
Test script to diagnose database connection authentication failure.
Issue #102: Database connection authentication failure

This script will:
1. Test DuckDB connection directly
2. Test with configuration settings
3. Identify specific authentication issues
4. Provide diagnostic information
"""

import sys
import os
import logging
from pathlib import Path

# Add the knowledge/database directory to path
sys.path.append(str(Path(__file__).parent / "knowledge" / "database"))
sys.path.append(str(Path(__file__).parent / "knowledge"))

try:
    import duckdb
    from knowledge.database.database_config import DatabaseConfig
    from knowledge.database.connection_manager import DuckDBConnectionManager
    from knowledge.database.database_interface import RIFDatabase
    print("✅ All database modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    try:
        # Alternative import path
        import duckdb
        sys.path.append(str(Path(__file__).parent))
        from knowledge.database.database_config import DatabaseConfig
        from knowledge.database.connection_manager import DuckDBConnectionManager  
        from knowledge.database.database_interface import RIFDatabase
        print("✅ All database modules imported successfully (alternative path)")
    except ImportError as e2:
        print(f"❌ Alternative import also failed: {e2}")
        sys.exit(1)

def test_basic_duckdb_connection():
    """Test basic DuckDB functionality without any configuration."""
    print("\n=== Testing Basic DuckDB Connection ===")
    try:
        # Test memory database first
        conn = duckdb.connect(":memory:")
        result = conn.execute("SELECT 'DuckDB basic test' as message").fetchone()
        print(f"✅ Memory database test: {result[0]}")
        conn.close()
        
        # Test file database
        test_db_path = "/tmp/test_rif.duckdb"
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
            
        conn = duckdb.connect(test_db_path)
        result = conn.execute("SELECT 'DuckDB file test' as message").fetchone()
        print(f"✅ File database test: {result[0]}")
        conn.close()
        
        # Clean up
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
            
        return True
        
    except Exception as e:
        print(f"❌ Basic DuckDB connection failed: {e}")
        return False

def test_configuration():
    """Test database configuration."""
    print("\n=== Testing Database Configuration ===")
    try:
        config = DatabaseConfig()
        print(f"✅ Configuration created: {config}")
        
        # Check if database path directory exists
        db_path = Path(config.database_path)
        print(f"Database path: {db_path.absolute()}")
        print(f"Parent directory exists: {db_path.parent.exists()}")
        print(f"Database file exists: {db_path.exists()}")
        
        # Check schema file exists
        schema_files = config.schema_files
        for schema_file in schema_files:
            schema_path = Path(schema_file)
            print(f"Schema file {schema_file}: exists={schema_path.exists()}")
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_connection_manager():
    """Test the connection manager."""
    print("\n=== Testing Connection Manager ===")
    try:
        config = DatabaseConfig()
        
        # Ensure directories exist
        db_path = Path(config.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        manager = DuckDBConnectionManager(config)
        print(f"✅ Connection manager created")
        
        # Test getting a connection
        with manager.get_connection() as conn:
            result = conn.execute("SELECT 'Connection manager test' as message").fetchone()
            print(f"✅ Connection manager test: {result[0]}")
        
        # Get pool stats
        stats = manager.get_pool_stats()
        print(f"✅ Pool stats: {stats}")
        
        manager.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Connection manager test failed: {e}")
        return False

def test_schema_initialization():
    """Test schema initialization."""
    print("\n=== Testing Schema Initialization ===")
    try:
        config = DatabaseConfig()
        
        # Ensure directories exist
        db_path = Path(config.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create connection and test schema
        conn = duckdb.connect(config.get_connection_string())
        
        # Check if schema file exists and is readable
        schema_file = config.schema_files[0]
        schema_path = Path(schema_file)
        
        if not schema_path.exists():
            print(f"❌ Schema file missing: {schema_path}")
            return False
            
        # Read and execute schema
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
            
        print(f"✅ Schema file readable: {len(schema_sql)} characters")
        
        # Execute schema statements
        statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
        print(f"✅ Found {len(statements)} schema statements")
        
        for i, statement in enumerate(statements):
            try:
                if statement.upper().startswith(('CREATE', 'INSERT', 'SELECT')):
                    conn.execute(statement)
                    print(f"  ✅ Statement {i+1}: {statement[:50]}...")
            except Exception as e:
                print(f"  ⚠️ Statement {i+1} failed: {e}")
        
        # Verify tables exist
        tables_query = """
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'main' 
            AND table_name IN ('entities', 'relationships', 'agent_memory')
        """
        tables = conn.execute(tables_query).fetchall()
        print(f"✅ Tables created: {[t[0] for t in tables]}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Schema initialization failed: {e}")
        return False

def test_rif_database():
    """Test the full RIF database interface."""
    print("\n=== Testing RIF Database Interface ===")
    try:
        config = DatabaseConfig()
        
        # Ensure directories exist
        db_path = Path(config.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create RIF database
        db = RIFDatabase(config)
        print("✅ RIF Database created")
        
        # Test database statistics
        stats = db.get_database_stats()
        print(f"✅ Database stats: {stats}")
        
        # Test verification
        verification = db.verify_setup()
        print(f"✅ Setup verification: {verification}")
        
        # Test basic operations
        try:
            entity_data = {
                'type': 'function',
                'name': 'test_function',
                'file_path': '/test/file.py',
                'line_start': 1,
                'line_end': 10
            }
            entity_id = db.store_entity(entity_data)
            print(f"✅ Entity stored with ID: {entity_id}")
            
            # Retrieve entity
            retrieved = db.get_entity(entity_id)
            if retrieved:
                print(f"✅ Entity retrieved: {retrieved['name']}")
            else:
                print("⚠️ Entity not found after storage")
                
        except Exception as e:
            print(f"⚠️ Entity operations failed: {e}")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ RIF Database test failed: {e}")
        return False

def main():
    """Run all database diagnostic tests."""
    print("🔍 Database Connection Diagnostic Tool")
    print("Issue #102: Database connection authentication failure")
    print("=" * 60)
    
    # Set up logging to see detailed error messages
    logging.basicConfig(level=logging.DEBUG)
    
    tests = [
        ("Basic DuckDB Connection", test_basic_duckdb_connection),
        ("Database Configuration", test_configuration),
        ("Connection Manager", test_connection_manager),
        ("Schema Initialization", test_schema_initialization),
        ("RIF Database Interface", test_rif_database),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("🔍 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed - No authentication issues detected!")
        print("   The database connection should be working properly.")
    else:
        print("🚨 Some tests failed - Authentication/connection issues detected")
        print("   Check the detailed output above for specific error messages.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)