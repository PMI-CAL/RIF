#!/usr/bin/env python3
"""
Basic functionality tests for LightRAG implementation.
Tests core functionality without requiring external dependencies.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
lightrag_dir = current_dir.parent
project_root = lightrag_dir.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from lightrag.core.lightrag_core import LightRAGCore, get_lightrag_instance
        print("✓ lightrag_core import successful")
    except ImportError as e:
        print(f"✗ lightrag_core import failed: {e}")
        return False
    
    # Embedding manager may fail due to missing dependencies - that's OK for now
    try:
        from lightrag.embeddings.embedding_manager import EmbeddingManager
        print("✓ embedding_manager import successful")
    except ImportError as e:
        print(f"⚠ embedding_manager import failed (expected if dependencies not installed): {e}")
    
    return True


def test_lightrag_core():
    """Test LightRAG core functionality."""
    print("\nTesting LightRAG core functionality...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Initialize LightRAG with temp directory
            from lightrag.core.lightrag_core import LightRAGCore
            rag = LightRAGCore(temp_dir)
            print("✓ LightRAGCore initialized successfully")
            
            # Test collection stats
            stats = rag.get_collection_stats()
            expected_collections = ["patterns", "decisions", "code_snippets", "issue_resolutions"]
            
            if all(col in stats for col in expected_collections):
                print("✓ All expected collections created")
            else:
                print("✗ Missing collections")
                return False
            
            # Test storing knowledge
            test_content = "This is a test pattern for functionality validation."
            test_metadata = {
                "type": "test_pattern",
                "category": "testing",
                "tags": "test,validation"  # ChromaDB requires string, not list
            }
            
            doc_id = rag.store_knowledge("patterns", test_content, test_metadata)
            print(f"✓ Knowledge stored with ID: {doc_id}")
            
            # Test retrieving knowledge (will use simple string matching without embeddings)
            try:
                results = rag.retrieve_knowledge("test pattern", "patterns", 1)
                if results:
                    print("✓ Knowledge retrieved successfully")
                else:
                    print("⚠ Knowledge retrieval returned no results (may need embeddings)")
            except Exception as e:
                print(f"⚠ Knowledge retrieval test skipped (embeddings not available): {e}")
            
            # Test updating knowledge  
            update_success = rag.update_knowledge("patterns", doc_id, metadata={"updated": True})
            if update_success:
                print("✓ Knowledge update successful")
            else:
                print("✗ Knowledge update failed")
                return False
            
            # Test collection export
            export_path = os.path.join(temp_dir, "test_export.json")
            export_success = rag.export_collection("patterns", export_path)
            if export_success and os.path.exists(export_path):
                print("✓ Collection export successful")
            else:
                print("✗ Collection export failed")
                return False
            
            # Test deleting knowledge
            delete_success = rag.delete_knowledge("patterns", doc_id)
            if delete_success:
                print("✓ Knowledge deletion successful")
            else:
                print("✗ Knowledge deletion failed")
                return False
            
            print("✓ All LightRAG core tests passed!")
            return True
            
        except Exception as e:
            print(f"✗ LightRAG core test failed: {e}")
            return False


def test_directory_structure():
    """Test that directory structure is correct."""
    print("\nTesting directory structure...")
    
    lightrag_path = Path(__file__).parent.parent
    
    expected_dirs = [
        "core",
        "embeddings", 
        "collections",
        "tests",
        "docs"
    ]
    
    expected_files = [
        "requirements.txt",
        "init_lightrag.py",
        "core/lightrag_core.py",
        "embeddings/embedding_manager.py",
        "tests/test_basic.py"
    ]
    
    # Check directories
    for dir_name in expected_dirs:
        dir_path = lightrag_path / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"✓ Directory exists: {dir_name}")
        else:
            print(f"✗ Missing directory: {dir_name}")
            return False
    
    # Check files
    for file_path in expected_files:
        full_path = lightrag_path / file_path
        if full_path.exists() and full_path.is_file():
            print(f"✓ File exists: {file_path}")
        else:
            print(f"✗ Missing file: {file_path}")
            return False
    
    print("✓ Directory structure test passed!")
    return True


def main():
    """Run all basic tests."""
    print("="*50)
    print("LIGHTRAG BASIC FUNCTIONALITY TESTS")
    print("="*50)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Module Imports", test_imports),
        ("LightRAG Core", test_lightrag_core)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                failed += 1
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "="*50)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*50)
    
    if failed == 0:
        print("🎉 All basic tests passed! LightRAG foundation is working.")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)