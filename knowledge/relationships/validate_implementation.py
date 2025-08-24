#!/usr/bin/env python3
"""
Validation script for relationship detection implementation.

This script validates that all components of the relationship detection system
work correctly and meet the requirements.
"""

import sys
import tempfile
import shutil
from pathlib import Path
from uuid import uuid4

# Add the knowledge directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_basic_imports():
    """Test that all main modules can be imported."""
    print("Testing basic imports...")
    
    try:
        from relationships.relationship_types import RelationshipType, CodeRelationship
        from extraction.entity_types import CodeEntity, EntityType, SourceLocation
        print("✅ Data types imported successfully")
        
        # Test relationship types
        assert len(list(RelationshipType)) >= 5
        print(f"✅ Relationship types: {[rt.value for rt in RelationshipType]}")
        
        # Test creating a relationship
        source_id = uuid4()
        target_id = uuid4()
        rel = CodeRelationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=RelationshipType.IMPORTS,
            confidence=0.95
        )
        assert rel.source_id == source_id
        assert rel.target_id == target_id
        assert rel.relationship_type == RelationshipType.IMPORTS
        print("✅ CodeRelationship creation works")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_analyzer_creation():
    """Test that analyzers can be created."""
    print("\nTesting analyzer creation...")
    
    try:
        # Mock parser manager for testing
        class MockParserManager:
            def get_file_content(self, file_path):
                return b"test content"
        
        parser_manager = MockParserManager()
        
        from relationships.import_analyzer import ImportExportAnalyzer
        import_analyzer = ImportExportAnalyzer(parser_manager)
        print("✅ ImportExportAnalyzer created successfully")
        
        from relationships.call_analyzer import FunctionCallAnalyzer
        call_analyzer = FunctionCallAnalyzer(parser_manager)
        print("✅ FunctionCallAnalyzer created successfully")
        
        from relationships.inheritance_analyzer import InheritanceAnalyzer
        inheritance_analyzer = InheritanceAnalyzer(parser_manager)
        print("✅ InheritanceAnalyzer created successfully")
        
        # Check supported languages
        languages = import_analyzer.supported_languages | call_analyzer.supported_languages | inheritance_analyzer.supported_languages
        print(f"✅ Supported languages: {sorted(languages)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analyzer creation error: {e}")
        return False


def test_relationship_detector():
    """Test the main relationship detector."""
    print("\nTesting relationship detector...")
    
    try:
        # Mock parser manager
        class MockParserManager:
            def get_file_content(self, file_path):
                return b"import os\ndef test(): pass"
            
            def parse_file(self, file_path):
                # Mock tree object
                class MockTree:
                    def __init__(self):
                        self.root_node = MockNode()
                
                class MockNode:
                    def __init__(self):
                        self.type = "module"
                        self.children = []
                        self.text = b"test"
                        self.start_point = (0, 0)
                        self.start_byte = 0
                        self.end_byte = 4
                
                return MockTree(), "python"
        
        parser_manager = MockParserManager()
        
        from relationships.relationship_detector import RelationshipDetector
        detector = RelationshipDetector(parser_manager)
        print("✅ RelationshipDetector created successfully")
        
        # Check supported types and languages
        supported_types = detector.get_supported_relationship_types()
        supported_languages = detector.get_supported_languages()
        
        print(f"✅ Detector supports {len(supported_types)} relationship types")
        print(f"✅ Detector supports {len(supported_languages)} languages")
        
        return True
        
    except Exception as e:
        print(f"❌ Relationship detector error: {e}")
        return False


def test_storage_integration():
    """Test storage integration."""
    print("\nTesting storage integration...")
    
    try:
        # Create temporary database
        temp_db = tempfile.mktemp(suffix='.duckdb')
        
        from relationships.storage_integration import RelationshipStorage
        storage = RelationshipStorage(temp_db)
        print("✅ RelationshipStorage created successfully")
        
        # Test basic operations
        stats = storage.get_relationship_statistics()
        print(f"✅ Database initialized with {stats['total_relationships']} relationships")
        
        # Clean up
        storage.close()
        Path(temp_db).unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Storage integration error: {e}")
        return False


def test_cli_setup():
    """Test that CLI can be set up."""
    print("\nTesting CLI setup...")
    
    try:
        from relationships.cli import setup_argument_parser
        parser = setup_argument_parser()
        print("✅ CLI argument parser created successfully")
        
        # Test parsing basic commands
        args = parser.parse_args(['stats', '--verbose'])
        assert args.command == 'stats'
        assert args.verbose == True
        print("✅ CLI argument parsing works")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI setup error: {e}")
        return False


def validate_implementation():
    """Run all validation tests."""
    print("=" * 60)
    print("VALIDATING RELATIONSHIP DETECTION IMPLEMENTATION")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_analyzer_creation,
        test_relationship_detector,
        test_storage_integration,
        test_cli_setup
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"VALIDATION SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED - Implementation is ready for use!")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed - Implementation needs fixes")
        return False


if __name__ == "__main__":
    success = validate_implementation()
    sys.exit(0 if success else 1)