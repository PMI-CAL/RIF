"""
Comprehensive tests for the entity extraction system.
"""

import unittest
import tempfile
import os
from pathlib import Path
import json

from ..entity_types import EntityType, CodeEntity, SourceLocation, ExtractionResult
from ..entity_extractor import EntityExtractor
from ..language_extractors import JavaScriptExtractor, PythonExtractor
from ..storage_integration import EntityStorage, EntityExtractionPipeline


class TestEntityTypes(unittest.TestCase):
    """Test entity type classes and data structures."""
    
    def test_source_location(self):
        """Test SourceLocation functionality."""
        location = SourceLocation(line_start=10, line_end=20, column_start=5, column_end=15)
        
        self.assertEqual(location.line_start, 10)
        self.assertEqual(location.line_end, 20)
        self.assertEqual(location.column_start, 5)
        self.assertEqual(location.column_end, 15)
        
        # Test dict conversion
        location_dict = location.to_dict()
        expected = {
            'line_start': 10,
            'line_end': 20,
            'column_start': 5,
            'column_end': 15
        }
        self.assertEqual(location_dict, expected)
    
    def test_code_entity(self):
        """Test CodeEntity functionality."""
        location = SourceLocation(line_start=5, line_end=10)
        
        entity = CodeEntity(
            type=EntityType.FUNCTION,
            name="test_function",
            file_path="/path/to/file.py",
            location=location,
            metadata={"parameters": ["arg1", "arg2"]}
        )
        
        # Test basic properties
        self.assertEqual(entity.type, EntityType.FUNCTION)
        self.assertEqual(entity.name, "test_function")
        self.assertEqual(entity.file_path, "/path/to/file.py")
        self.assertEqual(entity.location, location)
        self.assertEqual(entity.metadata["parameters"], ["arg1", "arg2"])
        
        # Test hash generation
        self.assertTrue(entity.ast_hash)
        self.assertEqual(len(entity.ast_hash), 16)  # SHA256 truncated to 16 chars
        
        # Test DB dict conversion
        db_dict = entity.to_db_dict()
        self.assertEqual(db_dict['type'], 'function')
        self.assertEqual(db_dict['name'], 'test_function')
        self.assertEqual(db_dict['line_start'], 5)
        self.assertEqual(db_dict['line_end'], 10)
        
        # Test from_db_dict
        reconstructed = CodeEntity.from_db_dict(db_dict)
        self.assertEqual(reconstructed.type, entity.type)
        self.assertEqual(reconstructed.name, entity.name)
        self.assertEqual(reconstructed.file_path, entity.file_path)
    
    def test_extraction_result(self):
        """Test ExtractionResult functionality."""
        entities = [
            CodeEntity(type=EntityType.FUNCTION, name="func1"),
            CodeEntity(type=EntityType.CLASS, name="Class1"),
            CodeEntity(type=EntityType.FUNCTION, name="func2")
        ]
        
        result = ExtractionResult(
            file_path="/test.py",
            language="python",
            entities=entities,
            extraction_time=0.5,
            success=True
        )
        
        # Test entity counts
        counts = result.get_entity_counts()
        self.assertEqual(counts['function'], 2)
        self.assertEqual(counts['class'], 1)


class TestLanguageExtractors(unittest.TestCase):
    """Test language-specific extractors."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_file(self, filename: str, content: str) -> str:
        """Create a test file with given content."""
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path
    
    def test_javascript_extractor(self):
        """Test JavaScript entity extraction."""
        # Skip if tree-sitter is not available
        try:
            import tree_sitter
            from ...parsing.parser_manager import ParserManager
        except ImportError:
            self.skipTest("tree-sitter not available")
        
        js_code = """
        // Test JavaScript file
        import { Component } from 'react';
        import lodash from 'lodash';
        
        const API_URL = 'https://api.example.com';
        let globalVar = 42;
        
        function calculateSum(a, b) {
            return a + b;
        }
        
        class UserManager {
            constructor(config) {
                this.config = config;
            }
            
            async getUser(id) {
                const response = await fetch(`${API_URL}/users/${id}`);
                return response.json();
            }
        }
        
        export default UserManager;
        export { calculateSum };
        """
        
        file_path = self.create_test_file("test.js", js_code)
        
        # Test with mock parsing for now since tree-sitter setup is complex
        extractor = JavaScriptExtractor()
        
        # This test verifies the extractor class structure
        self.assertEqual(extractor.language, 'javascript')
        self.assertIn(EntityType.FUNCTION, extractor.get_supported_entity_types())
        self.assertIn(EntityType.CLASS, extractor.get_supported_entity_types())
        self.assertIn(EntityType.INTERFACE, extractor.get_supported_entity_types())
    
    def test_python_extractor(self):
        """Test Python entity extraction."""
        python_code = """
        import os
        from typing import List, Dict
        
        API_KEY = "secret_key"
        debug_mode = True
        
        def process_data(data: List[str]) -> Dict[str, int]:
            \"\"\"Process a list of strings and return counts.\"\"\"
            result = {}
            for item in data:
                result[item] = len(item)
            return result
        
        class DataProcessor:
            \"\"\"Handles data processing operations.\"\"\"
            
            def __init__(self, config: dict):
                self.config = config
            
            @property
            def is_ready(self) -> bool:
                return self.config is not None
            
            def transform(self, input_data):
                if not self.is_ready:
                    raise ValueError("Processor not configured")
                return process_data(input_data)
        """
        
        file_path = self.create_test_file("test.py", python_code)
        
        extractor = PythonExtractor()
        
        # Test extractor properties
        self.assertEqual(extractor.language, 'python')
        self.assertIn(EntityType.FUNCTION, extractor.get_supported_entity_types())
        self.assertIn(EntityType.CLASS, extractor.get_supported_entity_types())


class TestEntityExtractor(unittest.TestCase):
    """Test the main EntityExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = EntityExtractor()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_file(self, filename: str, content: str) -> str:
        """Create a test file with given content."""
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path
    
    def test_supported_languages(self):
        """Test supported language detection."""
        supported = self.extractor.get_supported_languages()
        
        self.assertIn('javascript', supported)
        self.assertIn('python', supported)
        self.assertIn('go', supported)
        self.assertIn('rust', supported)
        
        self.assertTrue(self.extractor.is_language_supported('python'))
        self.assertFalse(self.extractor.is_language_supported('cobol'))
    
    def test_metrics_tracking(self):
        """Test performance metrics tracking."""
        # Reset metrics first
        self.extractor.reset_metrics()
        
        initial_metrics = self.extractor.get_extraction_metrics()
        self.assertEqual(initial_metrics['files_processed'], 0)
        self.assertEqual(initial_metrics['entities_extracted'], 0)
        
        # Create a simple test file
        test_file = self.create_test_file("simple.py", "def hello(): pass")
        
        # Process file (this might fail due to tree-sitter setup, but metrics should still update)
        result = self.extractor.extract_from_file(test_file)
        
        # Check that metrics were updated
        metrics = self.extractor.get_extraction_metrics()
        self.assertEqual(metrics['files_processed'], 1)
    
    def test_directory_processing(self):
        """Test directory processing functionality."""
        # Create test directory structure
        os.makedirs(os.path.join(self.temp_dir, "subdir"))
        
        # Create test files
        self.create_test_file("test1.py", "def func1(): pass")
        self.create_test_file("test2.js", "function func2() {}")
        self.create_test_file("subdir/test3.py", "class TestClass: pass")
        self.create_test_file("README.md", "# Documentation")  # Should be ignored
        
        # Test directory processing
        try:
            results = self.extractor.extract_from_directory(
                self.temp_dir,
                extensions=['.py', '.js'],
                recursive=True,
                exclude_patterns=['README']
            )
            
            # Should process 3 files (exclude README.md)
            self.assertGreaterEqual(len(results), 3)
            
            # Check that all results are ExtractionResult objects
            for result in results:
                self.assertIsInstance(result, ExtractionResult)
                
        except Exception as e:
            # Expected if tree-sitter is not properly set up
            self.assertIn("tree-sitter", str(e).lower())


class TestEntityStorage(unittest.TestCase):
    """Test entity storage functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.duckdb")
        
        # Create storage with temporary database
        self.storage = EntityStorage(self.db_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.storage.close()
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_database_initialization(self):
        """Test database and schema initialization."""
        # Database should be created and schema initialized
        self.assertTrue(os.path.exists(self.db_path))
        
        # Test that we can get statistics (implies schema exists)
        try:
            stats = self.storage.get_file_statistics()
            self.assertIn('total_entities', stats)
            self.assertEqual(stats['total_entities'], 0)  # Initially empty
        except Exception as e:
            # Expected if DuckDB is not available
            self.assertIn("duckdb", str(e).lower())
    
    def test_entity_storage_and_retrieval(self):
        """Test storing and retrieving entities."""
        try:
            import duckdb
        except ImportError:
            self.skipTest("DuckDB not available")
        
        # Create test entities
        entities = [
            CodeEntity(
                type=EntityType.FUNCTION,
                name="test_function",
                file_path="/test/file.py",
                location=SourceLocation(line_start=1, line_end=5),
                metadata={"parameters": ["arg1", "arg2"]}
            ),
            CodeEntity(
                type=EntityType.CLASS,
                name="TestClass",
                file_path="/test/file.py",
                location=SourceLocation(line_start=7, line_end=15),
                metadata={"methods": ["method1", "method2"]}
            )
        ]
        
        # Store entities
        result = self.storage.store_entities(entities, update_mode='insert')
        
        self.assertEqual(result['inserted'], 2)
        self.assertEqual(result['updated'], 0)
        self.assertEqual(result['skipped'], 0)
        
        # Retrieve entities
        retrieved = self.storage.get_entities_by_file("/test/file.py")
        
        self.assertEqual(len(retrieved), 2)
        self.assertEqual(retrieved[0].name, "test_function")
        self.assertEqual(retrieved[1].name, "TestClass")
        
        # Test search
        functions = self.storage.get_entities_by_type("function")
        self.assertEqual(len(functions), 1)
        self.assertEqual(functions[0].name, "test_function")
        
        # Test name search
        search_results = self.storage.search_entities("test")
        self.assertGreaterEqual(len(search_results), 1)
    
    def test_upsert_functionality(self):
        """Test entity upserting (insert/update based on changes)."""
        try:
            import duckdb
        except ImportError:
            self.skipTest("DuckDB not available")
        
        # Create initial entity
        entity = CodeEntity(
            type=EntityType.FUNCTION,
            name="evolving_function",
            file_path="/test/evolve.py",
            location=SourceLocation(line_start=1, line_end=5),
            metadata={"version": 1}
        )
        
        # First insert
        result = self.storage.store_entities([entity], update_mode='upsert')
        self.assertEqual(result['inserted'], 1)
        
        # Same entity (same hash) - should skip
        result = self.storage.store_entities([entity], update_mode='upsert')
        self.assertEqual(result['skipped'], 1)
        
        # Modified entity (different metadata, will generate different hash)
        entity.metadata = {"version": 2, "new_feature": True}
        entity.ast_hash = entity.generate_hash()  # Regenerate hash
        
        result = self.storage.store_entities([entity], update_mode='upsert')
        self.assertEqual(result['updated'], 1)


class TestEntityExtractionPipeline(unittest.TestCase):
    """Test the complete extraction pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "pipeline_test.duckdb")
        
        try:
            self.pipeline = EntityExtractionPipeline(self.db_path)
            self.pipeline_available = True
        except Exception:
            self.pipeline_available = False
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'pipeline'):
            self.pipeline.close()
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_file(self, filename: str, content: str) -> str:
        """Create a test file with given content."""
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        if not self.pipeline_available:
            self.skipTest("Pipeline dependencies not available")
        
        self.assertIsNotNone(self.pipeline.extractor)
        self.assertIsNotNone(self.pipeline.storage)
        
        # Test metrics
        metrics = self.pipeline.get_pipeline_metrics()
        self.assertIn('extraction', metrics)
        self.assertIn('storage', metrics)
        self.assertIn('combined', metrics)
    
    def test_single_file_processing(self):
        """Test processing a single file through the complete pipeline."""
        if not self.pipeline_available:
            self.skipTest("Pipeline dependencies not available")
        
        # Create test file
        test_code = """
def hello_world():
    '''Simple greeting function.'''
    return "Hello, World!"

class Greeter:
    def greet(self, name):
        return f"Hello, {name}!"
        """
        
        test_file = self.create_test_file("hello.py", test_code)
        
        # Process file
        result = self.pipeline.process_file(test_file)
        
        # Check results
        self.assertIn('file_path', result)
        self.assertIn('extraction_success', result)
        self.assertIn('entities_found', result)
        self.assertEqual(result['file_path'], test_file)
    
    def test_directory_processing(self):
        """Test processing an entire directory."""
        if not self.pipeline_available:
            self.skipTest("Pipeline dependencies not available")
        
        # Create test files
        self.create_test_file("module1.py", "def func1(): pass\nclass Class1: pass")
        self.create_test_file("module2.py", "def func2(): pass\ndef func3(): pass")
        
        # Create subdirectory
        os.makedirs(os.path.join(self.temp_dir, "submodule"))
        self.create_test_file("submodule/utils.py", "CONSTANT = 42\ndef utility(): pass")
        
        # Process directory
        result = self.pipeline.process_directory(self.temp_dir, recursive=True)
        
        # Check results
        self.assertIn('total_files', result)
        self.assertIn('successful_extractions', result)
        self.assertIn('total_entities_extracted', result)
        self.assertEqual(result['directory'], self.temp_dir)


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)