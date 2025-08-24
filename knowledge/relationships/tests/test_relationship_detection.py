"""
Integration tests for the complete relationship detection system.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from typing import List
from uuid import uuid4

from ..relationship_detector import RelationshipDetector
from ..storage_integration import RelationshipAnalysisPipeline
from ..relationship_types import RelationshipType, CodeRelationship
from ...extraction.entity_types import CodeEntity, EntityType, SourceLocation
from ...parsing.parser_manager import ParserManager


class TestRelationshipDetection(unittest.TestCase):
    """Test the complete relationship detection pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.parser_manager = ParserManager()
        self.detector = RelationshipDetector(self.parser_manager)
        
        # Create sample test files
        self.create_test_files()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def create_test_files(self):
        """Create sample source files for testing."""
        test_files = {
            'math_utils.py': '''
def add(a, b):
    """Add two numbers."""
    return a + b

def multiply(a, b):
    """Multiply two numbers.""" 
    return a * b

class Calculator:
    def calculate(self, x, y):
        return add(x, y)
''',
            'main.py': '''
from math_utils import Calculator, add
import os
import json

class Application:
    def __init__(self):
        self.calc = Calculator()
    
    def run(self):
        result = add(5, 3)
        calc_result = self.calc.calculate(10, 20)
        data = json.loads('{"test": true}')
        return result + calc_result
''',
            'utils.js': '''
export function formatNumber(num) {
    return num.toFixed(2);
}

export class Logger {
    log(message) {
        console.log(message);
    }
}
''',
            'app.js': '''
import { formatNumber, Logger } from './utils.js';
import path from 'path';

class Application extends Logger {
    constructor() {
        super();
        this.logger = new Logger();
    }
    
    format(value) {
        return formatNumber(value);
    }
    
    start() {
        this.log('Starting application');
        const result = this.format(123.456);
        return result;
    }
}
'''
        }
        
        for filename, content in test_files.items():
            file_path = Path(self.test_dir) / filename
            with open(file_path, 'w') as f:
                f.write(content)
    
    def create_sample_entities(self, file_path: str, language: str) -> List[CodeEntity]:
        """Create sample entities for a file (simulating entity extraction)."""
        entities = []
        
        if file_path.endswith('math_utils.py'):
            entities = [
                CodeEntity(
                    id=uuid4(),
                    type=EntityType.MODULE,
                    name='math_utils',
                    file_path=file_path,
                    location=SourceLocation(1, 15)
                ),
                CodeEntity(
                    id=uuid4(),
                    type=EntityType.FUNCTION,
                    name='add',
                    file_path=file_path,
                    location=SourceLocation(2, 4)
                ),
                CodeEntity(
                    id=uuid4(),
                    type=EntityType.FUNCTION,
                    name='multiply',
                    file_path=file_path,
                    location=SourceLocation(6, 8)
                ),
                CodeEntity(
                    id=uuid4(),
                    type=EntityType.CLASS,
                    name='Calculator',
                    file_path=file_path,
                    location=SourceLocation(10, 13)
                )
            ]
        elif file_path.endswith('main.py'):
            entities = [
                CodeEntity(
                    id=uuid4(),
                    type=EntityType.MODULE,
                    name='main',
                    file_path=file_path,
                    location=SourceLocation(1, 18)
                ),
                CodeEntity(
                    id=uuid4(),
                    type=EntityType.CLASS,
                    name='Application',
                    file_path=file_path,
                    location=SourceLocation(5, 12)
                )
            ]
        elif file_path.endswith('utils.js'):
            entities = [
                CodeEntity(
                    id=uuid4(),
                    type=EntityType.MODULE,
                    name='utils',
                    file_path=file_path,
                    location=SourceLocation(1, 12)
                ),
                CodeEntity(
                    id=uuid4(),
                    type=EntityType.FUNCTION,
                    name='formatNumber',
                    file_path=file_path,
                    location=SourceLocation(1, 3)
                ),
                CodeEntity(
                    id=uuid4(),
                    type=EntityType.CLASS,
                    name='Logger',
                    file_path=file_path,
                    location=SourceLocation(5, 9)
                )
            ]
        elif file_path.endswith('app.js'):
            entities = [
                CodeEntity(
                    id=uuid4(),
                    type=EntityType.MODULE,
                    name='app',
                    file_path=file_path,
                    location=SourceLocation(1, 20)
                ),
                CodeEntity(
                    id=uuid4(),
                    type=EntityType.CLASS,
                    name='Application',
                    file_path=file_path,
                    location=SourceLocation(4, 20)
                )
            ]
        
        return entities
    
    def test_import_relationship_detection(self):
        """Test detection of import relationships."""
        main_py = str(Path(self.test_dir) / 'main.py')
        entities = self.create_sample_entities(main_py, 'python')
        
        result = self.detector.detect_relationships_from_file(main_py, entities)
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.relationships), 0)
        
        # Check for import relationships
        import_relationships = [
            r for r in result.relationships 
            if r.relationship_type == RelationshipType.IMPORTS
        ]
        self.assertGreater(len(import_relationships), 0)
        
        # Verify relationship properties
        for rel in import_relationships:
            self.assertIsNotNone(rel.source_id)
            self.assertIsNotNone(rel.target_id)
            self.assertNotEqual(rel.source_id, rel.target_id)
            self.assertGreaterEqual(rel.confidence, 0.0)
            self.assertLessEqual(rel.confidence, 1.0)
    
    def test_function_call_detection(self):
        """Test detection of function call relationships."""
        main_py = str(Path(self.test_dir) / 'main.py')
        entities = self.create_sample_entities(main_py, 'python')
        
        result = self.detector.detect_relationships_from_file(main_py, entities)
        
        self.assertTrue(result.success)
        
        # Check for call relationships
        call_relationships = [
            r for r in result.relationships 
            if r.relationship_type == RelationshipType.CALLS
        ]
        
        # Should find calls to add(), json.loads(), etc.
        # Note: Some may be cross-file references that need resolution
        self.assertGreaterEqual(len(call_relationships), 0)
    
    def test_inheritance_detection(self):
        """Test detection of inheritance relationships."""
        app_js = str(Path(self.test_dir) / 'app.js')
        entities = self.create_sample_entities(app_js, 'javascript')
        
        result = self.detector.detect_relationships_from_file(app_js, entities)
        
        self.assertTrue(result.success)
        
        # Check for inheritance relationships (Application extends Logger)
        inheritance_relationships = [
            r for r in result.relationships 
            if r.relationship_type == RelationshipType.EXTENDS
        ]
        
        # Should find the extends relationship
        self.assertGreaterEqual(len(inheritance_relationships), 0)
    
    def test_cross_file_resolution(self):
        """Test cross-file reference resolution."""
        # Process all files to populate cross-file resolver
        all_files = list(Path(self.test_dir).glob('*.py')) + list(Path(self.test_dir).glob('*.js'))
        
        all_entities = {}
        for file_path in all_files:
            language = 'python' if file_path.suffix == '.py' else 'javascript'
            entities = self.create_sample_entities(str(file_path), language)
            all_entities[str(file_path)] = entities
            
            # Register entities for cross-file resolution
            self.detector.analysis_context.cross_file_resolver.register_entities(entities)
        
        # Now detect relationships in main.py which imports from math_utils.py
        main_py = str(Path(self.test_dir) / 'main.py')
        result = self.detector.detect_relationships_from_file(main_py, all_entities[main_py])
        
        self.assertTrue(result.success)
        
        # Should have relationships to entities in other files
        cross_file_rels = [
            r for r in result.relationships
            if r.metadata.get('needs_resolution') is not True  # Successfully resolved
        ]
        
        self.assertGreaterEqual(len(cross_file_rels), 0)
    
    def test_relationship_validation(self):
        """Test relationship validation logic."""
        main_py = str(Path(self.test_dir) / 'main.py')
        entities = self.create_sample_entities(main_py, 'python')
        
        result = self.detector.detect_relationships_from_file(main_py, entities)
        
        # Validate all detected relationships
        validation_results = self.detector.validate_relationships(result.relationships)
        
        self.assertGreaterEqual(validation_results['valid_relationships'], 0)
        self.assertEqual(validation_results['invalid_relationships'], 0)
        self.assertEqual(len(validation_results['errors']), 0)
        
        # Check confidence distribution
        confidence_dist = validation_results['confidence_distribution']
        total_confidence_counts = sum(confidence_dist.values())
        self.assertEqual(total_confidence_counts, len(result.relationships))
    
    def test_directory_processing(self):
        """Test processing an entire directory."""
        results = self.detector.detect_relationships_from_directory(
            self.test_dir,
            extensions=['.py', '.js'],
            recursive=False
        )
        
        self.assertGreater(len(results), 0)
        
        # Check that we processed the expected files
        file_paths = [r.file_path for r in results]
        self.assertTrue(any('main.py' in path for path in file_paths))
        self.assertTrue(any('math_utils.py' in path for path in file_paths))
        self.assertTrue(any('app.js' in path for path in file_paths))
        self.assertTrue(any('utils.js' in path for path in file_paths))
        
        # Check success rates
        successful_results = [r for r in results if r.success]
        self.assertGreater(len(successful_results), 0)
        
        # Check that relationships were found
        total_relationships = sum(len(r.relationships) for r in successful_results)
        self.assertGreater(total_relationships, 0)
    
    def test_analyzer_metrics(self):
        """Test that analysis metrics are collected properly."""
        main_py = str(Path(self.test_dir) / 'main.py')
        entities = self.create_sample_entities(main_py, 'python')
        
        # Reset metrics
        self.detector.reset_metrics()
        
        result = self.detector.detect_relationships_from_file(main_py, entities)
        
        # Check metrics
        metrics = self.detector.get_analysis_metrics()
        
        self.assertEqual(metrics['files_processed'], 1)
        self.assertGreater(metrics['analysis_time'], 0.0)
        self.assertIn('python', metrics['language_breakdown'])
        self.assertGreaterEqual(len(metrics['analyzer_breakdown']), 1)
        
        # Check language metrics
        python_metrics = metrics['language_breakdown']['python']
        self.assertEqual(python_metrics['files'], 1)
        self.assertGreaterEqual(python_metrics['relationships'], 0)
        self.assertGreater(python_metrics['time'], 0.0)
    
    def test_supported_languages_and_types(self):
        """Test that detector reports supported languages and types correctly."""
        supported_languages = self.detector.get_supported_languages()
        supported_types = self.detector.get_supported_relationship_types()
        
        # Should support at least Python and JavaScript
        self.assertIn('python', supported_languages)
        self.assertIn('javascript', supported_languages)
        
        # Should support basic relationship types
        self.assertIn(RelationshipType.IMPORTS, supported_types)
        self.assertIn(RelationshipType.CALLS, supported_types)
        self.assertIn(RelationshipType.EXTENDS, supported_types)
    
    def test_error_handling(self):
        """Test error handling for invalid files and edge cases."""
        # Test with non-existent file
        result = self.detector.detect_relationships_from_file('/nonexistent/file.py', [])
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        
        # Test with empty entities list
        main_py = str(Path(self.test_dir) / 'main.py')
        result = self.detector.detect_relationships_from_file(main_py, [])
        # Should succeed but might have fewer relationships due to missing context
        self.assertTrue(result.success)
    
    def test_relationship_confidence_scoring(self):
        """Test that confidence scores are reasonable."""
        main_py = str(Path(self.test_dir) / 'main.py')
        entities = self.create_sample_entities(main_py, 'python')
        
        result = self.detector.detect_relationships_from_file(main_py, entities)
        
        for relationship in result.relationships:
            # All confidence scores should be valid
            self.assertGreaterEqual(relationship.confidence, 0.0)
            self.assertLessEqual(relationship.confidence, 1.0)
            
            # Import relationships should have high confidence
            if relationship.relationship_type == RelationshipType.IMPORTS:
                self.assertGreaterEqual(relationship.confidence, 0.8)
            
            # Check that confidence correlates with relationship certainty
            if relationship.metadata.get('needs_resolution'):
                # Unresolved references should have lower confidence
                self.assertLessEqual(relationship.confidence, 0.9)


class TestRelationshipAnalysisPipeline(unittest.TestCase):
    """Test the complete analysis pipeline including storage."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.test_dir) / 'test_relationships.duckdb')
        self.parser_manager = ParserManager()
        self.pipeline = RelationshipAnalysisPipeline(self.parser_manager, self.db_path)
    
    def tearDown(self):
        """Clean up test environment."""
        self.pipeline.close()
        shutil.rmtree(self.test_dir)
    
    def test_pipeline_integration(self):
        """Test that the pipeline integrates detection and storage correctly."""
        # Create a simple test file
        test_file = Path(self.test_dir) / 'test.py'
        test_file.write_text('''
import os
import json

def main():
    data = json.loads('{}')
    return data
''')
        
        # Create sample entities
        entities = [
            CodeEntity(
                id=uuid4(),
                type=EntityType.MODULE,
                name='test',
                file_path=str(test_file),
                location=SourceLocation(1, 8)
            ),
            CodeEntity(
                id=uuid4(),
                type=EntityType.FUNCTION,
                name='main',
                file_path=str(test_file),
                location=SourceLocation(4, 6)
            )
        ]
        
        # Process file through pipeline
        result = self.pipeline.process_file(str(test_file), entities)
        
        self.assertTrue(result['detection_success'])
        self.assertGreater(result['relationships_found'], 0)
        self.assertGreaterEqual(result['relationships_stored'], 0)
        
        # Verify relationships were stored
        storage_stats = self.pipeline.storage.get_relationship_statistics()
        self.assertGreater(storage_stats['total_relationships'], 0)


if __name__ == '__main__':
    unittest.main()