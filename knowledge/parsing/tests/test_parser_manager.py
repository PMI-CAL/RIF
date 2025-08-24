"""
Tests for the ParserManager core functionality.

These tests verify the foundation components work correctly
before language grammars are implemented.
"""

import unittest
import tempfile
import os
from pathlib import Path

# Import from parent package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knowledge.parsing import (
    ParserManager, 
    LanguageNotSupportedError,
    GrammarNotFoundError,
    get_parser_manager
)


class TestParserManagerFoundation(unittest.TestCase):
    """Test basic ParserManager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.parser_manager = ParserManager.get_instance()
        
        # Create temporary test files
        self.temp_dir = tempfile.mkdtemp()
        
        # JavaScript test file
        self.js_file = os.path.join(self.temp_dir, "test.js")
        with open(self.js_file, 'w') as f:
            f.write("function hello() { console.log('Hello World'); }")
            
        # Python test file  
        self.py_file = os.path.join(self.temp_dir, "test.py")
        with open(self.py_file, 'w') as f:
            f.write("def hello():\n    print('Hello World')")
            
        # Go test file
        self.go_file = os.path.join(self.temp_dir, "test.go")
        with open(self.go_file, 'w') as f:
            f.write('package main\n\nimport "fmt"\n\nfunc main() {\n    fmt.Println("Hello World")\n}')
            
        # Rust test file
        self.rs_file = os.path.join(self.temp_dir, "test.rs")
        with open(self.rs_file, 'w') as f:
            f.write('fn main() {\n    println!("Hello World");\n}')
            
        # Unsupported file
        self.unknown_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.unknown_file, 'w') as f:
            f.write("This is just text")
    
    def tearDown(self):
        """Clean up test environment."""
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir)
        
        # Reset parser manager metrics
        self.parser_manager.reset_metrics()
    
    def test_singleton_pattern(self):
        """Test that ParserManager implements singleton correctly."""
        pm1 = ParserManager.get_instance()
        pm2 = ParserManager()
        pm3 = get_parser_manager()
        
        self.assertIs(pm1, pm2)
        self.assertIs(pm1, pm3)
        self.assertIs(pm2, pm3)
    
    def test_supported_languages(self):
        """Test language support queries."""
        supported = self.parser_manager.get_supported_languages()
        
        self.assertIn('javascript', supported)
        self.assertIn('python', supported)
        self.assertIn('go', supported)
        self.assertIn('rust', supported)
        self.assertEqual(len(supported), 4)
        
        # Test individual language support
        self.assertTrue(self.parser_manager.is_language_supported('javascript'))
        self.assertTrue(self.parser_manager.is_language_supported('python'))
        self.assertTrue(self.parser_manager.is_language_supported('go'))
        self.assertTrue(self.parser_manager.is_language_supported('rust'))
        self.assertFalse(self.parser_manager.is_language_supported('cpp'))
        self.assertFalse(self.parser_manager.is_language_supported('java'))
    
    def test_supported_extensions(self):
        """Test file extension support."""
        extensions = self.parser_manager.get_supported_extensions()
        
        expected_extensions = ['.js', '.jsx', '.mjs', '.cjs', '.py', '.pyx', '.pyi', '.go', '.rs']
        
        for ext in expected_extensions:
            self.assertIn(ext, extensions)
    
    def test_language_detection(self):
        """Test automatic language detection from file extensions."""
        # Test supported languages
        self.assertEqual(self.parser_manager.detect_language("test.js"), "javascript")
        self.assertEqual(self.parser_manager.detect_language("test.jsx"), "javascript")
        self.assertEqual(self.parser_manager.detect_language("test.mjs"), "javascript")
        self.assertEqual(self.parser_manager.detect_language("test.cjs"), "javascript")
        
        self.assertEqual(self.parser_manager.detect_language("test.py"), "python")
        self.assertEqual(self.parser_manager.detect_language("test.pyx"), "python")
        self.assertEqual(self.parser_manager.detect_language("test.pyi"), "python")
        
        self.assertEqual(self.parser_manager.detect_language("test.go"), "go")
        self.assertEqual(self.parser_manager.detect_language("test.rs"), "rust")
        
        # Test unsupported languages
        self.assertIsNone(self.parser_manager.detect_language("test.cpp"))
        self.assertIsNone(self.parser_manager.detect_language("test.java"))
        self.assertIsNone(self.parser_manager.detect_language("test.txt"))
        
        # Test case insensitivity
        self.assertEqual(self.parser_manager.detect_language("TEST.JS"), "javascript")
        self.assertEqual(self.parser_manager.detect_language("Test.PY"), "python")
    
    def test_parse_file_real_results(self):
        """Test parse_file returns real parsing results in Phase 2."""
        # Test JavaScript file
        result = self.parser_manager.parse_file(self.js_file)
        
        self.assertEqual(result['file_path'], self.js_file)
        self.assertEqual(result['language'], 'javascript')
        self.assertIsNotNone(result['tree'])  # Real result
        self.assertIsNotNone(result['root_node'])  # Real AST
        self.assertNotIn('mock', result)  # No longer mock
        self.assertGreater(result['source_size'], 0)
        self.assertIsInstance(result['parse_time'], float)
        self.assertIsInstance(result['timestamp'], int)
        
        # Test Python file
        result = self.parser_manager.parse_file(self.py_file)
        
        self.assertEqual(result['language'], 'python')
        self.assertIsNotNone(result['tree'])
        self.assertNotIn('mock', result)
        
        # Test explicit language specification
        result = self.parser_manager.parse_file(self.go_file, language='go')
        self.assertEqual(result['language'], 'go')
        self.assertIsNotNone(result['tree'])
        self.assertNotIn('mock', result)
    
    def test_parse_file_unsupported_language(self):
        """Test parsing unsupported file extensions."""
        with self.assertRaises(LanguageNotSupportedError) as context:
            self.parser_manager.parse_file(self.unknown_file)
        
        error = context.exception
        self.assertIn("Unknown language for file", str(error))
        self.assertEqual(error.file_path, self.unknown_file)
        
        # Test explicit unsupported language
        with self.assertRaises(LanguageNotSupportedError) as context:
            self.parser_manager.parse_file(self.js_file, language='cpp')
        
        error = context.exception
        self.assertEqual(error.language, 'cpp')
        self.assertIn('javascript', error.supported_languages)
    
    def test_metrics_collection(self):
        """Test performance metrics collection."""
        # Get initial metrics
        initial_metrics = self.parser_manager.get_metrics()
        
        # Parse some files
        self.parser_manager.parse_file(self.js_file)
        self.parser_manager.parse_file(self.py_file)
        self.parser_manager.parse_file(self.go_file)
        
        # Get updated metrics
        updated_metrics = self.parser_manager.get_metrics()
        
        # Check metrics structure
        self.assertIn('supported_languages', updated_metrics)
        self.assertIn('parse_counts', updated_metrics)
        self.assertIn('average_parse_times', updated_metrics)
        self.assertIn('memory_usage_mb', updated_metrics)
        self.assertIn('total_memory_mb', updated_metrics)
        self.assertIn('memory_limit_mb', updated_metrics)
        
        # Check parse counts increased
        self.assertEqual(updated_metrics['parse_counts']['javascript'], 1)
        self.assertEqual(updated_metrics['parse_counts']['python'], 1)
        self.assertEqual(updated_metrics['parse_counts']['go'], 1)
        
        # Check average parse times exist
        self.assertIn('javascript', updated_metrics['average_parse_times'])
        self.assertIn('python', updated_metrics['average_parse_times'])
        self.assertIn('go', updated_metrics['average_parse_times'])
        
        # Verify supported languages
        self.assertEqual(len(updated_metrics['supported_languages']), 4)
    
    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        # Parse some files to generate metrics
        self.parser_manager.parse_file(self.js_file)
        self.parser_manager.parse_file(self.py_file)
        
        # Verify metrics exist
        metrics_before = self.parser_manager.get_metrics()
        self.assertGreater(metrics_before['parse_counts']['javascript'], 0)
        self.assertGreater(metrics_before['parse_counts']['python'], 0)
        
        # Reset metrics
        self.parser_manager.reset_metrics()
        
        # Verify metrics cleared
        metrics_after = self.parser_manager.get_metrics()
        self.assertEqual(metrics_after['parse_counts'].get('javascript', 0), 0)
        self.assertEqual(metrics_after['parse_counts'].get('python', 0), 0)
        self.assertEqual(len(metrics_after['average_parse_times']), 0)
    
    def test_file_validation(self):
        """Test file size and access validation."""
        # Test with non-existent file
        with self.assertRaises(Exception) as context:
            self.parser_manager.parse_file("/nonexistent/file.js")
        
        # Should be ParsingError about file access
        self.assertIn("Cannot access file", str(context.exception))
    
    def test_large_file_handling(self):
        """Test handling of large files."""
        # Create a large temp file (larger than 10MB limit)
        large_file = os.path.join(self.temp_dir, "large.js")
        
        # Create content that exceeds the limit
        large_content = "// Large file\n" + "console.log('line');\n" * 500000  # ~10MB+
        
        with open(large_file, 'w') as f:
            f.write(large_content)
        
        # Should fail due to size limit
        with self.assertRaises(Exception) as context:
            self.parser_manager.parse_file(large_file)
        
        self.assertIn("exceeds maximum", str(context.exception))


if __name__ == '__main__':
    unittest.main()