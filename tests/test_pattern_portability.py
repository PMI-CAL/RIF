"""
Test suite for Pattern Export/Import System - Issue #80

Tests cover:
- Pattern export functionality
- Pattern import with various merge strategies
- Version compatibility checking
- Conflict resolution
- Data validation
- File I/O operations
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude.commands.pattern_portability import (
    PatternPortability, MergeStrategy, ConflictResolution,
    ConflictInfo, ImportResult
)
from knowledge.pattern_application.core import Pattern, TechStack


class TestPatternPortability:
    """Test class for PatternPortability functionality."""
    
    @pytest.fixture
    def temp_patterns_dir(self):
        """Create a temporary directory for test patterns."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_pattern(self):
        """Create a sample pattern for testing."""
        return Pattern(
            pattern_id="test-pattern-001",
            name="Test Pattern",
            description="A test pattern for unit testing",
            complexity="medium",
            tech_stack=TechStack(
                primary_language="python",
                frameworks=["django", "flask"],
                tools=["pytest"]
            ),
            domain="testing",
            tags=["test", "sample"],
            confidence=0.8,
            success_rate=0.75,
            usage_count=5,
            implementation_steps=[
                {"step": 1, "description": "Setup test environment"},
                {"step": 2, "description": "Write test cases"}
            ],
            code_examples=[
                {
                    "language": "python",
                    "description": "Sample test",
                    "code": "def test_sample(): assert True"
                }
            ],
            validation_criteria=["Tests pass", "Code coverage > 80%"]
        )
    
    @pytest.fixture
    def pattern_portability(self, temp_patterns_dir):
        """Create PatternPortability instance with temp directory."""
        return PatternPortability(
            patterns_dir=temp_patterns_dir,
            project_id="test-project"
        )
    
    def create_sample_pattern_file(self, patterns_dir: str, pattern_id: str = "sample-001"):
        """Create a sample pattern JSON file in the given directory."""
        pattern_data = {
            "pattern_id": pattern_id,
            "pattern_name": "Sample Pattern",
            "description": "A sample pattern for testing",
            "complexity": "medium",
            "domain": "general",
            "tags": ["sample", "test"],
            "confidence": 0.7,
            "success_rate": 0.6,
            "usage_count": 3,
            "tech_stack": {
                "primary_language": "python",
                "frameworks": ["pytest"],
                "tools": ["git"]
            },
            "implementation_steps": [
                {"step": 1, "description": "Initialize project"}
            ],
            "code_examples": [
                {"language": "python", "code": "print('hello')"}
            ],
            "validation_criteria": ["Code compiles"]
        }
        
        filepath = Path(patterns_dir) / f"{pattern_id}.json"
        with open(filepath, 'w') as f:
            json.dump(pattern_data, f, indent=2)
        
        return str(filepath)
    
    def test_initialization(self, temp_patterns_dir):
        """Test PatternPortability initialization."""
        portability = PatternPortability(
            patterns_dir=temp_patterns_dir,
            project_id="test-project"
        )
        
        assert portability.project_id == "test-project"
        assert portability.patterns_dir == Path(temp_patterns_dir)
        assert portability.patterns_dir.exists()
    
    def test_get_all_patterns_empty(self, pattern_portability):
        """Test getting all patterns from empty directory."""
        patterns = pattern_portability.get_all_patterns()
        assert patterns == []
    
    def test_get_all_patterns_with_files(self, pattern_portability, temp_patterns_dir):
        """Test loading patterns from JSON files."""
        # Create test pattern files
        self.create_sample_pattern_file(temp_patterns_dir, "pattern-001")
        self.create_sample_pattern_file(temp_patterns_dir, "pattern-002")
        
        patterns = pattern_portability.get_all_patterns()
        assert len(patterns) == 2
        
        pattern_ids = [p.pattern_id for p in patterns]
        assert "pattern-001" in pattern_ids
        assert "pattern-002" in pattern_ids
    
    def test_get_pattern_by_id(self, pattern_portability, temp_patterns_dir):
        """Test retrieving a specific pattern by ID."""
        self.create_sample_pattern_file(temp_patterns_dir, "specific-pattern")
        
        pattern = pattern_portability.get_pattern("specific-pattern")
        assert pattern is not None
        assert pattern.pattern_id == "specific-pattern"
        assert pattern.name == "Sample Pattern"
        
        # Test non-existent pattern
        non_existent = pattern_portability.get_pattern("does-not-exist")
        assert non_existent is None
    
    def test_serialize_pattern(self, pattern_portability, sample_pattern):
        """Test pattern serialization."""
        serialized = pattern_portability.serialize_pattern(sample_pattern)
        
        assert isinstance(serialized, dict)
        assert serialized['pattern_id'] == "test-pattern-001"
        assert serialized['name'] == "Test Pattern"
        assert serialized['complexity'] == "medium"
        assert 'export_metadata' in serialized
        assert serialized['export_metadata']['export_version'] == "1.0.0"
        assert serialized['export_metadata']['source_project'] == "test-project"
    
    def test_deserialize_pattern(self, pattern_portability):
        """Test pattern deserialization."""
        pattern_data = {
            "pattern_id": "deserialized-pattern",
            "name": "Deserialized Pattern",
            "description": "Pattern created from data",
            "complexity": "high",
            "domain": "serialization",
            "tags": ["deserialize", "test"],
            "confidence": 0.9,
            "success_rate": 0.85,
            "usage_count": 10,
            "tech_stack": {
                "primary_language": "java",
                "frameworks": ["spring"],
                "tools": ["maven"]
            }
        }
        
        pattern = pattern_portability.deserialize_pattern(pattern_data)
        
        assert pattern.pattern_id == "deserialized-pattern"
        assert pattern.name == "Deserialized Pattern"
        assert pattern.complexity == "high"
        assert pattern.confidence == 0.9
        assert pattern.tech_stack.primary_language == "java"
        assert "spring" in pattern.tech_stack.frameworks
    
    def test_export_patterns_all(self, pattern_portability, temp_patterns_dir):
        """Test exporting all patterns."""
        # Create test patterns
        self.create_sample_pattern_file(temp_patterns_dir, "export-001")
        self.create_sample_pattern_file(temp_patterns_dir, "export-002")
        
        json_data = pattern_portability.export_patterns()
        export_data = json.loads(json_data)
        
        assert export_data['version'] == "1.0.0"
        assert 'exported_at' in export_data
        assert len(export_data['patterns']) == 2
        assert export_data['metadata']['pattern_count'] == 2
        assert export_data['metadata']['source_project'] == "test-project"
    
    def test_export_patterns_specific(self, pattern_portability, temp_patterns_dir):
        """Test exporting specific patterns by ID."""
        self.create_sample_pattern_file(temp_patterns_dir, "specific-001")
        self.create_sample_pattern_file(temp_patterns_dir, "specific-002")
        self.create_sample_pattern_file(temp_patterns_dir, "specific-003")
        
        json_data = pattern_portability.export_patterns(pattern_ids=["specific-001", "specific-003"])
        export_data = json.loads(json_data)
        
        assert len(export_data['patterns']) == 2
        exported_ids = [p['pattern_id'] for p in export_data['patterns']]
        assert "specific-001" in exported_ids
        assert "specific-003" in exported_ids
        assert "specific-002" not in exported_ids
    
    def test_export_to_file(self, pattern_portability, temp_patterns_dir):
        """Test exporting patterns to a file."""
        self.create_sample_pattern_file(temp_patterns_dir, "file-export")
        
        output_file = Path(temp_patterns_dir) / "export.json"
        json_data = pattern_portability.export_patterns(output_file=str(output_file))
        
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            file_data = json.load(f)
        
        assert file_data['version'] == "1.0.0"
        assert len(file_data['patterns']) == 1
    
    def test_validate_version(self, pattern_portability):
        """Test version validation."""
        assert pattern_portability.validate_version("1.0.0") is True
        assert pattern_portability.validate_version("0.9.0") is False
        assert pattern_portability.validate_version("2.0.0") is False
    
    def test_pattern_exists(self, pattern_portability, temp_patterns_dir):
        """Test checking if pattern exists."""
        self.create_sample_pattern_file(temp_patterns_dir, "exists-test")
        
        assert pattern_portability.pattern_exists("exists-test") is True
        assert pattern_portability.pattern_exists("does-not-exist") is False
    
    def test_import_patterns_new(self, pattern_portability):
        """Test importing new patterns (no conflicts)."""
        import_data = {
            "version": "1.0.0",
            "patterns": [
                {
                    "pattern_id": "imported-001",
                    "name": "Imported Pattern 1",
                    "description": "First imported pattern",
                    "complexity": "low",
                    "domain": "import_test",
                    "confidence": 0.8
                },
                {
                    "pattern_id": "imported-002",
                    "name": "Imported Pattern 2",
                    "description": "Second imported pattern",
                    "complexity": "high",
                    "domain": "import_test",
                    "confidence": 0.9
                }
            ]
        }
        
        result = pattern_portability.import_patterns(import_data)
        
        assert result.imported_count == 2
        assert result.skipped_count == 0
        assert result.error_count == 0
        assert len(result.imported_patterns) == 2
        assert "imported-001" in result.imported_patterns
        assert "imported-002" in result.imported_patterns
    
    def test_import_patterns_conservative_merge(self, pattern_portability, temp_patterns_dir):
        """Test importing with conservative merge strategy (skip conflicts)."""
        # Create existing pattern
        self.create_sample_pattern_file(temp_patterns_dir, "conflict-pattern")
        
        import_data = {
            "version": "1.0.0",
            "patterns": [
                {
                    "pattern_id": "conflict-pattern",
                    "name": "Conflicting Pattern",
                    "description": "This will conflict",
                    "complexity": "high"
                },
                {
                    "pattern_id": "new-pattern",
                    "name": "New Pattern",
                    "description": "This is new",
                    "complexity": "medium"
                }
            ]
        }
        
        result = pattern_portability.import_patterns(import_data, MergeStrategy.CONSERVATIVE)
        
        assert result.imported_count == 1  # Only new pattern
        assert result.skipped_count == 1   # Conflicting pattern skipped
        assert result.error_count == 0
        assert len(result.conflicts) == 1
        assert result.conflicts[0].resolution == ConflictResolution.SKIPPED
    
    def test_import_patterns_overwrite_merge(self, pattern_portability, temp_patterns_dir):
        """Test importing with overwrite merge strategy."""
        self.create_sample_pattern_file(temp_patterns_dir, "overwrite-test")
        
        import_data = {
            "version": "1.0.0",
            "patterns": [
                {
                    "pattern_id": "overwrite-test",
                    "name": "Overwritten Pattern",
                    "description": "This will overwrite existing",
                    "complexity": "very-high",
                    "confidence": 0.95
                }
            ]
        }
        
        result = pattern_portability.import_patterns(import_data, MergeStrategy.OVERWRITE)
        
        assert result.imported_count == 1
        assert result.skipped_count == 0
        assert result.error_count == 0
        assert len(result.conflicts) == 1
        assert result.conflicts[0].resolution == ConflictResolution.OVERWRITTEN
        
        # Verify the pattern was actually overwritten
        pattern = pattern_portability.get_pattern("overwrite-test")
        assert pattern.name == "Overwritten Pattern"
        assert pattern.complexity == "very-high"
    
    def test_import_patterns_versioned_merge(self, pattern_portability, temp_patterns_dir):
        """Test importing with versioned merge strategy."""
        self.create_sample_pattern_file(temp_patterns_dir, "version-test")
        
        import_data = {
            "version": "1.0.0",
            "patterns": [
                {
                    "pattern_id": "version-test",
                    "name": "Versioned Pattern",
                    "description": "This will create a new version",
                    "complexity": "medium"
                }
            ]
        }
        
        result = pattern_portability.import_patterns(import_data, MergeStrategy.VERSIONED)
        
        assert result.imported_count == 1
        assert result.skipped_count == 0
        assert result.error_count == 0
        assert len(result.conflicts) == 1
        assert result.conflicts[0].resolution == ConflictResolution.VERSIONED
        
        # Check that the original pattern still exists
        original = pattern_portability.get_pattern("version-test")
        assert original is not None
        
        # Check that versioned pattern was created
        versioned_id = result.conflicts[0].pattern_id
        assert versioned_id.startswith("version-test_v")
        versioned_pattern = pattern_portability.get_pattern(versioned_id)
        assert versioned_pattern is not None
    
    def test_import_invalid_version(self, pattern_portability):
        """Test importing patterns with invalid version."""
        import_data = {
            "version": "2.0.0",  # Invalid version
            "patterns": [
                {
                    "pattern_id": "test-pattern",
                    "name": "Test Pattern"
                }
            ]
        }
        
        result = pattern_portability.import_patterns(import_data)
        
        assert result.imported_count == 0
        assert result.error_count == 1
        assert "Incompatible version" in result.errors[0]
    
    def test_calculate_avg_success_rate(self, pattern_portability, sample_pattern):
        """Test calculating average success rate."""
        patterns = []
        
        # Test empty list
        avg = pattern_portability.calculate_avg_success_rate(patterns)
        assert avg == 0.0
        
        # Test with patterns
        pattern1 = Pattern(pattern_id="p1", name="P1", description="", complexity="low")
        pattern1.success_rate = 0.8
        
        pattern2 = Pattern(pattern_id="p2", name="P2", description="", complexity="medium")
        pattern2.success_rate = 0.6
        
        patterns = [pattern1, pattern2]
        avg = pattern_portability.calculate_avg_success_rate(patterns)
        assert avg == 0.7  # (0.8 + 0.6) / 2
    
    def test_validate_patterns(self, pattern_portability):
        """Test pattern validation before import."""
        pattern_data_list = [
            {
                "pattern_id": "valid-pattern",
                "name": "Valid Pattern",
                "description": "A valid pattern",
                "confidence": 0.8,
                "success_rate": 0.75,
                "complexity": "medium"
            },
            {
                "pattern_id": "invalid-pattern",
                "name": "",  # Missing name
                "confidence": "not-a-number",  # Invalid confidence
                "success_rate": 1.5,  # Out of range
                "complexity": "invalid-complexity"
            },
            {
                # Missing pattern_id
                "name": "Missing ID Pattern",
                "description": "Pattern without ID"
            }
        ]
        
        results = pattern_portability.validate_patterns(pattern_data_list)
        
        assert len(results) == 3
        
        # First pattern should be valid
        assert results[0]['valid'] is True
        assert len(results[0]['errors']) == 0
        
        # Second pattern should have errors
        assert results[1]['valid'] is False
        assert len(results[1]['errors']) > 0
        assert any("Missing required field: name" in error for error in results[1]['errors'])
        assert any("Confidence must be a number" in error for error in results[1]['errors'])
        
        # Third pattern should have missing ID error
        assert results[2]['valid'] is False
        assert any("Missing required field: pattern_id" in error for error in results[2]['errors'])
    
    def test_get_export_stats(self, pattern_portability, temp_patterns_dir):
        """Test getting export statistics."""
        # Create patterns with different complexities and domains
        patterns_data = [
            {"pattern_id": "stats-001", "complexity": "low", "domain": "web", "success_rate": 0.8},
            {"pattern_id": "stats-002", "complexity": "medium", "domain": "web", "success_rate": 0.6},
            {"pattern_id": "stats-003", "complexity": "high", "domain": "api", "success_rate": 0.9},
        ]
        
        for pattern_data in patterns_data:
            full_data = {
                "pattern_name": f"Pattern {pattern_data['pattern_id']}",
                "description": "Test pattern for stats",
                **pattern_data
            }
            filepath = Path(temp_patterns_dir) / f"{pattern_data['pattern_id']}.json"
            with open(filepath, 'w') as f:
                json.dump(full_data, f)
        
        stats = pattern_portability.get_export_stats()
        
        assert stats['total_patterns'] == 3
        assert stats['complexity_breakdown']['low'] == 1
        assert stats['complexity_breakdown']['medium'] == 1
        assert stats['complexity_breakdown']['high'] == 1
        assert stats['domain_breakdown']['web'] == 2
        assert stats['domain_breakdown']['api'] == 1
        assert abs(stats['avg_success_rate'] - 0.7667) < 0.001  # (0.8 + 0.6 + 0.9) / 3
    
    def test_merge_patterns(self, pattern_portability):
        """Test pattern merging functionality."""
        existing = Pattern(
            pattern_id="merge-test",
            name="Existing Pattern",
            description="Original description",
            complexity="medium",
            tags=["original", "test"],
            usage_count=5,
            confidence=0.7,
            success_rate=0.6,
            implementation_steps=[{"step": 1, "description": "Original step"}],
            code_examples=[{"language": "python", "code": "print('original')"}]
        )
        
        new_data = {
            "pattern_id": "merge-test",
            "name": "Updated Pattern",
            "description": "Updated description",
            "tags": ["test", "updated", "new"],
            "usage_count": 3,
            "confidence": 0.8,
            "success_rate": 0.5,
            "implementation_steps": [{"step": 2, "description": "New step"}],
            "code_examples": [{"language": "java", "code": "System.out.println('new');"}]
        }
        
        merged_data, merged_fields = pattern_portability._merge_patterns(existing, new_data)
        
        # Check merged tags (union)
        assert set(merged_data['tags']) == {"original", "test", "updated", "new"}
        
        # Check summed usage count
        assert merged_data['usage_count'] == 8  # 5 + 3
        
        # Check higher confidence
        assert merged_data['confidence'] == 0.8  # Higher of 0.7 and 0.8
        
        # Check that success rate kept higher value (0.6 > 0.5)
        assert merged_data['success_rate'] == 0.6
        
        # Check appended implementation steps
        assert len(merged_data['implementation_steps']) == 2
        
        # Check merged fields tracking
        assert 'tags' in merged_fields
        assert 'usage_count' in merged_fields
        assert 'confidence' in merged_fields
    
    def test_create_versioned_id(self, pattern_portability):
        """Test creation of versioned pattern IDs."""
        original_id = "test-pattern-001"
        versioned_id = pattern_portability._create_versioned_id(original_id)
        
        assert versioned_id.startswith("test-pattern-001_v")
        assert len(versioned_id) > len(original_id)
        
        # Should contain timestamp
        timestamp_part = versioned_id.split("_v")[1]
        assert len(timestamp_part) == 15  # YYYYMMDD_HHMMSS format
    
    def test_json_import_export_roundtrip(self, pattern_portability, temp_patterns_dir):
        """Test complete export-import roundtrip maintains data integrity."""
        # Create original pattern
        original_pattern_data = {
            "pattern_id": "roundtrip-test",
            "pattern_name": "Roundtrip Test Pattern",
            "description": "Pattern for testing roundtrip import/export",
            "complexity": "high",
            "domain": "testing",
            "tags": ["roundtrip", "test", "integration"],
            "confidence": 0.85,
            "success_rate": 0.92,
            "usage_count": 15,
            "tech_stack": {
                "primary_language": "python",
                "frameworks": ["pytest", "unittest"],
                "tools": ["git", "docker"]
            },
            "implementation_steps": [
                {"step": 1, "description": "Setup test environment"},
                {"step": 2, "description": "Write comprehensive tests"},
                {"step": 3, "description": "Execute test suite"}
            ],
            "code_examples": [
                {
                    "language": "python",
                    "description": "Example test case",
                    "code": "def test_roundtrip():\n    assert export_import_works()"
                }
            ],
            "validation_criteria": [
                "All tests pass",
                "Coverage > 90%",
                "No linting errors"
            ]
        }
        
        # Save original pattern
        original_file = Path(temp_patterns_dir) / "roundtrip-test.json"
        with open(original_file, 'w') as f:
            json.dump(original_pattern_data, f, indent=2)
        
        # Export patterns
        exported_json = pattern_portability.export_patterns()
        
        # Clear patterns directory
        for file in Path(temp_patterns_dir).glob("*.json"):
            file.unlink()
        
        # Import patterns back
        result = pattern_portability.import_patterns(exported_json)
        
        # Verify import success
        assert result.imported_count == 1
        assert result.error_count == 0
        
        # Verify pattern integrity
        imported_pattern = pattern_portability.get_pattern("roundtrip-test")
        assert imported_pattern is not None
        assert imported_pattern.name == "Roundtrip Test Pattern"
        assert imported_pattern.complexity == "high"
        assert imported_pattern.confidence == 0.85
        assert imported_pattern.success_rate == 0.92
        assert imported_pattern.usage_count == 15
        assert set(imported_pattern.tags) == {"roundtrip", "test", "integration"}
        assert len(imported_pattern.implementation_steps) == 3
        assert len(imported_pattern.code_examples) == 1
        assert len(imported_pattern.validation_criteria) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])