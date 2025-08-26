#!/usr/bin/env python3
"""
Comprehensive Tests for Requirement Extraction System

Tests the requirement_extractor.py module with various document formats
and requirement types to ensure accurate extraction and categorization.

Author: RIF-Implementer
Issue: #240
"""

import unittest
import tempfile
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

# Import the module to test
import sys
sys.path.append(str(Path(__file__).parent.parent / "claude" / "commands"))
from requirement_extractor import RequirementExtractor, RequirementType, Priority, Complexity


class TestRequirementExtractor(unittest.TestCase):
    """Test cases for the RequirementExtractor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_xml = self._create_sample_xml()
        self.extractor = RequirementExtractor(flattened_xml=self.sample_xml)
    
    def _create_sample_xml(self) -> str:
        """Create sample flattened XML for testing"""
        xml_content = """<?xml version="1.0" ?>
<project_documentation version="1.0" timestamp="2025-01-01T00:00:00" source_dir="/test">
  <metadata>
    <project_name>TestProject</project_name>
    <generated_at>2025-01-01T00:00:00</generated_at>
    <source_directory>/test</source_directory>
  </metadata>
  <documents>
    <document type="prd" path="test.prd.md" name="test" size="1000" lines="50">
      <section level="1" title="User Management">
        <content>
Epic 1: User Authentication System
This epic covers all authentication-related functionality.

User Story US-001: As a user, I want to register for an account so that I can access the system.
Acceptance Criteria:
- Email validation must work correctly
- Password must meet security requirements
- Confirmation email must be sent

The system must validate user credentials during login.
FR-001: The system shall encrypt all passwords using bcrypt.
        </content>
      </section>
      <section level="2" title="Performance Requirements">
        <content>
Performance: The system should respond within 2 seconds for all user operations.
Security: All API endpoints must implement rate limiting.
NFR-001: System availability should be 99.9% uptime.
        </content>
      </section>
    </document>
    <document type="architecture" path="arch.md" name="architecture" size="800" lines="40">
      <section level="1" title="System Architecture">
        <content>
Technical Requirement TR-001: The system shall use microservices architecture.
API: RESTful APIs must follow OpenAPI 3.0 specification.
Database: PostgreSQL will be used for primary data storage.
Constraint: The system must be deployable on AWS infrastructure.
        </content>
      </section>
    </document>
    <document type="design" path="design.md" name="ui-design" size="600" lines="30">
      <section level="1" title="User Interface Design">
        <content>
As a user, I want to see a clean dashboard so that I can quickly access my information.
Usability: The interface must be accessible to users with disabilities.
Business Rule: Only authenticated users can access protected resources.
        </content>
      </section>
    </document>
    <document type="requirements" path="requirements.md" name="requirements" size="1200" lines="60">
      <section level="1" title="Functional Requirements">
        <content>
Epic 2: Data Management
This epic handles all data-related operations.

FR-002: The system must provide data export functionality.
FR-003: The system shall support batch data import.
AC-001: Given valid data, When user exports, Then CSV file is generated.
Objective: Improve data accessibility for business users.
Constraint: Export size limited to 10MB per file.
        </content>
      </section>
    </document>
  </documents>
  <dependencies/>
  <cross_references/>
</project_documentation>"""
        return xml_content
    
    def test_initialization_with_xml_string(self):
        """Test extractor initialization with XML string"""
        extractor = RequirementExtractor(flattened_xml=self.sample_xml)
        self.assertIsNotNone(extractor.root)
        self.assertEqual(extractor.root.tag, "project_documentation")
    
    def test_initialization_with_xml_file(self):
        """Test extractor initialization with XML file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(self.sample_xml)
            xml_file_path = f.name
        
        try:
            extractor = RequirementExtractor(xml_file_path=xml_file_path)
            self.assertIsNotNone(extractor.root)
            self.assertEqual(extractor.root.tag, "project_documentation")
        finally:
            Path(xml_file_path).unlink()  # Clean up
    
    def test_initialization_error_handling(self):
        """Test error handling for invalid initialization"""
        with self.assertRaises(ValueError):
            RequirementExtractor()  # No parameters provided
        
        with self.assertRaises(ValueError):
            RequirementExtractor(xml_file_path="/non/existent/file.xml")
    
    def test_extract_all_requirements(self):
        """Test the main extraction method"""
        results = self.extractor.extract_all_requirements()
        
        # Check structure of results
        self.assertIn('epics', results)
        self.assertIn('stories', results)
        self.assertIn('requirements', results)
        self.assertIn('hierarchy', results)
        self.assertIn('statistics', results)
        self.assertIn('extraction_metadata', results)
        
        # Check that we found some requirements
        self.assertGreater(len(results['requirements']), 0)
        self.assertGreater(len(results['epics']), 0)
        self.assertGreater(len(results['stories']), 0)
    
    def test_epic_extraction(self):
        """Test extraction of epics"""
        results = self.extractor.extract_all_requirements()
        epics = results['epics']
        
        self.assertGreater(len(epics), 0)
        
        # Check epic structure
        epic = epics[0]
        required_fields = ['id', 'type', 'title', 'description', 'source_document', 'source_section']
        for field in required_fields:
            self.assertIn(field, epic)
        
        # Check that epic ID follows expected format
        self.assertTrue(epic['id'].startswith('EPIC-'))
        
        # Check that type is correct
        self.assertEqual(epic['type'], RequirementType.EPIC.value)
    
    def test_user_story_extraction(self):
        """Test extraction of user stories"""
        results = self.extractor.extract_all_requirements()
        stories = results['stories']
        
        self.assertGreater(len(stories), 0)
        
        # Check story structure
        story = stories[0]
        required_fields = ['id', 'type', 'title', 'actor', 'action', 'description']
        for field in required_fields:
            self.assertIn(field, story)
        
        # Check that story ID follows expected format
        self.assertTrue(story['id'].startswith('US-'))
        
        # Check that type is correct
        self.assertEqual(story['type'], RequirementType.USER_STORY.value)
    
    def test_functional_requirement_extraction(self):
        """Test extraction of functional requirements"""
        results = self.extractor.extract_all_requirements()
        
        functional_reqs = [req for req in results['requirements'] 
                          if req['type'] == RequirementType.FUNCTIONAL.value]
        
        self.assertGreater(len(functional_reqs), 0)
        
        # Check functional requirement structure
        req = functional_reqs[0]
        required_fields = ['id', 'type', 'title', 'description', 'priority', 'complexity']
        for field in required_fields:
            self.assertIn(field, req)
        
        # Check that functional requirement ID follows expected format
        self.assertTrue(req['id'].startswith('FR-'))
    
    def test_non_functional_requirement_extraction(self):
        """Test extraction of non-functional requirements"""
        results = self.extractor.extract_all_requirements()
        
        nfr_reqs = [req for req in results['requirements'] 
                   if req['type'] == RequirementType.NON_FUNCTIONAL.value]
        
        self.assertGreater(len(nfr_reqs), 0)
        
        # Check that NFR was properly extracted
        nfr = nfr_reqs[0]
        self.assertTrue(nfr['id'].startswith('NFR-'))
    
    def test_technical_requirement_extraction(self):
        """Test extraction of technical requirements"""
        results = self.extractor.extract_all_requirements()
        
        tech_reqs = [req for req in results['requirements'] 
                    if req['type'] == RequirementType.TECHNICAL.value]
        
        self.assertGreater(len(tech_reqs), 0)
        
        # Check that technical requirement was properly extracted
        tech_req = tech_reqs[0]
        self.assertTrue(tech_req['id'].startswith('TR-'))
    
    def test_acceptance_criteria_extraction(self):
        """Test extraction of acceptance criteria"""
        results = self.extractor.extract_all_requirements()
        
        # Check that user stories have acceptance criteria
        stories = results['stories']
        story_with_criteria = None
        for story in stories:
            if story.get('acceptance_criteria'):
                story_with_criteria = story
                break
        
        self.assertIsNotNone(story_with_criteria, "No story found with acceptance criteria")
        self.assertGreater(len(story_with_criteria['acceptance_criteria']), 0)
    
    def test_constraint_extraction(self):
        """Test extraction of constraints"""
        results = self.extractor.extract_all_requirements()
        
        constraints = [req for req in results['requirements'] 
                      if req['type'] == RequirementType.CONSTRAINT.value]
        
        self.assertGreater(len(constraints), 0)
        
        constraint = constraints[0]
        self.assertTrue(constraint['id'].startswith('CON-'))
    
    def test_business_requirement_extraction(self):
        """Test extraction of business requirements"""
        results = self.extractor.extract_all_requirements()
        
        business_reqs = [req for req in results['requirements'] 
                        if req['type'] == RequirementType.BUSINESS.value]
        
        self.assertGreater(len(business_reqs), 0)
        
        business_req = business_reqs[0]
        self.assertTrue(business_req['id'].startswith('BR-'))
    
    def test_priority_assignment(self):
        """Test that priorities are assigned correctly"""
        results = self.extractor.extract_all_requirements()
        
        # Check that all requirements have priorities
        for req in results['requirements']:
            self.assertIn('priority', req)
            self.assertIn(req['priority'], [p.value for p in Priority])
        
        # Check that stories have priorities too
        for story in results['stories']:
            if 'priority' in story:
                self.assertIn(story['priority'], [p.value for p in Priority])
    
    def test_complexity_estimation(self):
        """Test that complexity is estimated correctly"""
        results = self.extractor.extract_all_requirements()
        
        # Check that all requirements have complexity
        for req in results['requirements']:
            self.assertIn('complexity', req)
            self.assertIn(req['complexity'], [c.value for c in Complexity])
        
        # Check that stories have complexity too
        for story in results['stories']:
            if 'complexity' in story:
                self.assertIn(story['complexity'], [c.value for c in Complexity])
    
    def test_token_estimation(self):
        """Test that token counts are estimated"""
        results = self.extractor.extract_all_requirements()
        
        # Check that all requirements have token estimates
        for req in results['requirements']:
            self.assertIn('estimated_tokens', req)
            self.assertIsInstance(req['estimated_tokens'], int)
            self.assertGreater(req['estimated_tokens'], 0)
    
    def test_story_points_estimation(self):
        """Test that story points are estimated for user stories"""
        results = self.extractor.extract_all_requirements()
        
        # Check that stories have point estimates
        for story in results['stories']:
            if 'estimated_points' in story:
                self.assertIsInstance(story['estimated_points'], int)
                self.assertGreaterEqual(story['estimated_points'], 1)
                self.assertLessEqual(story['estimated_points'], 13)  # Fibonacci scale cap
    
    def test_tag_extraction(self):
        """Test that relevant tags are extracted"""
        results = self.extractor.extract_all_requirements()
        
        # Check that requirements have tags
        for req in results['requirements']:
            self.assertIn('tags', req)
            self.assertIsInstance(req['tags'], list)
    
    def test_hierarchy_building(self):
        """Test that requirement hierarchy is built correctly"""
        results = self.extractor.extract_all_requirements()
        hierarchy = results['hierarchy']
        
        self.assertIsInstance(hierarchy, dict)
        
        # Check that epics are in hierarchy
        epic_ids = [epic['id'] for epic in results['epics']]
        for epic_id in epic_ids:
            if epic_id in hierarchy:
                self.assertEqual(hierarchy[epic_id]['type'], 'epic')
                self.assertIn('stories', hierarchy[epic_id])
                self.assertIn('requirements', hierarchy[epic_id])
    
    def test_statistics_generation(self):
        """Test that statistics are generated correctly"""
        results = self.extractor.extract_all_requirements()
        stats = results['statistics']
        
        required_stats = ['total_requirements', 'total_epics', 'total_stories', 
                         'by_type', 'by_priority', 'by_complexity', 
                         'avg_tokens', 'total_acceptance_criteria']
        
        for stat in required_stats:
            self.assertIn(stat, stats)
        
        # Check that counts match actual items
        self.assertEqual(stats['total_requirements'], len(results['requirements']))
        self.assertEqual(stats['total_epics'], len(results['epics']))
        self.assertEqual(stats['total_stories'], len(results['stories']))
    
    def test_save_requirements_json(self):
        """Test saving requirements to JSON file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            saved_path = self.extractor.save_requirements_json(output_path)
            self.assertEqual(str(saved_path), output_path)
            self.assertTrue(Path(output_path).exists())
            
            # Verify the JSON content is valid
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.assertIn('requirements', data)
            self.assertIn('epics', data)
            self.assertIn('stories', data)
            
        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()  # Clean up
    
    def test_filtering_methods(self):
        """Test requirement filtering methods"""
        results = self.extractor.extract_all_requirements()
        
        # Test filtering by type
        functional_reqs = self.extractor.get_requirements_by_type(RequirementType.FUNCTIONAL)
        self.assertIsInstance(functional_reqs, list)
        
        # Test filtering by priority
        high_priority_reqs = self.extractor.get_requirements_by_priority(Priority.HIGH)
        self.assertIsInstance(high_priority_reqs, list)
        
        # Test filtering by complexity
        medium_complexity_reqs = self.extractor.get_requirements_by_complexity(Complexity.MEDIUM)
        self.assertIsInstance(medium_complexity_reqs, list)
    
    def test_duplicate_prevention(self):
        """Test that duplicate requirements are not created"""
        # Run extraction twice and check for duplicates
        results1 = self.extractor.extract_all_requirements()
        results2 = self.extractor.extract_all_requirements()
        
        # Should have same number of items (no duplicates added)
        self.assertEqual(len(results1['requirements']), len(results2['requirements']))
        self.assertEqual(len(results1['epics']), len(results2['epics']))
        self.assertEqual(len(results1['stories']), len(results2['stories']))
    
    def test_empty_content_handling(self):
        """Test handling of empty or invalid content"""
        empty_xml = """<?xml version="1.0" ?>
<project_documentation version="1.0" timestamp="2025-01-01T00:00:00" source_dir="/test">
  <documents>
    <document type="prd" path="empty.md" name="empty" size="0" lines="0">
      <section level="1" title="Empty Section">
        <content></content>
      </section>
    </document>
  </documents>
</project_documentation>"""
        
        extractor = RequirementExtractor(flattened_xml=empty_xml)
        results = extractor.extract_all_requirements()
        
        # Should handle empty content gracefully
        self.assertIsInstance(results, dict)
        self.assertIn('requirements', results)
        self.assertIn('epics', results)
        self.assertIn('stories', results)


class TestRequirementPatterns(unittest.TestCase):
    """Test cases for requirement pattern matching"""
    
    def test_epic_patterns(self):
        """Test epic pattern matching"""
        from requirement_extractor import RequirementPatterns
        patterns = RequirementPatterns()
        
        test_texts = [
            "Epic 1: User Management System",
            "Feature: Advanced Search",
            "Module: Payment Processing",
            "Component: Authentication Service"
        ]
        
        for text in test_texts:
            matched = False
            for pattern in patterns.EPIC_PATTERNS:
                if re.match(pattern, text, re.IGNORECASE):
                    matched = True
                    break
            self.assertTrue(matched, f"Failed to match epic pattern: {text}")
    
    def test_user_story_patterns(self):
        """Test user story pattern matching"""
        from requirement_extractor import RequirementPatterns
        patterns = RequirementPatterns()
        
        test_texts = [
            "As a user, I want to login so that I can access my account",
            "User Story: Create user registration form",
            "US-001: Implement password reset functionality",
            "Scenario: User views dashboard"
        ]
        
        for text in test_texts:
            matched = False
            for pattern in patterns.USER_STORY_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    matched = True
                    break
            self.assertTrue(matched, f"Failed to match user story pattern: {text}")
    
    def test_functional_requirement_patterns(self):
        """Test functional requirement pattern matching"""
        from requirement_extractor import RequirementPatterns
        patterns = RequirementPatterns()
        
        test_texts = [
            "The system shall validate user input",
            "FR-001: User authentication is required",
            "Function: Generate monthly reports",
            "Capability: Real-time data processing"
        ]
        
        for text in test_texts:
            matched = False
            for pattern in patterns.FUNCTIONAL_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    matched = True
                    break
            self.assertTrue(matched, f"Failed to match functional requirement pattern: {text}")


if __name__ == '__main__':
    # Add import path
    import re
    
    # Run the tests
    unittest.main(verbosity=2)