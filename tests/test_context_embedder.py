#!/usr/bin/env python3
"""
Unit tests for Context Embedding Engine
"""

import unittest
import json
import tempfile
from pathlib import Path
from claude.commands.context_embedder import ContextEmbedder, TokenManager, ContextTemplates, ContextWindow


class TestTokenManager(unittest.TestCase):
    """Test the TokenManager functionality"""
    
    def setUp(self):
        self.token_manager = TokenManager()
    
    def test_count_tokens(self):
        """Test token counting functionality"""
        test_cases = [
            ("", 0),
            ("Hello world", 2),
            ("This is a longer sentence with more tokens.", 9),
        ]
        
        for text, expected_min_tokens in test_cases:
            tokens = self.token_manager.count_tokens(text)
            if expected_min_tokens == 0:
                self.assertEqual(tokens, 0)
            else:
                # Tokens should be roughly proportional to word count
                self.assertGreaterEqual(tokens, expected_min_tokens - 2)
                self.assertLessEqual(tokens, expected_min_tokens + 2)
    
    def test_trim_to_tokens(self):
        """Test token trimming functionality"""
        long_text = "This is a very long text that should be trimmed. " * 20
        
        # Test trimming to small token count
        trimmed = self.token_manager.trim_to_tokens(long_text, 10)
        trimmed_tokens = self.token_manager.count_tokens(trimmed)
        self.assertLessEqual(trimmed_tokens, 10)
        self.assertTrue(trimmed.endswith("..."))
        
        # Test text that doesn't need trimming
        short_text = "Short text"
        not_trimmed = self.token_manager.trim_to_tokens(short_text, 100)
        self.assertEqual(short_text, not_trimmed)
    
    def test_optimize_context(self):
        """Test context optimization functionality"""
        context_parts = {
            'primary_requirement': 'This is the main requirement description.',
            'architecture_context': 'Architecture details go here.',
            'implementation_hints': 'Implementation guidance is provided here.',
            'design_context': 'Design information and UI details.'
        }
        
        optimized = self.token_manager.optimize_context(context_parts, max_total=20)
        
        total_tokens = sum(self.token_manager.count_tokens(text) for text in optimized.values())
        self.assertLessEqual(total_tokens, 20)
        
        # Primary requirement should be preserved best
        self.assertIn('This is the main requirement', optimized['primary_requirement'])


class TestContextEmbedder(unittest.TestCase):
    """Test the ContextEmbedder functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_xml = '''<?xml version="1.0"?>
<project_documentation>
  <documents>
    <document type="prd" name="test-prd">
      <section level="1" title="Test Section">
        <content>Test content for authentication and user management.</content>
      </section>
    </document>
    <document type="architecture" name="test-arch">
      <section level="1" title="Auth Service">
        <content>AuthenticationService handles user login. Uses TokenManager and UserRepository components.</content>
      </section>
    </document>
  </documents>
</project_documentation>'''
        
        self.sample_requirements = {
            "requirements": [
                {
                    "id": "TEST-001",
                    "type": "functional",
                    "title": "Test Authentication",
                    "description": "System shall provide authentication functionality",
                    "source_document": "test-prd.md",
                    "source_section": "Test Section", 
                    "priority": "high",
                    "complexity": "medium",
                    "dependencies": [],
                    "acceptance_criteria": ["User can log in", "System validates credentials"],
                    "tags": ["auth", "security"]
                }
            ],
            "stories": [
                {
                    "id": "US-TEST-001",
                    "type": "user_story",
                    "title": "User Login Test",
                    "actor": "user",
                    "action": "log in to the system",
                    "benefit": "access my account",
                    "description": "As a user, I want to log in to the system so that I can access my account",
                    "acceptance_criteria": ["Login form works", "Credentials validated"],
                    "estimated_points": 3,
                    "complexity": "medium",
                    "priority": "high",
                    "tags": ["ui", "auth"]
                }
            ]
        }
    
    def create_embedder(self):
        """Create embedder instance with test data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as xml_file:
            xml_file.write(self.sample_xml)
            xml_path = xml_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as req_file:
            json.dump(self.sample_requirements, req_file)
            req_path = req_file.name
        
        embedder = ContextEmbedder(xml_file_path=xml_path, requirements_file_path=req_path)
        
        # Clean up temp files
        Path(xml_path).unlink()
        Path(req_path).unlink()
        
        return embedder
    
    def test_initialization(self):
        """Test embedder initialization"""
        embedder = self.create_embedder()
        self.assertIsInstance(embedder, ContextEmbedder)
        self.assertTrue(len(embedder.documents) > 0)
        self.assertTrue(len(embedder.requirements) > 0)
    
    def test_get_requirement(self):
        """Test requirement retrieval"""
        embedder = self.create_embedder()
        
        req = embedder._get_requirement("TEST-001")
        self.assertIsNotNone(req)
        self.assertEqual(req['id'], "TEST-001")
        
        story = embedder._get_requirement("US-TEST-001")
        self.assertIsNotNone(story)
        self.assertEqual(story['id'], "US-TEST-001")
        
        missing = embedder._get_requirement("NONEXISTENT")
        self.assertIsNone(missing)
    
    def test_embed_context_for_requirement(self):
        """Test context embedding for a requirement"""
        embedder = self.create_embedder()
        
        context = embedder.embed_context_for_requirement("TEST-001")
        
        self.assertIsInstance(context, ContextWindow)
        self.assertGreater(len(context.primary_requirement), 0)
        self.assertGreater(context.estimated_tokens, 0)
        self.assertLessEqual(context.estimated_tokens, 2500)
    
    def test_embed_context_for_user_story(self):
        """Test context embedding for a user story"""
        embedder = self.create_embedder()
        
        context = embedder.embed_context_for_requirement("US-TEST-001")
        
        self.assertIsInstance(context, ContextWindow)
        self.assertGreater(len(context.primary_requirement), 0)
        # User stories should get design context
        self.assertGreater(len(context.design_context), 10)
    
    def test_missing_requirement(self):
        """Test handling of missing requirements"""
        embedder = self.create_embedder()
        
        context = embedder.embed_context_for_requirement("MISSING-001")
        
        self.assertIsInstance(context, ContextWindow)
        self.assertEqual(context.primary_requirement, "Requirement not found")
        self.assertEqual(context.estimated_tokens, 0)


class TestContextTemplates(unittest.TestCase):
    """Test the ContextTemplates functionality"""
    
    def setUp(self):
        self.sample_requirement = {
            "id": "TEST-001",
            "type": "functional",
            "title": "Test Requirement",
            "priority": "high",
            "complexity": "medium"
        }
        
        self.sample_context = ContextWindow(
            primary_requirement="Test primary requirement",
            related_requirements=["REQ-002", "REQ-003"],
            architecture_context="Test architecture context",
            design_context="Test design context",
            dependencies=["Database", "API Framework"],
            implementation_hints="Test implementation hints",
            acceptance_criteria=["Criterion 1", "Criterion 2"],
            estimated_tokens=500,
            overflow_strategy="none"
        )
    
    def test_issue_with_full_context(self):
        """Test issue template generation"""
        issue = ContextTemplates.issue_with_full_context(
            self.sample_requirement, 
            self.sample_context
        )
        
        self.assertIsInstance(issue, str)
        self.assertIn("Test Requirement", issue)
        self.assertIn("Test primary requirement", issue)
        self.assertIn("Implementation", issue)
        self.assertIn("Acceptance Criteria", issue)
        self.assertIn("Database", issue)
        self.assertIn("REQ-002", issue)
    
    def test_user_story_with_context(self):
        """Test user story template generation"""
        story = {
            "id": "US-001",
            "title": "Test Story",
            "actor": "user",
            "estimated_points": 5
        }
        
        story_issue = ContextTemplates.user_story_with_context(story, self.sample_context)
        
        self.assertIsInstance(story_issue, str)
        self.assertIn("User Story: Test Story", story_issue)
        self.assertIn("Actor**: user", story_issue)
        self.assertIn("Story Points**: 5", story_issue)
    
    def test_epic_with_context(self):
        """Test epic template generation"""
        epic = {
            "id": "EPIC-001",
            "title": "Test Epic",
            "estimated_effort": "2 weeks",
            "stories": ["US-001", "US-002"]
        }
        
        epic_issue = ContextTemplates.epic_with_context(epic, self.sample_context)
        
        self.assertIsInstance(epic_issue, str)
        self.assertIn("Epic: Test Epic", epic_issue)
        self.assertIn("Estimated Effort**: 2 weeks", epic_issue)
        self.assertIn("US-001", epic_issue)
        self.assertIn("US-002", epic_issue)


if __name__ == '__main__':
    unittest.main()