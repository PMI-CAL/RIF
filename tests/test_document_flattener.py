#!/usr/bin/env python3
"""
Comprehensive test suite for Document Flattener Engine

Tests all functionality including markdown parsing, XML generation,
dependency tracking, and various document format handling.

Author: RIF-Implementer
Issue: #239
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from claude.commands.document_flattener import DocumentFlattener


class TestDocumentFlattener(unittest.TestCase):
    """Test cases for DocumentFlattener class"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.flattener = DocumentFlattener(self.test_dir)
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
        
    def create_test_document(self, filename: str, content: str) -> Path:
        """Helper method to create test documents"""
        doc_path = self.test_dir / filename
        doc_path.write_text(content, encoding='utf-8')
        return doc_path
    
    def test_init_valid_directory(self):
        """Test initialization with valid directory"""
        self.assertEqual(self.flattener.docs_dir, self.test_dir)
        self.assertEqual(len(self.flattener.document_registry), 0)
        
    def test_init_invalid_directory(self):
        """Test initialization with invalid directory"""
        with self.assertRaises(FileNotFoundError):
            DocumentFlattener("/nonexistent/directory")
    
    def test_extract_sections_basic(self):
        """Test basic section extraction from markdown"""
        content = """# Main Title
Some content here

## Subsection
More content

### Sub-subsection
Final content"""
        
        sections = self.flattener._extract_sections(content)
        
        self.assertEqual(len(sections), 3)
        self.assertEqual(sections[0]['level'], 1)
        self.assertEqual(sections[0]['title'], 'Main Title')
        self.assertEqual(sections[1]['level'], 2)
        self.assertEqual(sections[1]['title'], 'Subsection')
        self.assertEqual(sections[2]['level'], 3)
        self.assertEqual(sections[2]['title'], 'Sub-subsection')
    
    def test_extract_sections_no_headings(self):
        """Test section extraction with no headings"""
        content = "Just some plain content without headings."
        
        sections = self.flattener._extract_sections(content)
        
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]['level'], 1)
        self.assertEqual(sections[0]['title'], 'Document Content')
        self.assertIn('Just some plain content', sections[0]['content'])
    
    def test_extract_sections_empty_content(self):
        """Test section extraction with empty content"""
        content = ""
        
        sections = self.flattener._extract_sections(content)
        
        self.assertEqual(len(sections), 0)
    
    def test_parse_user_stories_basic(self):
        """Test parsing basic user stories"""
        content = """As a user, I want to login so that I can access my account.
        
        Acceptance Criteria:
        - System validates credentials
        - User is redirected to dashboard
        - Invalid login shows error message"""
        
        stories = self.flattener._parse_user_stories(content)
        
        self.assertEqual(len(stories), 1)
        story = stories[0]
        self.assertEqual(story['user_type'], 'user')
        self.assertIn('login', story['want'])
        self.assertIn('access my account', story['reason'])
        self.assertEqual(len(story.get('criteria', [])), 3)
    
    def test_parse_user_stories_with_id(self):
        """Test parsing user stories with IDs"""
        content = """User Story US-001: As a developer, I want API documentation so that I can integrate easily."""
        
        stories = self.flattener._parse_user_stories(content)
        
        self.assertEqual(len(stories), 1)
        self.assertEqual(stories[0]['id'], 'US-001')
        self.assertEqual(stories[0]['user_type'], 'developer')
    
    def test_parse_functional_requirements(self):
        """Test parsing functional requirements"""
        content = """FR-001: The system must authenticate users within 2 seconds
        REQ-002: Critical data backup must occur daily
        R-003: Nice to have feature for user preferences"""
        
        requirements = self.flattener._parse_functional_requirements(content)
        
        self.assertEqual(len(requirements), 3)
        self.assertEqual(requirements[0]['id'], 'FR-001')
        self.assertEqual(requirements[0]['priority'], 'medium')
        self.assertEqual(requirements[1]['priority'], 'high')  # Contains 'critical'
        self.assertEqual(requirements[2]['priority'], 'low')   # Contains 'nice to have'
    
    def test_parse_components(self):
        """Test parsing system components"""
        content = """## AuthService
        
        Handles user authentication and session management.
        
        - UserService: Manages user profiles and data
        * DatabaseService - Provides data persistence layer"""
        
        components = self.flattener._parse_components(content)
        
        self.assertGreaterEqual(len(components), 1)
        
        # Find AuthService component
        auth_service = next((c for c in components if c['name'] == 'AuthService'), None)
        self.assertIsNotNone(auth_service)
        self.assertEqual(auth_service['type'], 'service')
        self.assertIn('authentication', auth_service['description'])
    
    def test_parse_ui_components(self):
        """Test parsing UI components"""
        content = """## LoginButton
        
        Primary button for user login functionality.
        
        - SignupForm: Form for new user registration
        * ModalDialog - Displays important notifications"""
        
        components = self.flattener._parse_ui_components(content)
        
        self.assertGreaterEqual(len(components), 1)
        
        # Find LoginButton component
        login_button = next((c for c in components if c['name'] == 'LoginButton'), None)
        self.assertIsNotNone(login_button)
        self.assertEqual(login_button['type'], 'button')
    
    def test_process_prd_document(self):
        """Test processing a PRD document"""
        prd_content = """# Product Requirements Document
        
        ## User Stories
        
        As a customer, I want to place orders so that I can buy products.
        
        Acceptance Criteria:
        - User can select products
        - Payment processing works
        - Order confirmation is sent
        
        ## Functional Requirements
        
        FR-001: System must process payments securely
        FR-002: Order history must be maintained"""
        
        doc_path = self.create_test_document("product.prd.md", prd_content)
        element = self.flattener._process_document(doc_path, "prd")
        
        self.assertIsNotNone(element)
        self.assertEqual(element.get('type'), 'prd')
        self.assertEqual(element.get('name'), 'product')
        
        # Check for sections
        sections = element.findall('section')
        self.assertGreater(len(sections), 0)
        
        # Check for requirements in sections
        requirements_found = False
        for section in sections:
            if section.find('requirements') is not None:
                requirements_found = True
                break
        
        self.assertTrue(requirements_found)
    
    def test_process_architecture_document(self):
        """Test processing an architecture document"""
        arch_content = """# System Architecture
        
        ## Core Components
        
        ### AuthService
        
        Handles user authentication and authorization.
        
        ### DatabaseService
        
        Manages data persistence and retrieval."""
        
        doc_path = self.create_test_document("architecture.md", arch_content)
        element = self.flattener._process_document(doc_path, "architecture")
        
        self.assertIsNotNone(element)
        self.assertEqual(element.get('type'), 'architecture')
        
        # Check for components in sections
        components_found = False
        sections = element.findall('section')
        for section in sections:
            if section.find('components') is not None:
                components_found = True
                break
        
        self.assertTrue(components_found)
    
    def test_build_dependency_graph(self):
        """Test building dependency graph"""
        # Create documents with references to each other
        doc1_content = """# Document 1
        
        See also architecture.md for system design.
        Refer to requirements.md for detailed specifications."""
        
        doc2_content = """# Architecture Document
        
        This implements the requirements from requirements.md."""
        
        doc3_content = """# Requirements Document
        
        Base requirements for the system."""
        
        self.create_test_document("doc1.md", doc1_content)
        self.create_test_document("architecture.md", doc2_content)
        self.create_test_document("requirements.md", doc3_content)
        
        # Process documents to build registry
        self.flattener._process_document(self.test_dir / "doc1.md", "documentation")
        self.flattener._process_document(self.test_dir / "architecture.md", "architecture")
        self.flattener._process_document(self.test_dir / "requirements.md", "requirements")
        
        # Build dependency graph
        graph = self.flattener._build_dependency_graph()
        
        self.assertIn("doc1.md", graph)
        self.assertIn("architecture.md", graph)
        self.assertIn("requirements.md", graph)
        
        # Check dependencies
        doc1_deps = graph["doc1.md"]["depends_on"]
        self.assertIn("architecture.md", doc1_deps)
        self.assertIn("requirements.md", doc1_deps)
    
    def test_flatten_all_documents_comprehensive(self):
        """Test complete flattening process with multiple document types"""
        # Create various document types
        prd_content = """# Product Requirements
        
        As a user, I want to login so that I can access the system.
        
        FR-001: Authentication must be secure"""
        
        arch_content = """# System Architecture
        
        ## AuthService
        
        Handles user authentication."""
        
        design_content = """# UI Design
        
        ## LoginButton
        
        Primary authentication button."""
        
        req_content = """# Requirements
        
        REQ-001: High priority security requirement"""
        
        self.create_test_document("product.prd.md", prd_content)
        self.create_test_document("architecture.md", arch_content)
        self.create_test_document("ui-design.md", design_content)
        self.create_test_document("requirements.md", req_content)
        
        # Flatten all documents
        xml_content = self.flattener.flatten_all_documents()
        
        # Parse and validate XML
        root = ET.fromstring(xml_content)
        
        # Check root element
        self.assertEqual(root.tag, "project_documentation")
        self.assertEqual(root.get("version"), "1.0")
        
        # Check metadata
        metadata = root.find("metadata")
        self.assertIsNotNone(metadata)
        
        total_docs = metadata.find("total_documents")
        self.assertIsNotNone(total_docs)
        self.assertGreaterEqual(int(total_docs.text), 4)
        
        # Check documents section
        documents = root.find("documents")
        self.assertIsNotNone(documents)
        
        doc_elements = documents.findall("document")
        # Should have our 4 test documents, possibly with additional general docs excluded
        self.assertGreaterEqual(len(doc_elements), 4)
        
        # Verify different document types are present
        doc_types = [doc.get("type") for doc in doc_elements]
        self.assertIn("prd", doc_types)
        self.assertIn("architecture", doc_types)
        self.assertIn("design", doc_types)
        self.assertIn("requirements", doc_types)
        
        # Check dependencies section
        dependencies = root.find("dependencies")
        self.assertIsNotNone(dependencies)
        
        # Check cross-references section
        references = root.find("cross_references")
        self.assertIsNotNone(references)
    
    def test_get_document_stats(self):
        """Test document statistics generation"""
        # Create test documents
        self.create_test_document("doc1.md", "# Test\nContent here")
        self.create_test_document("doc2.prd.md", "# PRD\nMore content")
        
        # Process documents
        self.flattener.flatten_all_documents()
        
        stats = self.flattener.get_document_stats()
        
        self.assertEqual(stats['total_documents'], 2)
        self.assertGreater(stats['total_size'], 0)
        self.assertGreater(stats['total_lines'], 0)
        self.assertIn('by_type', stats)
    
    def test_save_flattened_xml(self):
        """Test saving XML to file"""
        # Create a simple document and flatten
        self.create_test_document("test.md", "# Test Document\nContent")
        xml_content = self.flattener.flatten_all_documents()
        
        # Save to file
        output_path = self.test_dir / "output.xml"
        saved_path = self.flattener.save_flattened_xml(output_path)
        
        self.assertEqual(saved_path, output_path)
        self.assertTrue(output_path.exists())
        
        # Verify content
        saved_content = output_path.read_text(encoding='utf-8')
        self.assertIn('project_documentation', saved_content)
    
    def test_large_document_performance(self):
        """Test performance with larger documents"""
        # Create a large document
        large_content = []
        for i in range(100):
            large_content.append(f"# Section {i}\n")
            large_content.append(f"This is content for section {i}.\n")
            large_content.append(f"As a user, I want feature {i} so that I can do task {i}.\n")
            large_content.append(f"FR-{i:03d}: Functional requirement {i}\n")
            large_content.append("\n")
        
        self.create_test_document("large.prd.md", "".join(large_content))
        
        # Measure performance
        import time
        start_time = time.time()
        
        xml_content = self.flattener.flatten_all_documents()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should process in reasonable time (less than 5 seconds as per requirements)
        self.assertLess(duration, 5.0)
        
        # Verify content was processed
        self.assertIsNotNone(xml_content)
        self.assertIn('project_documentation', xml_content)
        
        # Check stats
        stats = self.flattener.get_document_stats()
        self.assertEqual(stats['total_documents'], 1)
        self.assertGreater(stats['total_sections'], 50)  # Should have many sections
    
    def test_xml_validation(self):
        """Test that generated XML is valid"""
        # Create test document
        self.create_test_document("test.md", """# Test Document
        
        ## Section 1
        Content here
        
        As a user, I want to test so that validation works.
        
        FR-001: System must validate XML""")
        
        xml_content = self.flattener.flatten_all_documents()
        
        # Parse XML to ensure it's valid
        try:
            root = ET.fromstring(xml_content)
            self.assertEqual(root.tag, "project_documentation")
        except ET.ParseError as e:
            self.fail(f"Generated XML is invalid: {e}")
    
    def test_unicode_handling(self):
        """Test handling of unicode characters in documents"""
        unicode_content = """# Test with Unicode 🚀
        
        Special characters: àáâãäåæçèéêë
        Symbols: ★ ☆ ♠ ♣ ♥ ♦
        
        As a user, I want unicode support so that I can use special characters."""
        
        doc_path = self.create_test_document("unicode.md", unicode_content)
        element = self.flattener._process_document(doc_path, "documentation")
        
        self.assertIsNotNone(element)
        
        # Verify unicode is preserved
        sections = element.findall('section')
        self.assertGreater(len(sections), 0)
        
        # Check that content contains unicode
        content_elem = sections[0].find('content')
        self.assertIn('🚀', content_elem.text)
    
    def test_empty_directory(self):
        """Test behavior with empty directory"""
        empty_dir = Path(tempfile.mkdtemp())
        try:
            flattener = DocumentFlattener(empty_dir)
            xml_content = flattener.flatten_all_documents()
            
            # Should generate valid XML even with no documents
            root = ET.fromstring(xml_content)
            self.assertEqual(root.tag, "project_documentation")
            
            # Check metadata shows zero documents
            metadata = root.find("metadata")
            total_docs = metadata.find("total_documents")
            self.assertEqual(int(total_docs.text), 0)
            
        finally:
            shutil.rmtree(empty_dir)
    
    def test_malformed_markdown(self):
        """Test handling of malformed markdown"""
        malformed_content = """### This starts with h3
        
        # Then has h1
        
        #### Then h4
        
        As a user I want something (missing comma)
        
        FR-: Empty requirement ID"""
        
        doc_path = self.create_test_document("malformed.md", malformed_content)
        element = self.flattener._process_document(doc_path, "documentation")
        
        # Should still process without crashing
        self.assertIsNotNone(element)
        
        # Should have sections
        sections = element.findall('section')
        self.assertGreater(len(sections), 0)
    
    def test_cli_functionality(self):
        """Test CLI interface"""
        # Create test document
        self.create_test_document("test.md", "# CLI Test\nContent for CLI testing")
        
        # Test import and basic functionality
        from claude.commands.document_flattener import main
        
        # This test verifies the main function exists and can be imported
        # Full CLI testing would require subprocess calls
        self.assertTrue(callable(main))


class TestDocumentFlattenerIntegration(unittest.TestCase):
    """Integration tests with real-world document examples"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Clean up integration test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_realistic_prd_processing(self):
        """Test processing a realistic PRD document"""
        realistic_prd = """# E-Commerce Platform PRD
        
        ## Overview
        
        This document outlines requirements for a modern e-commerce platform.
        
        ## User Stories
        
        ### Customer Stories
        
        US-001: As a customer, I want to browse products so that I can find items to purchase.
        
        Acceptance Criteria:
        - Product catalog displays correctly
        - Search functionality works
        - Filtering options are available
        - Product details are comprehensive
        
        US-002: As a customer, I want to add items to cart so that I can purchase multiple products.
        
        Acceptance Criteria:
        - Items can be added to cart
        - Cart persists across sessions
        - Quantities can be modified
        - Items can be removed
        
        ### Admin Stories
        
        US-003: As an admin, I want to manage inventory so that product availability is accurate.
        
        ## Functional Requirements
        
        FR-001: The system must handle concurrent user sessions
        FR-002: Critical payment processing must be PCI compliant
        FR-003: Nice to have recommendation engine for related products
        
        ## Non-Functional Requirements
        
        NFR-001: System must support 1000 concurrent users
        NFR-002: Page load time must be under 2 seconds
        """
        
        prd_path = self.test_dir / "ecommerce.prd.md"
        prd_path.write_text(realistic_prd, encoding='utf-8')
        
        flattener = DocumentFlattener(self.test_dir)
        xml_content = flattener.flatten_all_documents()
        
        # Parse and validate
        root = ET.fromstring(xml_content)
        
        # Check that document was processed as PRD
        documents = root.find("documents")
        prd_doc = documents.find("document[@type='prd']")
        self.assertIsNotNone(prd_doc)
        
        # Check for user stories in sections
        user_stories_found = 0
        sections = prd_doc.findall("section")
        
        for section in sections:
            requirements = section.find("requirements")
            if requirements is not None:
                stories = requirements.findall("user_story")
                user_stories_found += len(stories)
        
        # Should find multiple user stories
        self.assertGreater(user_stories_found, 0)
        
    def test_realistic_architecture_processing(self):
        """Test processing a realistic architecture document"""
        realistic_arch = """# System Architecture
        
        ## Overview
        
        Microservices architecture for scalable e-commerce platform.
        
        ## Core Services
        
        ### UserService
        
        Manages user accounts, authentication, and profiles.
        
        Dependencies:
        - DatabaseService for persistence
        - EmailService for notifications
        
        ### ProductService
        
        Handles product catalog, inventory, and search functionality.
        
        ### OrderService
        
        Processes orders, payments, and fulfillment workflows.
        
        ## Data Architecture
        
        ### DatabaseService
        
        PostgreSQL-based data persistence layer with read replicas.
        
        ### CacheService
        
        Redis-based caching for improved performance.
        
        ## Integration Points
        
        - PaymentAPI: External payment processing
        - ShippingAPI: Third-party logistics integration
        - AnalyticsHandler: User behavior tracking
        """
        
        arch_path = self.test_dir / "system-architecture.md"
        arch_path.write_text(realistic_arch, encoding='utf-8')
        
        flattener = DocumentFlattener(self.test_dir)
        xml_content = flattener.flatten_all_documents()
        
        # Parse and validate
        root = ET.fromstring(xml_content)
        
        # Check that document was processed as architecture
        documents = root.find("documents")
        arch_doc = documents.find("document[@type='architecture']")
        self.assertIsNotNone(arch_doc)
        
        # Check for components in sections
        components_found = 0
        sections = arch_doc.findall("section")
        
        for section in sections:
            components = section.find("components")
            if components is not None:
                component_list = components.findall("component")
                components_found += len(component_list)
        
        # Should find multiple components
        self.assertGreater(components_found, 0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)