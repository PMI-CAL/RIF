#!/usr/bin/env python3
"""
Document Flattener Engine for RIF Framework

Converts project documentation into structured XML format optimized for AI consumption.
Inspired by BMAD-METHOD's codebase flattener approach.

Author: RIF-Implementer
Issue: #239
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Union, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentFlattener:
    """
    Converts project documentation into structured XML for AI consumption.
    Inspired by BMAD-METHOD's codebase flattener approach.
    """
    
    def __init__(self, docs_dir: Union[str, Path]):
        """
        Initialize the document flattener.
        
        Args:
            docs_dir: Path to the documentation directory
        """
        self.docs_dir = Path(docs_dir)
        self.document_registry = {}
        self.dependency_graph = {}
        self.flattened_content = None
        
        # Validate directory exists
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Documentation directory not found: {self.docs_dir}")
        
        logger.info(f"Initialized DocumentFlattener for: {self.docs_dir}")
        
    def flatten_all_documents(self) -> str:
        """
        Main entry point - flattens all documents into XML structure
        
        Returns:
            XML string representation of all flattened documents
        """
        logger.info("Starting document flattening process...")
        start_time = datetime.now()
        
        root = ET.Element("project_documentation")
        root.set("version", "1.0")
        root.set("timestamp", start_time.isoformat())
        root.set("source_dir", str(self.docs_dir))
        
        # Add metadata section
        metadata = ET.SubElement(root, "metadata")
        self._add_metadata(metadata)
        
        # Process each document type
        documents = ET.SubElement(root, "documents")
        
        # Process PRDs (Product Requirements Documents)
        prd_patterns = ["*.prd.md", "*.PRD.md", "*product-requirements*.md", "*prd*.md"]
        prds = self._process_document_type("prd", *prd_patterns)
        for prd in prds:
            documents.append(prd)
        
        # Process Architecture docs
        arch_patterns = ["*architecture*.md", "*arch*.md", "*system-design*.md", "*technical-design*.md"]
        arch_docs = self._process_document_type("architecture", *arch_patterns)
        for arch in arch_docs:
            documents.append(arch)
        
        # Process Design docs
        design_patterns = ["*design*.md", "*ui*.md", "*ux*.md", "*wireframe*.md", "*mockup*.md"]
        design_docs = self._process_document_type("design", *design_patterns)
        for design in design_docs:
            documents.append(design)
        
        # Process Requirements
        req_patterns = ["*requirements*.md", "*req*.md", "*specification*.md", "*spec*.md"]
        req_docs = self._process_document_type("requirements", *req_patterns)
        for req in req_docs:
            documents.append(req)
        
        # Process general documentation
        doc_patterns = ["*.md"]
        # Exclude already processed patterns and common non-doc files
        excluded_files = set()
        for doc_name in self.document_registry.keys():
            excluded_files.add(doc_name)
        
        # Also exclude common non-documentation files
        exclude_names = {"README.md", "CHANGELOG.md", "LICENSE.md", "CONTRIBUTING.md", "CLAUDE.md"}
        
        general_docs = self._process_document_type("documentation", *doc_patterns, exclude_files=excluded_files, exclude_names=exclude_names)
        for doc in general_docs:
            documents.append(doc)
        
        # Build dependency graph after all documents are processed
        self.dependency_graph = self._build_dependency_graph()
        
        # Add dependency graph
        dependencies = ET.SubElement(root, "dependencies")
        self._add_dependency_graph(dependencies)
        
        # Add cross-references
        references = ET.SubElement(root, "cross_references")
        self._add_cross_references(references)
        
        # Update metadata with final counts
        self._update_metadata(metadata)
        
        # Convert to string with pretty printing
        xml_string = self._prettify_xml(root)
        
        # Store result and log completion
        self.flattened_content = xml_string
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Document flattening completed in {duration:.2f} seconds")
        logger.info(f"Processed {len(self.document_registry)} documents")
        
        return xml_string
    
    def _process_document_type(self, doc_type: str, *patterns, exclude_patterns: Set[str] = None, exclude_files: Set[str] = None, exclude_names: Set[str] = None) -> List[ET.Element]:
        """
        Process all documents matching the given patterns
        
        Args:
            doc_type: Type of document (prd, architecture, design, requirements, documentation)
            *patterns: Glob patterns to match files
            exclude_patterns: Patterns to exclude from processing
            exclude_files: Specific file names to exclude
            exclude_names: Specific base names to exclude
            
        Returns:
            List of XML elements representing the documents
        """
        elements = []
        processed_files = set()
        exclude_patterns = exclude_patterns or set()
        exclude_files = exclude_files or set()
        exclude_names = exclude_names or set()
        
        for pattern in patterns:
            # Skip if this pattern should be excluded
            if pattern in exclude_patterns:
                continue
                
            for doc_path in self.docs_dir.rglob(pattern):
                # Skip if already processed or if matches exclude patterns
                if doc_path in processed_files:
                    continue
                
                # Skip if file name is in exclude list
                if doc_path.name in exclude_files or doc_path.name in exclude_names:
                    continue
                
                # Check if this file matches any exclude pattern
                should_exclude = False
                for exclude_pattern in exclude_patterns:
                    if doc_path.match(exclude_pattern):
                        should_exclude = True
                        break
                
                if should_exclude:
                    continue
                
                element = self._process_document(doc_path, doc_type)
                if element is not None:
                    elements.append(element)
                    processed_files.add(doc_path)
                    
        return elements
    
    def _process_document(self, doc_path: Path, doc_type: str) -> Optional[ET.Element]:
        """
        Process a single document into XML structure
        
        Args:
            doc_path: Path to the document file
            doc_type: Type of the document
            
        Returns:
            XML element representing the document or None if processing failed
        """
        if not doc_path.exists() or not doc_path.is_file():
            return None
        
        try:
            # Read content with proper encoding handling
            content = doc_path.read_text(encoding='utf-8', errors='ignore')
            
            # Skip empty files
            if not content.strip():
                logger.warning(f"Skipping empty file: {doc_path}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to read file {doc_path}: {e}")
            return None
            
        doc_element = ET.Element("document")
        doc_element.set("type", doc_type)
        doc_element.set("path", str(doc_path.relative_to(self.docs_dir)))
        # For PRD files, remove the .prd part from the name
        name = doc_path.stem
        if name.endswith('.prd'):
            name = name[:-4]  # Remove '.prd' suffix
        doc_element.set("name", name)
        doc_element.set("size", str(len(content)))
        doc_element.set("lines", str(len(content.splitlines())))
        
        # Extract sections
        sections = self._extract_sections(content)
        for section in sections:
            section_elem = ET.SubElement(doc_element, "section")
            section_elem.set("level", str(section['level']))
            section_elem.set("title", section['title'])
            
            # Add content
            content_elem = ET.SubElement(section_elem, "content")
            content_elem.text = section['content'].strip()
            
            # Extract specific elements based on document type
            if doc_type == "prd":
                self._extract_requirements(section, section_elem)
            elif doc_type == "architecture":
                self._extract_components(section, section_elem)
            elif doc_type == "design":
                self._extract_ui_elements(section, section_elem)
            elif doc_type == "requirements":
                self._extract_functional_requirements(section, section_elem)
        
        # Register document
        self.document_registry[doc_path.name] = {
            'type': doc_type,
            'path': str(doc_path),
            'sections': len(sections),
            'size': len(content),
            'lines': len(content.splitlines())
        }
        
        logger.debug(f"Processed document: {doc_path} ({len(sections)} sections)")
        return doc_element
    
    def _extract_sections(self, markdown_content: str) -> List[Dict]:
        """
        Extract hierarchical sections from markdown
        
        Args:
            markdown_content: Raw markdown content
            
        Returns:
            List of section dictionaries with level, title, and content
        """
        sections = []
        current_section = None
        lines = markdown_content.split('\n')
        
        for line_num, line in enumerate(lines):
            # Check if line is a heading
            if line.strip().startswith('#') and ' ' in line:
                # Save previous section
                if current_section:
                    sections.append(current_section)
                
                # Start new section
                heading_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
                if heading_match:
                    level = len(heading_match.group(1))
                    title = heading_match.group(2).strip()
                    current_section = {
                        'level': level,
                        'title': title,
                        'content': '',
                        'line_start': line_num + 1
                    }
            else:
                # Add line to current section content
                if current_section:
                    current_section['content'] += line + '\n'
        
        # Add last section
        if current_section:
            sections.append(current_section)
        
        # If no sections found, create a default section with all content
        if not sections and markdown_content.strip():
            sections.append({
                'level': 1,
                'title': 'Document Content',
                'content': markdown_content,
                'line_start': 1
            })
            
        return sections
    
    def _extract_requirements(self, section: Dict, parent: ET.Element):
        """
        Extract requirements, user stories, and acceptance criteria from PRD sections
        
        Args:
            section: Section dictionary containing content
            parent: Parent XML element to attach requirements to
        """
        requirements_elem = ET.SubElement(parent, "requirements")
        content = section['content'].lower()
        
        # Extract user stories
        user_stories = self._parse_user_stories(section['content'])
        for story in user_stories:
            story_elem = ET.SubElement(requirements_elem, "user_story")
            story_elem.set("id", story.get('id', 'unknown'))
            
            # Add description
            desc_elem = ET.SubElement(story_elem, "description")
            desc_elem.text = story.get('description', '')
            
            # Add acceptance criteria if present
            if 'criteria' in story and story['criteria']:
                criteria_elem = ET.SubElement(story_elem, "acceptance_criteria")
                for criterion in story['criteria']:
                    criterion_elem = ET.SubElement(criteria_elem, "criterion")
                    criterion_elem.text = criterion
        
        # Extract functional requirements
        func_reqs = self._parse_functional_requirements(section['content'])
        for req in func_reqs:
            req_elem = ET.SubElement(requirements_elem, "functional_requirement")
            req_elem.set("id", req.get('id', 'unknown'))
            req_elem.text = req.get('description', '')
            
            # Add priority if specified
            if 'priority' in req:
                req_elem.set("priority", req['priority'])
    
    def _extract_components(self, section: Dict, parent: ET.Element):
        """
        Extract system components from architecture sections
        
        Args:
            section: Section dictionary containing content
            parent: Parent XML element to attach components to
        """
        components_elem = ET.SubElement(parent, "components")
        
        # Look for component definitions
        components = self._parse_components(section['content'])
        for component in components:
            comp_elem = ET.SubElement(components_elem, "component")
            comp_elem.set("name", component.get('name', 'unknown'))
            comp_elem.set("type", component.get('type', 'unknown'))
            
            # Add description
            if 'description' in component:
                desc_elem = ET.SubElement(comp_elem, "description")
                desc_elem.text = component['description']
            
            # Add dependencies
            if 'dependencies' in component and component['dependencies']:
                deps_elem = ET.SubElement(comp_elem, "dependencies")
                for dep in component['dependencies']:
                    dep_elem = ET.SubElement(deps_elem, "dependency")
                    dep_elem.text = dep
    
    def _extract_ui_elements(self, section: Dict, parent: ET.Element):
        """
        Extract UI elements from design sections
        
        Args:
            section: Section dictionary containing content
            parent: Parent XML element to attach UI elements to
        """
        ui_elem = ET.SubElement(parent, "ui_elements")
        
        # Look for UI component definitions
        ui_components = self._parse_ui_components(section['content'])
        for component in ui_components:
            comp_elem = ET.SubElement(ui_elem, "ui_component")
            comp_elem.set("name", component.get('name', 'unknown'))
            comp_elem.set("type", component.get('type', 'unknown'))
            
            if 'description' in component:
                comp_elem.text = component['description']
    
    def _extract_functional_requirements(self, section: Dict, parent: ET.Element):
        """
        Extract functional requirements from requirements sections
        
        Args:
            section: Section dictionary containing content
            parent: Parent XML element to attach requirements to
        """
        reqs_elem = ET.SubElement(parent, "functional_requirements")
        
        # Parse requirements
        requirements = self._parse_functional_requirements(section['content'])
        for req in requirements:
            req_elem = ET.SubElement(reqs_elem, "requirement")
            req_elem.set("id", req.get('id', 'unknown'))
            req_elem.text = req.get('description', '')
            
            if 'priority' in req:
                req_elem.set("priority", req['priority'])
    
    def _parse_user_stories(self, content: str) -> List[Dict]:
        """
        Parse user stories from content
        
        Args:
            content: Text content to parse
            
        Returns:
            List of user story dictionaries
        """
        stories = []
        
        # Look for "As a ... I want ... So that ..." pattern - more flexible
        story_patterns = [
            # Pattern 1: User story with ID first (User Story US-001: As a...)
            re.compile(
                r'(?:user story|story)\s+([A-Z]+-\d+):\s*'
                r'as an?\s+([^,\n]+),?\s*'
                r'i want(?:\s+to)?\s+([^,\n]+?)(?:,|\s+so\s+that\s+([^.\n]+?))?[.\n]?',
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            ),
            # Pattern 2: User story with optional ID after colon
            re.compile(
                r'(?:user story|story):\s*(?:([A-Z]+-\d+):?)?\s*'
                r'as an?\s+([^,\n]+?),?\s*'
                r'i want(?:\s+to)?\s+([^,\n]+?)(?:,|\s+so\s+that\s+([^.\n]+?))?[.\n]?',
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            ),
            # Pattern 3: Direct "As a..." without "User story:" prefix
            re.compile(
                r'(?:^|\n)\s*(?:([A-Z]+-\d+):?\s*)?'
                r'as an?\s+([^,\n]+?),?\s*'
                r'i want(?:\s+to)?\s+([^,\n]+?)(?:,|\s+so\s+that\s+([^.\n]+?))?[.\n]?',
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            )
        ]
        
        for i, pattern in enumerate(story_patterns):
            for match in pattern.finditer(content):
                # Group extraction varies by pattern
                if i == 0:  # Pattern 1: User story US-001: As a...
                    story_id = match.group(1)
                    user_type = match.group(2).strip()
                    want = match.group(3).strip()
                    reason = match.group(4).strip() if match.group(4) else ""
                else:  # Pattern 2 and 3: Other formats
                    story_id = match.group(1) or f"US-{len(stories) + 1:03d}"
                    user_type = match.group(2).strip()
                    want = match.group(3).strip()
                    reason = match.group(4).strip() if match.group(4) else ""
                
                description = f"As a {user_type}, I want {want}"
                if reason:
                    description += f" so that {reason}"
                
                # Avoid duplicates
                if not any(s['description'] == description for s in stories):
                    stories.append({
                        'id': story_id,
                        'description': description,
                        'user_type': user_type,
                        'want': want,
                        'reason': reason
                    })
        
        # Look for acceptance criteria
        criteria_pattern = re.compile(
            r'(?:acceptance criteria|criteria):?\s*\n((?:\s*[-*]\s*[^\n]+\n?)+)',
            re.IGNORECASE | re.MULTILINE
        )
        
        criteria_matches = list(criteria_pattern.finditer(content))
        
        # Associate criteria with stories (simple heuristic: closest story before criteria)
        for i, story in enumerate(stories):
            # Find criteria that comes after this story in the text
            for criteria_match in criteria_matches:
                # Extract individual criteria items
                criteria_text = criteria_match.group(1)
                criteria_items = re.findall(r'[-*]\s*([^\n]+)', criteria_text)
                
                if criteria_items:
                    story['criteria'] = [item.strip() for item in criteria_items]
                    break
        
        return stories
    
    def _parse_functional_requirements(self, content: str) -> List[Dict]:
        """
        Parse functional requirements from content
        
        Args:
            content: Text content to parse
            
        Returns:
            List of requirement dictionaries
        """
        requirements = []
        
        # Look for numbered requirements (FR-001, REQ-001, etc.)
        req_pattern = re.compile(
            r'(?:FR-|REQ-|R-)(\d+):?\s*([^\n]+)',
            re.IGNORECASE | re.MULTILINE
        )
        
        for match in req_pattern.finditer(content):
            req_id = f"FR-{match.group(1)}"
            description = match.group(2).strip()
            
            # Look for priority indicators
            priority = "medium"  # default
            if any(word in description.lower() for word in ['critical', 'high priority', 'must have']):
                priority = "high"
            elif any(word in description.lower() for word in ['low priority', 'nice to have', 'optional']):
                priority = "low"
            
            requirements.append({
                'id': req_id,
                'description': description,
                'priority': priority
            })
        
        return requirements
    
    def _parse_components(self, content: str) -> List[Dict]:
        """
        Parse system components from architecture content
        
        Args:
            content: Text content to parse
            
        Returns:
            List of component dictionaries
        """
        components = []
        
        # Look for component definitions (various patterns)
        component_patterns = [
            # Pattern 1: Heading-based components (## or ### ComponentName)
            re.compile(r'##\s+([A-Z]\w*(?:Service|Component|Module|API|Handler|Controller|Engine|Manager|Gateway))\s*\n((?:[^\n#]+\n?)*)', re.MULTILINE | re.IGNORECASE),
            # Pattern 2: List-based components (- ComponentName: description)
            re.compile(r'^\s*[-*]\s*([A-Z]\w*(?:Service|Component|Module|API|Handler|Controller|Engine|Manager|Gateway)):\s*([^\n]+)', re.MULTILINE | re.IGNORECASE),
            # Pattern 3: Bullet points with dashes (* ComponentName - description)
            re.compile(r'^\s*[*]\s*([A-Z]\w*(?:Service|Component|Module|API|Handler|Controller|Engine|Manager|Gateway))\s*-\s*([^\n]+)', re.MULTILINE | re.IGNORECASE),
            # Pattern 4: Generic heading with common component names (## or ###)
            re.compile(r'##\s+([A-Z]\w+)\s*\n((?:[^\n#]+\n?)*)', re.MULTILINE),
        ]
        
        processed_names = set()
        
        for pattern in component_patterns:
            for match in pattern.finditer(content):
                name = match.group(1).strip()
                description = match.group(2).strip() if match.group(2) else ""
                
                # Skip if already processed
                if name in processed_names:
                    continue
                
                # Only include if it looks like a component name
                if (len(name) > 2 and 
                    any(keyword in name.lower() for keyword in 
                        ['service', 'component', 'module', 'api', 'handler', 
                         'controller', 'engine', 'manager', 'gateway', 'auth', 
                         'user', 'data', 'database', 'cache', 'payment', 'order'])):
                    
                    # Determine component type
                    comp_type = "component"  # default
                    name_lower = name.lower()
                    if "service" in name_lower:
                        comp_type = "service"
                    elif "api" in name_lower:
                        comp_type = "api"
                    elif "handler" in name_lower:
                        comp_type = "handler"
                    elif "controller" in name_lower:
                        comp_type = "controller"
                    elif "module" in name_lower:
                        comp_type = "module"
                    elif "engine" in name_lower:
                        comp_type = "engine"
                    elif "manager" in name_lower:
                        comp_type = "manager"
                    elif "gateway" in name_lower:
                        comp_type = "gateway"
                    
                    components.append({
                        'name': name,
                        'type': comp_type,
                        'description': description
                    })
                    
                    processed_names.add(name)
        
        return components
    
    def _parse_ui_components(self, content: str) -> List[Dict]:
        """
        Parse UI components from design content
        
        Args:
            content: Text content to parse
            
        Returns:
            List of UI component dictionaries
        """
        components = []
        
        # Look for UI component patterns
        ui_patterns = [
            # Pattern 1: Heading-based UI components (## or ### ComponentName)
            re.compile(r'##\s+([A-Z]\w*(?:Button|Form|Modal|Dialog|Panel|Widget|Component|Input|Menu|Navigation|Header|Footer))\s*\n((?:[^\n#]+\n?)*)', re.MULTILINE | re.IGNORECASE),
            # Pattern 2: List-based UI components (- ComponentName: description)
            re.compile(r'^\s*[-*]\s*([A-Z]\w*(?:Button|Form|Modal|Dialog|Panel|Widget|Component|Input|Menu|Navigation|Header|Footer)):\s*([^\n]+)', re.MULTILINE | re.IGNORECASE),
            # Pattern 3: Bullet points with dashes (* ComponentName - description)
            re.compile(r'^\s*[*]\s*([A-Z]\w*(?:Button|Form|Modal|Dialog|Panel|Widget|Component|Input|Menu|Navigation|Header|Footer))\s*-\s*([^\n]+)', re.MULTILINE | re.IGNORECASE),
            # Pattern 4: Generic heading with common UI names (## or ###)
            re.compile(r'##\s+([A-Z]\w+)\s*\n((?:[^\n#]+\n?)*)', re.MULTILINE),
        ]
        
        processed_names = set()
        
        for pattern in ui_patterns:
            for match in pattern.finditer(content):
                name = match.group(1).strip()
                description = match.group(2).strip() if match.group(2) else ""
                
                # Skip if already processed
                if name in processed_names:
                    continue
                
                # Only include if it looks like a UI component name
                if (len(name) > 2 and 
                    any(keyword in name.lower() for keyword in 
                        ['button', 'form', 'modal', 'dialog', 'panel', 'widget', 
                         'component', 'input', 'menu', 'navigation', 'header', 
                         'footer', 'login', 'signup', 'search', 'filter'])):
                    
                    # Determine UI component type
                    ui_type = "component"  # default
                    name_lower = name.lower()
                    if "button" in name_lower:
                        ui_type = "button"
                    elif "form" in name_lower:
                        ui_type = "form"
                    elif "modal" in name_lower:
                        ui_type = "modal"
                    elif "dialog" in name_lower:
                        ui_type = "dialog"
                    elif "panel" in name_lower:
                        ui_type = "panel"
                    elif "widget" in name_lower:
                        ui_type = "widget"
                    elif "input" in name_lower:
                        ui_type = "input"
                    elif "menu" in name_lower:
                        ui_type = "menu"
                    elif "navigation" in name_lower:
                        ui_type = "navigation"
                    elif "header" in name_lower:
                        ui_type = "header"
                    elif "footer" in name_lower:
                        ui_type = "footer"
                    
                    components.append({
                        'name': name,
                        'type': ui_type,
                        'description': description
                    })
                    
                    processed_names.add(name)
        
        return components
    
    def _build_dependency_graph(self) -> Dict:
        """
        Build a graph of document dependencies and relationships
        
        Returns:
            Dictionary representing the dependency graph
        """
        graph = {}
        
        for doc_name, doc_info in self.document_registry.items():
            dependencies = []
            
            # Read document content
            try:
                doc_content = Path(doc_info['path']).read_text(encoding='utf-8', errors='ignore')
            except Exception as e:
                logger.warning(f"Could not read {doc_info['path']} for dependency analysis: {e}")
                continue
            
            # Look for references to other documents
            for other_doc in self.document_registry:
                if other_doc != doc_name:
                    # Check for direct filename references
                    if other_doc in doc_content:
                        dependencies.append(other_doc)
                    
                    # Check for references without extension
                    doc_stem = Path(other_doc).stem
                    if doc_stem in doc_content and doc_stem != Path(doc_name).stem:
                        dependencies.append(other_doc)
            
            # Look for section references
            section_refs = self._extract_section_references(doc_content)
            dependencies.extend(section_refs)
            
            graph[doc_name] = {
                'depends_on': list(set(dependencies)),  # Remove duplicates
                'type': doc_info['type'],
                'priority': self._calculate_priority(doc_info['type']),
                'size': doc_info['size']
            }
        
        return graph
    
    def _extract_section_references(self, content: str) -> List[str]:
        """
        Extract section references from content
        
        Args:
            content: Document content to analyze
            
        Returns:
            List of referenced sections
        """
        references = []
        
        # Look for various reference patterns
        ref_patterns = [
            re.compile(r'see\s+(?:section\s+)?([^.\n]+)', re.IGNORECASE),
            re.compile(r'refer to\s+([^.\n]+)', re.IGNORECASE),
            re.compile(r'as described in\s+([^.\n]+)', re.IGNORECASE),
            re.compile(r'detailed in\s+([^.\n]+)', re.IGNORECASE)
        ]
        
        for pattern in ref_patterns:
            for match in pattern.finditer(content):
                ref = match.group(1).strip()
                if len(ref) < 100:  # Reasonable reference length
                    references.append(ref)
        
        return references
    
    def _calculate_priority(self, doc_type: str) -> int:
        """
        Calculate priority score for document type
        
        Args:
            doc_type: Type of document
            
        Returns:
            Priority score (higher = more important)
        """
        priorities = {
            'prd': 5,           # Product requirements are highest priority
            'requirements': 4,   # Technical requirements are important
            'architecture': 3,   # Architecture docs are medium-high priority
            'design': 2,        # Design docs are medium priority
            'documentation': 1   # General docs are lowest priority
        }
        
        return priorities.get(doc_type, 1)
    
    def _add_metadata(self, metadata_elem: ET.Element):
        """
        Add metadata section to XML
        
        Args:
            metadata_elem: XML element to add metadata to
        """
        # Project information
        project_elem = ET.SubElement(metadata_elem, "project_name")
        project_elem.text = self.docs_dir.name
        
        # Generation timestamp
        generated_elem = ET.SubElement(metadata_elem, "generated_at")
        generated_elem.text = datetime.now().isoformat()
        
        # Source directory
        source_elem = ET.SubElement(metadata_elem, "source_directory")
        source_elem.text = str(self.docs_dir)
        
        # Statistics will be updated later in _update_metadata
        
    def _update_metadata(self, metadata_elem: ET.Element):
        """
        Update metadata with final statistics
        
        Args:
            metadata_elem: XML metadata element to update
        """
        # Document counts
        total_docs = ET.SubElement(metadata_elem, "total_documents")
        total_docs.text = str(len(self.document_registry))
        
        # Count by document type
        type_counts = {}
        for doc_info in self.document_registry.values():
            doc_type = doc_info['type']
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        type_counts_elem = ET.SubElement(metadata_elem, "document_types")
        for doc_type, count in type_counts.items():
            type_elem = ET.SubElement(type_counts_elem, "type")
            type_elem.set("name", doc_type)
            type_elem.text = str(count)
        
        # Total content statistics
        total_size = sum(doc['size'] for doc in self.document_registry.values())
        total_lines = sum(doc['lines'] for doc in self.document_registry.values())
        total_sections = sum(doc['sections'] for doc in self.document_registry.values())
        
        size_elem = ET.SubElement(metadata_elem, "total_content_size")
        size_elem.text = str(total_size)
        
        lines_elem = ET.SubElement(metadata_elem, "total_lines")
        lines_elem.text = str(total_lines)
        
        sections_elem = ET.SubElement(metadata_elem, "total_sections")
        sections_elem.text = str(total_sections)
    
    def _add_dependency_graph(self, dependencies_elem: ET.Element):
        """
        Add dependency graph to XML
        
        Args:
            dependencies_elem: XML element to add dependencies to
        """
        for doc_name, doc_data in self.dependency_graph.items():
            for dependency in doc_data['depends_on']:
                dep_elem = ET.SubElement(dependencies_elem, "dependency")
                dep_elem.set("from", doc_name)
                dep_elem.set("to", dependency)
                dep_elem.set("type", "reference")
    
    def _add_cross_references(self, references_elem: ET.Element):
        """
        Add cross-references to XML
        
        Args:
            references_elem: XML element to add cross-references to
        """
        # Build cross-references from dependency graph
        for doc_name, doc_data in self.dependency_graph.items():
            doc_type = doc_data['type']
            
            for dependency in doc_data['depends_on']:
                if dependency in self.document_registry:
                    dep_type = self.document_registry[dependency]['type']
                    
                    ref_elem = ET.SubElement(references_elem, "reference")
                    ref_elem.set("from", f"{doc_type}#{Path(doc_name).stem}")
                    ref_elem.set("to", f"{dep_type}#{Path(dependency).stem}")
                    ref_elem.set("strength", "medium")  # Could be enhanced with analysis
    
    def _prettify_xml(self, elem: ET.Element) -> str:
        """
        Create a pretty-printed XML string
        
        Args:
            elem: Root XML element
            
        Returns:
            Formatted XML string
        """
        # Use minidom for pretty printing
        from xml.dom import minidom
        
        rough_string = ET.tostring(elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    
    def save_flattened_xml(self, output_path: Union[str, Path]) -> Path:
        """
        Save the flattened XML to a file
        
        Args:
            output_path: Path where to save the XML file
            
        Returns:
            Path object of the saved file
        """
        if not self.flattened_content:
            raise RuntimeError("No flattened content available. Call flatten_all_documents() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_path.write_text(self.flattened_content, encoding='utf-8')
        logger.info(f"Flattened XML saved to: {output_path}")
        
        return output_path
    
    def get_document_stats(self) -> Dict:
        """
        Get statistics about processed documents
        
        Returns:
            Dictionary containing document statistics
        """
        if not self.document_registry:
            return {}
        
        stats = {
            'total_documents': len(self.document_registry),
            'total_size': sum(doc['size'] for doc in self.document_registry.values()),
            'total_lines': sum(doc['lines'] for doc in self.document_registry.values()),
            'total_sections': sum(doc['sections'] for doc in self.document_registry.values()),
            'by_type': {}
        }
        
        # Group by type
        for doc_info in self.document_registry.values():
            doc_type = doc_info['type']
            if doc_type not in stats['by_type']:
                stats['by_type'][doc_type] = {
                    'count': 0,
                    'size': 0,
                    'lines': 0,
                    'sections': 0
                }
            
            stats['by_type'][doc_type]['count'] += 1
            stats['by_type'][doc_type]['size'] += doc_info['size']
            stats['by_type'][doc_type]['lines'] += doc_info['lines']
            stats['by_type'][doc_type]['sections'] += doc_info['sections']
        
        return stats


def main():
    """Main function for CLI usage"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Flatten project documentation to XML")
    parser.add_argument("docs_dir", help="Path to documentation directory")
    parser.add_argument("-o", "--output", help="Output XML file path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        flattener = DocumentFlattener(args.docs_dir)
        xml_content = flattener.flatten_all_documents()
        
        if args.output:
            output_path = flattener.save_flattened_xml(args.output)
            print(f"XML saved to: {output_path}")
        else:
            print(xml_content)
        
        # Print statistics
        stats = flattener.get_document_stats()
        print(f"\nProcessed {stats['total_documents']} documents", file=sys.stderr)
        print(f"Total content: {stats['total_size']:,} characters, {stats['total_lines']:,} lines", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()