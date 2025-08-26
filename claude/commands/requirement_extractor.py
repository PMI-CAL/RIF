#!/usr/bin/env python3
"""
Requirement Extraction Engine for RIF Framework

Extracts and categorizes all project requirements, user stories, epics, and acceptance criteria
from flattened documentation XML for granular issue generation. Builds on the document_flattener
module and provides sophisticated requirement parsing and classification.

Author: RIF-Implementer
Issue: #240
Dependencies: Issue #239 (document_flattener.py)
"""

import re
import xml.etree.ElementTree as ET
import json
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RequirementType(Enum):
    """Types of requirements that can be extracted"""
    EPIC = "epic"
    USER_STORY = "user_story"
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    TECHNICAL = "technical"
    ACCEPTANCE_CRITERIA = "acceptance_criteria"
    CONSTRAINT = "constraint"
    BUSINESS = "business"


class Priority(Enum):
    """Priority levels for requirements"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Complexity(Enum):
    """Complexity levels for requirements"""
    VERY_HIGH = "very-high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Requirement:
    """Structured requirement with metadata"""
    id: str
    type: RequirementType
    title: str
    description: str
    source_document: str
    source_section: str
    priority: Priority
    complexity: Complexity
    dependencies: List[str]
    acceptance_criteria: List[str]
    estimated_tokens: int
    tags: List[str]
    actor: Optional[str] = None  # For user stories
    action: Optional[str] = None  # For user stories
    benefit: Optional[str] = None  # For user stories
    estimated_points: Optional[int] = None  # For user stories


class RequirementPatterns:
    """Patterns for identifying different types of requirements"""
    
    EPIC_PATTERNS = [
        r'Epic\s*\d+:?\s*(.+)',
        r'Feature:\s*(.+)',
        r'Module:\s*(.+)',
        r'Component:\s*(.+)',
        r'Phase\s*\d+:\s*(.+)',
        r'Theme:\s*(.+)'
    ]
    
    USER_STORY_PATTERNS = [
        r'As\s+(?:a|an)\s+(.+?),?\s*I\s+want\s+(?:to\s+)?(.+?)(?:\s+so\s+that\s+(.+?))?(?:[.\n]|$)',
        r'User\s+Story:?\s*(.+)',
        r'Story\s+\d+:?\s*(.+)',
        r'Scenario:?\s*(.+)',
        r'US-?\d+:?\s*(.+)'
    ]
    
    FUNCTIONAL_PATTERNS = [
        r'The\s+system\s+(shall|must|should|will)\s+(.+)',
        r'FR-?\d+:?\s*(.+)',
        r'Functional\s+Requirement:?\s*(.+)',
        r'Function:?\s*(.+)',
        r'Capability:?\s*(.+)',
        r'Feature:?\s*(.+)'
    ]
    
    NON_FUNCTIONAL_PATTERNS = [
        r'Performance:?\s*(.+)',
        r'Security:?\s*(.+)',
        r'Scalability:?\s*(.+)',
        r'Reliability:?\s*(.+)',
        r'Availability:?\s*(.+)',
        r'Usability:?\s*(.+)',
        r'Maintainability:?\s*(.+)',
        r'Portability:?\s*(.+)',
        r'NFR-?\d+:?\s*(.+)',
        r'Quality:?\s*(.+)'
    ]
    
    TECHNICAL_PATTERNS = [
        r'Technical\s+Requirement:?\s*(.+)',
        r'Architecture:?\s*(.+)',
        r'Technology:?\s*(.+)',
        r'Platform:?\s*(.+)',
        r'Integration:?\s*(.+)',
        r'TR-?\d+:?\s*(.+)',
        r'API:?\s*(.+)',
        r'Database:?\s*(.+)'
    ]
    
    ACCEPTANCE_CRITERIA_PATTERNS = [
        r'Given\s+(.+?)\s+When\s+(.+?)\s+Then\s+(.+)',
        r'AC-?\d+:?\s*(.+)',
        r'Acceptance:?\s*(.+)',
        r'Success\s+Criteria:?\s*(.+)',
        r'Criteria:?\s*(.+)',
        r'Validation:?\s*(.+)'
    ]
    
    CONSTRAINT_PATTERNS = [
        r'Constraint:?\s*(.+)',
        r'Limitation:?\s*(.+)',
        r'Restriction:?\s*(.+)',
        r'Assumption:?\s*(.+)',
        r'Dependency:?\s*(.+)',
        r'Prerequisite:?\s*(.+)'
    ]
    
    BUSINESS_PATTERNS = [
        r'Business\s+Rule:?\s*(.+)',
        r'Policy:?\s*(.+)',
        r'Regulation:?\s*(.+)',
        r'Compliance:?\s*(.+)',
        r'Objective:?\s*(.+)',
        r'Goal:?\s*(.+)',
        r'KPI:?\s*(.+)',
        r'Metric:?\s*(.+)'
    ]


class RequirementExtractor:
    """
    Extracts and categorizes requirements from flattened documentation XML
    """
    
    def __init__(self, flattened_xml: str = None, xml_file_path: str = None):
        """
        Initialize the requirement extractor
        
        Args:
            flattened_xml: XML string from document flattener
            xml_file_path: Path to XML file (alternative to flattened_xml)
        """
        if flattened_xml:
            self.root = ET.fromstring(flattened_xml)
        elif xml_file_path and Path(xml_file_path).exists():
            tree = ET.parse(xml_file_path)
            self.root = tree.getroot()
        else:
            raise ValueError("Either flattened_xml or valid xml_file_path must be provided")
        
        self.requirements: List[Requirement] = []
        self.epics: List[Dict] = []
        self.stories: List[Dict] = []
        self.requirement_counter = 0
        self.epic_counter = 0
        self.story_counter = 0
        
        # Pattern matchers
        self.patterns = RequirementPatterns()
        
        logger.info("RequirementExtractor initialized")
    
    def _requirement_to_dict(self, req: Requirement) -> Dict:
        """Convert a Requirement object to a dictionary with serializable values"""
        req_dict = asdict(req)
        # Convert enums to string values
        req_dict['type'] = req.type.value
        req_dict['priority'] = req.priority.value
        req_dict['complexity'] = req.complexity.value
        return req_dict
    
    def extract_all_requirements(self) -> Dict:
        """
        Main extraction method - processes all documents and returns structured requirements
        
        Returns:
            Dictionary containing all extracted requirements organized by type
        """
        logger.info("Starting requirement extraction process...")
        start_time = datetime.now()
        
        # Process different document types
        self._extract_from_prd_documents()
        self._extract_from_architecture_documents()
        self._extract_from_design_documents()
        self._extract_from_requirements_documents()
        
        # Build requirement hierarchy and relationships
        hierarchy = self._build_requirement_hierarchy()
        
        # Analyze dependencies between requirements
        self._analyze_dependencies()
        
        # Estimate complexity and token counts
        self._estimate_complexity_and_tokens()
        
        # Prioritize requirements
        self._prioritize_requirements()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Requirement extraction completed in {duration:.2f} seconds")
        logger.info(f"Extracted {len(self.requirements)} requirements, {len(self.epics)} epics, {len(self.stories)} user stories")
        
        return {
            'epics': self.epics,
            'stories': self.stories,
            'requirements': [self._requirement_to_dict(req) for req in self.requirements],
            'hierarchy': hierarchy,
            'statistics': self._generate_statistics(),
            'extraction_metadata': {
                'extracted_at': datetime.now().isoformat(),
                'duration_seconds': duration,
                'total_items': len(self.requirements) + len(self.epics) + len(self.stories)
            }
        }
    
    def _extract_from_prd_documents(self):
        """Extract requirements from PRD (Product Requirements Document) sections"""
        logger.info("Extracting requirements from PRD documents...")
        
        prd_docs = self.root.findall(".//document[@type='prd']")
        
        for doc in prd_docs:
            doc_name = doc.get('name', 'unknown')
            logger.debug(f"Processing PRD document: {doc_name}")
            
            sections = doc.findall(".//section")
            for section in sections:
                section_title = section.get('title', 'Unknown Section')
                content_elem = section.find('content')
                content = content_elem.text if content_elem is not None and content_elem.text else ""
                
                # Extract epics from section title first, then content
                self._extract_epics_from_title(section_title, content, doc_name, section_title)
                if content:
                    self._extract_epics(content, doc_name, section_title)
                
                # Extract user stories from this section
                self._extract_user_stories(content, doc_name, section_title)
                
                # Extract functional requirements
                self._extract_functional_requirements(content, doc_name, section_title)
                
                # Extract business requirements
                self._extract_business_requirements(content, doc_name, section_title)
                
                # Extract constraints
                self._extract_constraints(content, doc_name, section_title)
                
                # Extract acceptance criteria (often in PRDs)
                self._extract_acceptance_criteria(content, doc_name, section_title)
    
    def _extract_from_architecture_documents(self):
        """Extract requirements from architecture documents"""
        logger.info("Extracting requirements from architecture documents...")
        
        arch_docs = self.root.findall(".//document[@type='architecture']")
        
        for doc in arch_docs:
            doc_name = doc.get('name', 'unknown')
            logger.debug(f"Processing architecture document: {doc_name}")
            
            sections = doc.findall(".//section")
            for section in sections:
                section_title = section.get('title', 'Unknown Section')
                content_elem = section.find('content')
                if content_elem is None or not content_elem.text:
                    continue
                
                content = content_elem.text
                
                # Extract technical requirements
                self._extract_technical_requirements(content, doc_name, section_title)
                
                # Extract non-functional requirements (common in architecture docs)
                self._extract_non_functional_requirements(content, doc_name, section_title)
                
                # Extract constraints (architectural constraints)
                self._extract_constraints(content, doc_name, section_title)
    
    def _extract_from_design_documents(self):
        """Extract requirements from design documents"""
        logger.info("Extracting requirements from design documents...")
        
        design_docs = self.root.findall(".//document[@type='design']")
        
        for doc in design_docs:
            doc_name = doc.get('name', 'unknown')
            logger.debug(f"Processing design document: {doc_name}")
            
            sections = doc.findall(".//section")
            for section in sections:
                section_title = section.get('title', 'Unknown Section')
                content_elem = section.find('content')
                if content_elem is None or not content_elem.text:
                    continue
                
                content = content_elem.text
                
                # Extract user stories (often in design docs)
                self._extract_user_stories(content, doc_name, section_title)
                
                # Extract functional requirements
                self._extract_functional_requirements(content, doc_name, section_title)
                
                # Extract non-functional requirements (UI/UX related)
                self._extract_non_functional_requirements(content, doc_name, section_title)
    
    def _extract_from_requirements_documents(self):
        """Extract requirements from dedicated requirements documents"""
        logger.info("Extracting requirements from requirements documents...")
        
        req_docs = self.root.findall(".//document[@type='requirements']")
        
        for doc in req_docs:
            doc_name = doc.get('name', 'unknown')
            logger.debug(f"Processing requirements document: {doc_name}")
            
            sections = doc.findall(".//section")
            for section in sections:
                section_title = section.get('title', 'Unknown Section')
                content_elem = section.find('content')
                if content_elem is None or not content_elem.text:
                    continue
                
                content = content_elem.text
                
                # Extract all types of requirements from dedicated requirements docs
                self._extract_epics(content, doc_name, section_title)
                self._extract_user_stories(content, doc_name, section_title)
                self._extract_functional_requirements(content, doc_name, section_title)
                self._extract_non_functional_requirements(content, doc_name, section_title)
                self._extract_technical_requirements(content, doc_name, section_title)
                self._extract_business_requirements(content, doc_name, section_title)
                self._extract_constraints(content, doc_name, section_title)
                self._extract_acceptance_criteria(content, doc_name, section_title)
    
    def _extract_epics(self, content: str, source_doc: str, section_title: str):
        """Extract epics from content"""
        for pattern in self.patterns.EPIC_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                epic_title = match.group(1).strip()
                
                # Skip if too short or generic
                if len(epic_title) < 5 or epic_title.lower() in ['overview', 'introduction', 'background']:
                    continue
                
                self.epic_counter += 1
                epic_id = f"EPIC-{self.epic_counter:03d}"
                
                # Extract stories within this epic (look in surrounding context)
                epic_stories = self._find_related_stories(content, match.start(), match.end())
                
                # Extract description (content following the epic title)
                description = self._extract_following_content(content, match.end(), 500)
                
                epic = {
                    'id': epic_id,
                    'type': RequirementType.EPIC.value,
                    'title': epic_title,
                    'description': description,
                    'source_document': source_doc,
                    'source_section': section_title,
                    'stories': [story['id'] for story in epic_stories],
                    'sub_epics': [],
                    'estimated_effort': self._estimate_epic_effort(epic_title, description),
                    'complexity': self._estimate_complexity_level(epic_title, description).value,
                    'tags': self._extract_tags(epic_title + " " + description)
                }
                
                self.epics.append(epic)
                
                # Add the related stories to our stories list
                self.stories.extend(epic_stories)
    
    def _extract_epics_from_title(self, title: str, content: str, source_doc: str, section_title: str):
        """Extract epics from section titles"""
        for pattern in self.patterns.EPIC_PATTERNS:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                epic_title = match.group(1).strip()
                
                # Skip if too short or generic
                if len(epic_title) < 5 or epic_title.lower() in ['overview', 'introduction', 'background']:
                    continue
                
                # Skip if already found this epic
                if any(epic['title'] == epic_title for epic in self.epics):
                    continue
                
                self.epic_counter += 1
                epic_id = f"EPIC-{self.epic_counter:03d}"
                
                # Use section content as epic description
                description = content[:500] if content else epic_title
                
                # Extract stories within this epic (look in section content)
                epic_stories = self._find_related_stories(content, 0, len(content)) if content else []
                
                epic = {
                    'id': epic_id,
                    'type': RequirementType.EPIC.value,
                    'title': epic_title,
                    'description': description,
                    'source_document': source_doc,
                    'source_section': section_title,
                    'stories': [story['id'] for story in epic_stories],
                    'sub_epics': [],
                    'estimated_effort': self._estimate_epic_effort(epic_title, description),
                    'complexity': self._estimate_complexity_level(epic_title, description).value,
                    'tags': self._extract_tags(epic_title + " " + description)
                }
                
                self.epics.append(epic)
                
                # Add the related stories to our stories list (avoid duplicates)
                for story in epic_stories:
                    if not any(s['id'] == story['id'] for s in self.stories):
                        self.stories.append(story)
                
                break  # Only match first pattern
    
    def _extract_user_stories(self, content: str, source_doc: str, section_title: str):
        """Extract user stories from content"""
        found_descriptions = set()  # Track descriptions to avoid duplicates
        
        for pattern in self.patterns.USER_STORY_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches:
                # Handle different pattern formats
                if "As" in pattern:
                    # Full user story format: As a... I want... so that...
                    actor = match.group(1).strip() if match.group(1) else "user"
                    action = match.group(2).strip() if match.group(2) else ""
                    benefit = match.group(3).strip() if match.group(3) else ""
                    
                    description = f"As a {actor}, I want {action}"
                    if benefit:
                        description += f" so that {benefit}"
                    
                    title = self._generate_story_title(actor, action)
                else:
                    # Other formats - extract just the story part, not the whole match
                    raw_description = match.group(1).strip()
                    
                    # If it contains a full user story format, parse it
                    actor, action, benefit = self._parse_story_components(raw_description)
                    
                    if action:  # Valid story components found
                        description = f"As a {actor}, I want {action}"
                        if benefit:
                            description += f" so that {benefit}"
                        title = self._generate_story_title(actor, action)
                    else:
                        # Fall back to using the raw description
                        description = raw_description
                        title = description[:50] + "..." if len(description) > 50 else description
                        actor, action, benefit = "user", description, ""
                
                # Skip duplicates and invalid stories
                if description in found_descriptions or len(description) < 10:
                    continue
                
                found_descriptions.add(description)
                
                self.story_counter += 1
                story_id = f"US-{self.story_counter:03d}"
                
                # Extract acceptance criteria for this story
                criteria = self._find_related_acceptance_criteria(content, match.start(), match.end())
                
                story = {
                    'id': story_id,
                    'type': RequirementType.USER_STORY.value,
                    'title': title,
                    'actor': actor,
                    'action': action,
                    'benefit': benefit,
                    'description': description,
                    'source_document': source_doc,
                    'source_section': section_title,
                    'acceptance_criteria': criteria,
                    'estimated_points': self._estimate_story_points(description, criteria),
                    'complexity': self._estimate_complexity_level(description).value,
                    'priority': self._determine_priority(description).value,
                    'tags': self._extract_tags(description)
                }
                
                self.stories.append(story)
    
    def _extract_functional_requirements(self, content: str, source_doc: str, section_title: str):
        """Extract functional requirements from content"""
        self._extract_requirements_by_patterns(
            content, source_doc, section_title,
            self.patterns.FUNCTIONAL_PATTERNS,
            RequirementType.FUNCTIONAL,
            "FR"
        )
    
    def _extract_non_functional_requirements(self, content: str, source_doc: str, section_title: str):
        """Extract non-functional requirements from content"""
        self._extract_requirements_by_patterns(
            content, source_doc, section_title,
            self.patterns.NON_FUNCTIONAL_PATTERNS,
            RequirementType.NON_FUNCTIONAL,
            "NFR"
        )
    
    def _extract_technical_requirements(self, content: str, source_doc: str, section_title: str):
        """Extract technical requirements from content"""
        self._extract_requirements_by_patterns(
            content, source_doc, section_title,
            self.patterns.TECHNICAL_PATTERNS,
            RequirementType.TECHNICAL,
            "TR"
        )
    
    def _extract_business_requirements(self, content: str, source_doc: str, section_title: str):
        """Extract business requirements from content"""
        self._extract_requirements_by_patterns(
            content, source_doc, section_title,
            self.patterns.BUSINESS_PATTERNS,
            RequirementType.BUSINESS,
            "BR"
        )
    
    def _extract_constraints(self, content: str, source_doc: str, section_title: str):
        """Extract constraints from content"""
        self._extract_requirements_by_patterns(
            content, source_doc, section_title,
            self.patterns.CONSTRAINT_PATTERNS,
            RequirementType.CONSTRAINT,
            "CON"
        )
    
    def _extract_acceptance_criteria(self, content: str, source_doc: str, section_title: str):
        """Extract acceptance criteria from content"""
        self._extract_requirements_by_patterns(
            content, source_doc, section_title,
            self.patterns.ACCEPTANCE_CRITERIA_PATTERNS,
            RequirementType.ACCEPTANCE_CRITERIA,
            "AC"
        )
    
    def _extract_requirements_by_patterns(self, content: str, source_doc: str, section_title: str, 
                                       patterns: List[str], req_type: RequirementType, id_prefix: str):
        """Generic method to extract requirements using patterns"""
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                description = match.group(1).strip() if match.groups() else match.group(0).strip()
                
                # Skip if too short or generic
                if len(description) < 5:
                    continue
                
                # Skip duplicates
                if any(req.description == description for req in self.requirements):
                    continue
                
                self.requirement_counter += 1
                req_id = f"{id_prefix}-{self.requirement_counter:03d}"
                
                # Generate title from description
                title = description[:60] + "..." if len(description) > 60 else description
                title = re.sub(r'^(The\s+system\s+(?:shall|must|should|will)\s+)', '', title, flags=re.IGNORECASE)
                
                # Extract related acceptance criteria
                criteria = self._find_related_acceptance_criteria(content, match.start(), match.end())
                
                requirement = Requirement(
                    id=req_id,
                    type=req_type,
                    title=title,
                    description=description,
                    source_document=source_doc,
                    source_section=section_title,
                    priority=self._determine_priority(description),
                    complexity=self._estimate_complexity_level(description),
                    dependencies=[],  # Will be populated later
                    acceptance_criteria=criteria,
                    estimated_tokens=0,  # Will be calculated later
                    tags=self._extract_tags(description)
                )
                
                self.requirements.append(requirement)
    
    def _find_related_stories(self, content: str, start_pos: int, end_pos: int) -> List[Dict]:
        """Find user stories related to an epic within the content"""
        # Look for stories in the 1000 characters following the epic
        search_content = content[end_pos:end_pos + 1000]
        related_stories = []
        
        for pattern in self.patterns.USER_STORY_PATTERNS:
            matches = re.finditer(pattern, search_content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if "As" in pattern:
                    actor = match.group(1).strip() if match.group(1) else "user"
                    action = match.group(2).strip() if match.group(2) else ""
                    benefit = match.group(3).strip() if match.group(3) else ""
                    
                    description = f"As a {actor}, I want {action}"
                    if benefit:
                        description += f" so that {benefit}"
                    
                    title = self._generate_story_title(actor, action)
                else:
                    description = match.group(1).strip()
                    title = description[:50] + "..." if len(description) > 50 else description
                    actor, action, benefit = self._parse_story_components(description)
                
                if len(description) >= 10:
                    self.story_counter += 1
                    story_id = f"US-{self.story_counter:03d}"
                    
                    related_stories.append({
                        'id': story_id,
                        'type': RequirementType.USER_STORY.value,
                        'title': title,
                        'actor': actor,
                        'action': action,
                        'benefit': benefit,
                        'description': description,
                        'acceptance_criteria': [],
                        'estimated_points': self._estimate_story_points(description, []),
                        'complexity': self._estimate_complexity_level(description).value,
                        'tags': self._extract_tags(description)
                    })
        
        return related_stories
    
    def _find_related_acceptance_criteria(self, content: str, start_pos: int, end_pos: int) -> List[str]:
        """Find acceptance criteria related to a requirement/story"""
        # Look for criteria in the 1000 characters following the requirement
        search_content = content[end_pos:end_pos + 1000]
        criteria = []
        
        # First, look for explicit "Acceptance Criteria" sections
        ac_section_match = re.search(r'Acceptance\s+Criteria:?\s*\n([\s\S]*?)(?=\n\n|\n[A-Z][^a-z\n]{5,}|$)', search_content, re.IGNORECASE)
        if ac_section_match:
            ac_content = ac_section_match.group(1)
            # Extract bullet points from acceptance criteria section
            ac_bullets = re.findall(r'[-*]\s*([^\n]{5,200})', ac_content)
            for bullet in ac_bullets:
                cleaned_bullet = bullet.strip()
                if len(cleaned_bullet) > 5 and cleaned_bullet not in criteria:
                    # Filter out duplicates and partial matches
                    is_duplicate = False
                    for existing in criteria:
                        if cleaned_bullet in existing or existing in cleaned_bullet:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        criteria.append(cleaned_bullet)
        
        # Then look for patterns (only if no criteria found from explicit section)
        if not criteria:
            for pattern in self.patterns.ACCEPTANCE_CRITERIA_PATTERNS:
                matches = re.finditer(pattern, search_content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    if "Given" in pattern:
                        # Gherkin format
                        criterion = f"Given {match.group(1)} When {match.group(2)} Then {match.group(3)}"
                    else:
                        criterion = match.group(1).strip()
                    
                    if len(criterion) > 5 and criterion not in criteria:
                        criteria.append(criterion)
        
        # Clean up criteria - remove generic terms and noise
        cleaned_criteria = []
        for criterion in criteria:
            # Skip noisy/generic criteria
            if criterion.lower() in ['criteria:', 'acceptance criteria:', 'acceptance', 'criteria']:
                continue
            # Skip if it's just a fragment
            if len(criterion.split()) < 3:
                continue
            cleaned_criteria.append(criterion)
        
        return cleaned_criteria[:10]  # Limit to 10 criteria
    
    def _extract_following_content(self, content: str, start_pos: int, max_chars: int) -> str:
        """Extract content following a match, up to max_chars"""
        following_content = content[start_pos:start_pos + max_chars]
        
        # Stop at the next heading or major delimiter
        stop_patterns = [r'\n#+\s', r'\n\n[A-Z][^a-z\n]{10,}', r'\n-{3,}', r'\n={3,}']
        
        for pattern in stop_patterns:
            match = re.search(pattern, following_content)
            if match:
                following_content = following_content[:match.start()]
                break
        
        return following_content.strip()
    
    def _generate_story_title(self, actor: str, action: str) -> str:
        """Generate a concise title for a user story"""
        action_words = action.split()[:5]  # First 5 words
        action_short = " ".join(action_words)
        return f"{actor.title()} - {action_short}"
    
    def _parse_story_components(self, description: str) -> Tuple[str, str, str]:
        """Parse actor, action, and benefit from story description"""
        # Try to extract from standard format
        as_match = re.search(r'As\s+(?:a|an)\s+([^,]+),?\s*I\s+want\s+(?:to\s+)?(.+?)(?:\s+so\s+that\s+(.+?))?[.\n]?$', description, re.IGNORECASE)
        
        if as_match:
            actor = as_match.group(1).strip()
            action = as_match.group(2).strip()
            benefit = as_match.group(3).strip() if as_match.group(3) else ""
            return actor, action, benefit
        
        # Fallback: treat entire description as action
        return "user", description, ""
    
    def _estimate_story_points(self, description: str, criteria: List[str]) -> int:
        """Estimate story points based on description and criteria"""
        base_points = 1
        
        # Add points based on description length
        if len(description) > 100:
            base_points += 1
        if len(description) > 200:
            base_points += 1
        
        # Add points based on acceptance criteria count
        base_points += min(len(criteria) // 2, 3)
        
        # Add points for complexity keywords
        complexity_keywords = ['integration', 'api', 'database', 'authentication', 'validation', 'complex', 'multiple']
        for keyword in complexity_keywords:
            if keyword in description.lower():
                base_points += 1
        
        # Cap at reasonable maximum
        return min(base_points, 13)
    
    def _estimate_epic_effort(self, title: str, description: str) -> str:
        """Estimate effort for an epic"""
        content = (title + " " + description).lower()
        
        # Count complexity indicators
        complexity_score = 0
        complexity_indicators = {
            'simple': ['simple', 'basic', 'minimal', 'quick'],
            'medium': ['standard', 'typical', 'normal', 'moderate'],
            'complex': ['complex', 'advanced', 'comprehensive', 'full', 'complete'],
            'very_complex': ['enterprise', 'scalable', 'distributed', 'microservice', 'integration']
        }
        
        for level, keywords in complexity_indicators.items():
            for keyword in keywords:
                if keyword in content:
                    if level == 'simple':
                        complexity_score += 1
                    elif level == 'medium':
                        complexity_score += 2
                    elif level == 'complex':
                        complexity_score += 3
                    elif level == 'very_complex':
                        complexity_score += 4
        
        # Estimate effort based on complexity score
        if complexity_score <= 2:
            return "1 week"
        elif complexity_score <= 5:
            return "2-3 weeks"
        elif complexity_score <= 10:
            return "1 month"
        else:
            return "2+ months"
    
    def _estimate_complexity_level(self, content: str, additional_content: str = "") -> Complexity:
        """Estimate complexity level for a requirement"""
        full_content = (content + " " + additional_content).lower()
        
        # Count complexity indicators
        complexity_score = 0
        
        # Technical complexity indicators
        technical_keywords = ['api', 'database', 'integration', 'authentication', 'encryption', 'algorithm']
        complexity_score += sum(1 for keyword in technical_keywords if keyword in full_content)
        
        # Scale indicators
        scale_keywords = ['multiple', 'various', 'different', 'several', 'many', 'all', 'entire']
        complexity_score += sum(1 for keyword in scale_keywords if keyword in full_content) * 0.5
        
        # Process complexity indicators
        process_keywords = ['workflow', 'process', 'sequence', 'steps', 'phases', 'stages']
        complexity_score += sum(1 for keyword in process_keywords if keyword in full_content) * 0.5
        
        # Determine complexity level
        if complexity_score < 1:
            return Complexity.LOW
        elif complexity_score < 2.5:
            return Complexity.MEDIUM
        elif complexity_score < 4:
            return Complexity.HIGH
        else:
            return Complexity.VERY_HIGH
    
    def _determine_priority(self, content: str, additional_content: str = "") -> Priority:
        """Determine priority level for a requirement"""
        full_content = (content + " " + additional_content).lower()
        
        # High priority indicators
        high_priority_keywords = ['critical', 'must', 'required', 'essential', 'mandatory', 'security', 'compliance']
        if any(keyword in full_content for keyword in high_priority_keywords):
            return Priority.HIGH
        
        # Critical priority indicators
        critical_keywords = ['urgent', 'immediately', 'asap', 'blocker', 'showstopper']
        if any(keyword in full_content for keyword in critical_keywords):
            return Priority.CRITICAL
        
        # Low priority indicators
        low_priority_keywords = ['nice', 'optional', 'future', 'enhancement', 'improvement', 'could']
        if any(keyword in full_content for keyword in low_priority_keywords):
            return Priority.LOW
        
        # Default to medium priority
        return Priority.MEDIUM
    
    def _extract_tags(self, content: str, additional_content: str = "") -> List[str]:
        """Extract relevant tags from content"""
        full_content = (content + " " + additional_content).lower()
        
        tag_categories = {
            'ui': ['interface', 'ui', 'frontend', 'user interface', 'form', 'button', 'screen'],
            'backend': ['backend', 'server', 'api', 'service', 'database', 'storage'],
            'security': ['security', 'authentication', 'authorization', 'encryption', 'privacy'],
            'integration': ['integration', 'third-party', 'external', 'api', 'webhook'],
            'performance': ['performance', 'speed', 'fast', 'optimization', 'scalability'],
            'mobile': ['mobile', 'ios', 'android', 'responsive', 'device'],
            'data': ['data', 'database', 'analytics', 'reporting', 'export', 'import'],
            'admin': ['admin', 'administration', 'management', 'configuration', 'settings']
        }
        
        tags = []
        for tag, keywords in tag_categories.items():
            if any(keyword in full_content for keyword in keywords):
                tags.append(tag)
        
        return tags[:5]  # Limit to 5 tags
    
    def _build_requirement_hierarchy(self) -> Dict:
        """Build hierarchical relationships between requirements"""
        logger.info("Building requirement hierarchy...")
        
        hierarchy = {}
        
        # Map epics to their stories and requirements
        for epic in self.epics:
            epic_id = epic['id']
            hierarchy[epic_id] = {
                'type': 'epic',
                'title': epic['title'],
                'stories': epic.get('stories', []),
                'requirements': [],
                'sub_epics': epic.get('sub_epics', [])
            }
            
            # Find requirements that might belong to this epic
            epic_keywords = set(epic['title'].lower().split())
            for req in self.requirements:
                req_keywords = set(req.description.lower().split())
                if len(epic_keywords.intersection(req_keywords)) >= 2:
                    hierarchy[epic_id]['requirements'].append(req.id)
        
        # Map stories to their acceptance criteria requirements
        for story in self.stories:
            story_id = story['id']
            hierarchy[story_id] = {
                'type': 'story',
                'title': story['title'],
                'requirements': [],
                'acceptance_criteria': story.get('acceptance_criteria', [])
            }
            
            # Find requirements that relate to this story
            story_keywords = set((story['actor'] + " " + story['action']).lower().split())
            for req in self.requirements:
                req_keywords = set(req.description.lower().split())
                if len(story_keywords.intersection(req_keywords)) >= 1:
                    hierarchy[story_id]['requirements'].append(req.id)
        
        return hierarchy
    
    def _analyze_dependencies(self):
        """Analyze dependencies between requirements"""
        logger.info("Analyzing requirement dependencies...")
        
        # Build keyword maps for each requirement
        req_keywords = {}
        for req in self.requirements:
            keywords = set(req.title.lower().split() + req.description.lower().split())
            req_keywords[req.id] = keywords
        
        # Find dependencies based on keyword overlap and logical relationships
        dependency_keywords = {
            'depends_on': ['depends', 'requires', 'needs', 'prerequisite', 'after'],
            'blocks': ['before', 'prerequisite for', 'required by'],
            'related': ['related', 'similar', 'also', 'additionally']
        }
        
        for req in self.requirements:
            req_text = (req.title + " " + req.description).lower()
            
            # Look for explicit dependency mentions
            for other_req in self.requirements:
                if req.id == other_req.id:
                    continue
                
                other_keywords = req_keywords[other_req.id]
                req_keywords_set = req_keywords[req.id]
                
                # Check for keyword overlap (potential dependency)
                overlap = len(req_keywords_set.intersection(other_keywords))
                if overlap >= 2:
                    # Check for dependency indicators
                    for dep_type, indicators in dependency_keywords.items():
                        if any(indicator in req_text for indicator in indicators):
                            if other_req.id not in req.dependencies:
                                req.dependencies.append(other_req.id)
                            break
    
    def _estimate_complexity_and_tokens(self):
        """Estimate complexity and token counts for all requirements"""
        logger.info("Estimating complexity and token counts...")
        
        for req in self.requirements:
            # Estimate tokens needed for this requirement in an issue
            base_tokens = len(req.description) // 4  # Rough estimation: 1 token ≈ 4 characters
            
            # Add context tokens
            context_tokens = 500  # Base context
            
            # Add acceptance criteria tokens
            criteria_tokens = sum(len(c) // 4 for c in req.acceptance_criteria)
            
            # Add dependency context tokens
            dependency_tokens = len(req.dependencies) * 100
            
            # Add source document context
            doc_context_tokens = 200
            
            req.estimated_tokens = int(base_tokens + context_tokens + criteria_tokens + dependency_tokens + doc_context_tokens)
            
            # Ensure complexity is set (may have been set during extraction)
            if not req.complexity:
                req.complexity = self._estimate_complexity_level(req.description)
    
    def _prioritize_requirements(self):
        """Assign priorities to all requirements"""
        logger.info("Prioritizing requirements...")
        
        for req in self.requirements:
            # Priority may have been set during extraction
            if not req.priority:
                req.priority = self._determine_priority(req.description)
            
            # Adjust priority based on dependencies
            if len(req.dependencies) > 3:
                # High dependency count might increase priority
                if req.priority == Priority.LOW:
                    req.priority = Priority.MEDIUM
                elif req.priority == Priority.MEDIUM:
                    req.priority = Priority.HIGH
    
    def _generate_statistics(self) -> Dict:
        """Generate statistics about extracted requirements"""
        stats = {
            'total_requirements': len(self.requirements),
            'total_epics': len(self.epics),
            'total_stories': len(self.stories),
            'by_type': {},
            'by_priority': {},
            'by_complexity': {},
            'avg_tokens': 0,
            'total_acceptance_criteria': 0
        }
        
        # Count by requirement type
        for req in self.requirements:
            req_type = req.type.value
            stats['by_type'][req_type] = stats['by_type'].get(req_type, 0) + 1
            
            # Count by priority
            priority = req.priority.value
            stats['by_priority'][priority] = stats['by_priority'].get(priority, 0) + 1
            
            # Count by complexity
            complexity = req.complexity.value
            stats['by_complexity'][complexity] = stats['by_complexity'].get(complexity, 0) + 1
            
            # Count acceptance criteria
            stats['total_acceptance_criteria'] += len(req.acceptance_criteria)
        
        # Calculate average tokens
        if self.requirements:
            stats['avg_tokens'] = sum(req.estimated_tokens for req in self.requirements) / len(self.requirements)
        
        return stats
    
    def save_requirements_json(self, output_path: str) -> Path:
        """Save extracted requirements to JSON file"""
        requirements_data = self.extract_all_requirements()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(requirements_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Requirements saved to: {output_path}")
        return output_path
    
    def get_requirements_by_type(self, req_type: RequirementType) -> List[Requirement]:
        """Get requirements filtered by type"""
        return [req for req in self.requirements if req.type == req_type]
    
    def get_requirements_by_priority(self, priority: Priority) -> List[Requirement]:
        """Get requirements filtered by priority"""
        return [req for req in self.requirements if req.priority == priority]
    
    def get_requirements_by_complexity(self, complexity: Complexity) -> List[Requirement]:
        """Get requirements filtered by complexity"""
        return [req for req in self.requirements if req.complexity == complexity]


def main():
    """Main function for CLI usage"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Extract requirements from flattened documentation XML")
    parser.add_argument("xml_file", help="Path to flattened documentation XML file")
    parser.add_argument("-o", "--output", help="Output JSON file path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--stats-only", action="store_true", help="Show only statistics")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        extractor = RequirementExtractor(xml_file_path=args.xml_file)
        requirements_data = extractor.extract_all_requirements()
        
        if args.stats_only:
            print(json.dumps(requirements_data['statistics'], indent=2))
        elif args.output:
            output_path = extractor.save_requirements_json(args.output)
            print(f"Requirements saved to: {output_path}")
            print(f"Extracted {requirements_data['statistics']['total_requirements']} requirements")
            print(f"Extracted {requirements_data['statistics']['total_epics']} epics")
            print(f"Extracted {requirements_data['statistics']['total_stories']} user stories")
        else:
            print(json.dumps(requirements_data, indent=2))
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()