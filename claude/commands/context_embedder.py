#!/usr/bin/env python3
"""
Context Embedding Engine for RIF Framework

Implements a sophisticated context embedding engine that calculates optimal context for each issue,
ensuring every GitHub issue contains complete implementation guidance within Claude's context window limits.

Author: RIF-Implementer
Issue: #241
Dependencies: Issue #240 (requirement_extractor.py)
"""

import tiktoken
from typing import Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass
import json
import re
import logging
from pathlib import Path
from datetime import datetime

# Import dependencies from existing modules - supports both module and CLI usage
try:
    from .requirement_extractor import RequirementExtractor, RequirementType, Priority, Complexity
    from .document_flattener import DocumentFlattener
except ImportError:
    from requirement_extractor import RequirementExtractor, RequirementType, Priority, Complexity
    from document_flattener import DocumentFlattener

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContextWindow:
    """Represents a context window for an issue"""
    primary_requirement: str
    related_requirements: List[str]
    architecture_context: str
    design_context: str
    dependencies: List[str]
    implementation_hints: str
    acceptance_criteria: List[str]
    estimated_tokens: int
    overflow_strategy: str
    context_metadata: Dict = None

    def __post_init__(self):
        if self.context_metadata is None:
            self.context_metadata = {}


class TokenManager:
    """
    Manages token counting and optimization for context windows
    """
    
    def __init__(self):
        self.encoder = tiktoken.encoding_for_model("gpt-4")
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not text:
            return 0
        return len(self.encoder.encode(text))
    
    def trim_to_tokens(self, text: str, max_tokens: int) -> str:
        """Trim text to fit within token limit"""
        if not text:
            return ""
            
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Trim and add ellipsis
        if max_tokens < 3:
            return "..."
        trimmed_tokens = tokens[:max_tokens - 3]
        return self.encoder.decode(trimmed_tokens) + "..."
    
    def optimize_context(self, context_parts: Dict[str, str], 
                        max_total: int = 2200) -> Dict[str, str]:
        """
        Optimize multiple context parts to fit within total limit
        """
        # Calculate current total
        current_total = sum(self.count_tokens(text) for text in context_parts.values())
        
        if current_total <= max_total:
            return context_parts
        
        # Prioritized trimming (lowest to highest priority)
        priority_order = [
            'design_context',      # Lowest priority
            'related_requirements',
            'dependencies',
            'architecture_context',
            'implementation_hints',
            'acceptance_criteria',
            'primary_requirement'  # Highest priority - trim last
        ]
        
        optimized = context_parts.copy()
        
        for key in priority_order:
            if key not in optimized:
                continue
                
            # Calculate how much to trim
            current_total = sum(self.count_tokens(text) for text in optimized.values())
            overflow = current_total - max_total
            if overflow <= 0:
                break
            
            # Trim this component by up to 50%
            current_tokens = self.count_tokens(optimized[key])
            trim_amount = min(overflow, current_tokens // 2)
            
            if trim_amount > 0:
                new_limit = current_tokens - trim_amount
                optimized[key] = self.trim_to_tokens(optimized[key], new_limit)
        
        return optimized


class ContextEmbedder:
    """
    Embeds relevant context from all documentation into each issue.
    Optimizes for Claude's context window while maintaining completeness.
    """
    
    # Target token counts for different components
    TOKEN_TARGETS = {
        'issue_template': 200,      # Base issue structure
        'requirement': 300,          # Core requirement description
        'architecture': 400,         # Architecture context
        'design': 300,              # Design/UI context
        'dependencies': 200,         # Dependency information
        'implementation': 500,       # Implementation guidance
        'acceptance': 200,          # Acceptance criteria
        'buffer': 100               # Safety buffer
    }
    
    MAX_TOKENS = 2200  # Leave room for Claude's response
    
    def __init__(self, documents_xml: str = None, requirements: Dict = None, 
                 xml_file_path: str = None, requirements_file_path: str = None):
        """
        Initialize the context embedder
        
        Args:
            documents_xml: XML string from document flattener
            requirements: Requirements dictionary from requirement extractor
            xml_file_path: Path to XML file (alternative to documents_xml)
            requirements_file_path: Path to requirements JSON file
        """
        self.token_manager = TokenManager()
        self.context_cache = {}
        
        # Load documents
        if documents_xml:
            self.documents = self._parse_documents_xml(documents_xml)
        elif xml_file_path and Path(xml_file_path).exists():
            with open(xml_file_path, 'r', encoding='utf-8') as f:
                documents_xml = f.read()
            self.documents = self._parse_documents_xml(documents_xml)
        else:
            self.documents = {}
            logger.warning("No documents provided for context embedding")
        
        # Load requirements
        if requirements:
            self.requirements = requirements
        elif requirements_file_path and Path(requirements_file_path).exists():
            with open(requirements_file_path, 'r', encoding='utf-8') as f:
                self.requirements = json.load(f)
        else:
            self.requirements = {}
            logger.warning("No requirements provided for context embedding")
            
        logger.info("ContextEmbedder initialized")
        
    def _parse_documents_xml(self, xml_content: str) -> Dict:
        """Parse documents XML into structured format"""
        import xml.etree.ElementTree as ET
        
        try:
            root = ET.fromstring(xml_content)
            documents = {}
            
            for doc in root.findall(".//document"):
                doc_type = doc.get('type', 'unknown')
                doc_name = doc.get('name', 'unknown')
                
                doc_data = {
                    'type': doc_type,
                    'path': doc.get('path', ''),
                    'sections': []
                }
                
                for section in doc.findall('.//section'):
                    section_data = {
                        'title': section.get('title', ''),
                        'level': int(section.get('level', 1)),
                        'content': ''
                    }
                    
                    content_elem = section.find('content')
                    if content_elem is not None and content_elem.text:
                        section_data['content'] = content_elem.text.strip()
                    
                    doc_data['sections'].append(section_data)
                
                documents[doc_name] = doc_data
            
            return documents
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse documents XML: {e}")
            return {}
    
    def embed_context_for_requirement(self, requirement_id: str) -> ContextWindow:
        """
        Create optimal context window for a single requirement
        """
        requirement = self._get_requirement(requirement_id)
        if not requirement:
            logger.error(f"Requirement {requirement_id} not found")
            return self._create_empty_context_window()
        
        # Initialize context window
        context = ContextWindow(
            primary_requirement="",
            related_requirements=[],
            architecture_context="",
            design_context="",
            dependencies=[],
            implementation_hints="",
            acceptance_criteria=requirement.get('acceptance_criteria', []),
            estimated_tokens=0,
            overflow_strategy="none",
            context_metadata={
                'requirement_id': requirement_id,
                'generated_at': datetime.now().isoformat()
            }
        )
        
        # Calculate token budget for each component
        token_budget = self._calculate_token_budget(requirement)
        
        # 1. Embed primary requirement (highest priority)
        context.primary_requirement = self._embed_primary_requirement(
            requirement, 
            token_budget['requirement']
        )
        
        # 2. Embed architecture context
        context.architecture_context = self._embed_architecture_context(
            requirement,
            token_budget['architecture']
        )
        
        # 3. Embed design context if applicable
        if self._needs_design_context(requirement):
            context.design_context = self._embed_design_context(
                requirement,
                token_budget['design']
            )
        
        # 4. Embed dependencies
        context.dependencies = self._embed_dependencies(
            requirement,
            token_budget['dependencies']
        )
        
        # 5. Find related requirements
        context.related_requirements = self._find_related_requirements(
            requirement,
            token_budget.get('related', 150)
        )
        
        # 6. Generate implementation hints
        context.implementation_hints = self._generate_implementation_hints(
            requirement,
            context,
            token_budget['implementation']
        )
        
        # 7. Handle overflow if needed
        context = self._handle_overflow(context)
        
        # 8. Calculate final token count
        context.estimated_tokens = self._count_context_tokens(context)
        
        return context
    
    def _get_requirement(self, requirement_id: str) -> Optional[Dict]:
        """Get requirement by ID from various sources"""
        # Look in requirements list
        if 'requirements' in self.requirements:
            for req in self.requirements['requirements']:
                if req.get('id') == requirement_id:
                    return req
        
        # Look in stories
        if 'stories' in self.requirements:
            for story in self.requirements['stories']:
                if story.get('id') == requirement_id:
                    return story
        
        # Look in epics
        if 'epics' in self.requirements:
            for epic in self.requirements['epics']:
                if epic.get('id') == requirement_id:
                    return epic
        
        return None
    
    def _calculate_token_budget(self, requirement: Dict) -> Dict[str, int]:
        """Calculate token budget allocation for different components"""
        complexity = requirement.get('complexity', 'medium')
        req_type = requirement.get('type', 'functional')
        
        # Base budget
        budget = self.TOKEN_TARGETS.copy()
        
        # Adjust based on complexity
        if complexity == 'high' or complexity == 'very-high':
            budget['implementation'] += 200
            budget['architecture'] += 100
        elif complexity == 'low':
            budget['implementation'] -= 100
            budget['architecture'] -= 100
        
        # Adjust based on type
        if req_type == 'epic':
            budget['architecture'] += 150
            budget['dependencies'] += 100
        elif req_type == 'user_story':
            budget['design'] += 100
        elif req_type == 'technical':
            budget['implementation'] += 150
        
        return budget
    
    def _embed_primary_requirement(self, requirement: Dict, token_limit: int) -> str:
        """
        Format and embed the primary requirement with maximum clarity
        """
        req_type = requirement.get('type', 'functional')
        title = requirement.get('title', 'Untitled Requirement')
        description = requirement.get('description', '')
        
        if req_type == 'user_story':
            # Format user story
            actor = requirement.get('actor', 'user')
            action = requirement.get('action', '')
            benefit = requirement.get('benefit', '')
            
            template = f"""## User Story: {title}

**Actor**: {actor}
**Priority**: {requirement.get('priority', 'medium')}
**Complexity**: {requirement.get('complexity', 'medium')}
**Story Points**: {requirement.get('estimated_points', 'TBD')}

### Description
{description}

### Business Value
{benefit or 'Improves system functionality'}

### User Story Format
As a {actor}, I want {action}""" + (f" so that {benefit}" if benefit else "") + """

### Source
From: {source_document} > {source_section}
""".format(
    source_document=requirement.get('source_document', 'Unknown'),
    source_section=requirement.get('source_section', 'Unknown')
)
        
        elif req_type == 'epic':
            template = f"""## Epic: {title}

**Type**: {req_type}
**Priority**: {requirement.get('priority', 'medium')}
**Complexity**: {requirement.get('complexity', 'medium')}
**Estimated Effort**: {requirement.get('estimated_effort', 'TBD')}

### Description
{description}

### Stories Included
{self._format_epic_stories(requirement)}

### Source
From: {requirement.get('source_document', 'Unknown')} > {requirement.get('source_section', 'Unknown')}
"""
        
        else:
            template = f"""## Requirement: {title}

**Type**: {req_type}
**Priority**: {requirement.get('priority', 'medium')}
**Complexity**: {requirement.get('complexity', 'medium')}
**ID**: {requirement.get('id', 'Unknown')}

### Description
{description}

### Tags
{', '.join(requirement.get('tags', []))}

### Source
From: {requirement.get('source_document', 'Unknown')} > {requirement.get('source_section', 'Unknown')}
"""
        
        # Trim if exceeds token limit
        return self.token_manager.trim_to_tokens(template, token_limit)
    
    def _format_epic_stories(self, epic: Dict) -> str:
        """Format stories within an epic"""
        stories = epic.get('stories', [])
        if not stories:
            return "No specific stories defined yet."
        
        story_text = ""
        for story_id in stories[:5]:  # Limit to 5 stories
            story = self._get_requirement(story_id)
            if story:
                story_text += f"- {story_id}: {story.get('title', 'Unknown')}\n"
        
        if len(stories) > 5:
            story_text += f"- ... and {len(stories) - 5} more stories\n"
        
        return story_text.strip()
    
    def _embed_architecture_context(self, requirement: Dict, token_limit: int) -> str:
        """
        Extract and embed relevant architecture context
        """
        # Find related architecture sections
        arch_sections = self._find_related_architecture(requirement)
        
        if not arch_sections:
            return "No specific architecture requirements identified."
        
        context = "### Architecture Context\n\n"
        
        for section in arch_sections[:3]:  # Limit to top 3 relevant sections
            context += f"""#### {section['title']}
{section['content'][:500]}

**Key Components**:
{self._extract_components(section)}

**Integration Points**:
{self._extract_integrations(section)}

"""
        
        return self.token_manager.trim_to_tokens(context, token_limit)
    
    def _find_related_architecture(self, requirement: Dict) -> List[Dict]:
        """Find architecture sections related to the requirement"""
        related_sections = []
        req_keywords = self._extract_keywords(requirement.get('description', ''))
        
        for doc_name, doc_data in self.documents.items():
            if doc_data['type'] != 'architecture':
                continue
            
            for section in doc_data['sections']:
                section_keywords = self._extract_keywords(section['content'])
                
                # Calculate relevance score
                overlap = len(req_keywords.intersection(section_keywords))
                if overlap >= 2:  # At least 2 common keywords
                    related_sections.append({
                        'title': section['title'],
                        'content': section['content'],
                        'document': doc_name,
                        'relevance': overlap
                    })
        
        # Sort by relevance
        related_sections.sort(key=lambda x: x['relevance'], reverse=True)
        return related_sections
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text"""
        if not text:
            return set()
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
            'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy',
            'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'will', 'that',
            'this', 'with', 'have', 'from', 'they', 'know', 'want', 'been', 'good',
            'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like',
            'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well',
            'were'
        }
        
        return set(word for word in words if word not in stop_words and len(word) > 2)
    
    def _extract_components(self, section: Dict) -> str:
        """Extract system components from architecture section"""
        content = section['content']
        
        # Look for component patterns
        component_patterns = [
            r'(\w+Service)',
            r'(\w+Controller)', 
            r'(\w+Manager)',
            r'(\w+Handler)',
            r'(\w+Engine)',
            r'(\w+API)',
            r'(\w+Gateway)'
        ]
        
        components = set()
        for pattern in component_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            components.update(matches)
        
        if components:
            return "- " + "\n- ".join(sorted(components))
        return "Components to be identified during implementation"
    
    def _extract_integrations(self, section: Dict) -> str:
        """Extract integration points from architecture section"""
        content = section['content']
        
        # Look for integration patterns
        integration_keywords = [
            'API', 'REST', 'GraphQL', 'webhook', 'integration', 'interface',
            'endpoint', 'service', 'microservice', 'database', 'queue'
        ]
        
        integrations = []
        for keyword in integration_keywords:
            if keyword.lower() in content.lower():
                integrations.append(keyword)
        
        if integrations:
            return "- " + "\n- ".join(integrations[:5])
        return "Integration points to be defined during implementation"
    
    def _needs_design_context(self, requirement: Dict) -> bool:
        """Determine if requirement needs design context"""
        req_type = requirement.get('type', '')
        description = requirement.get('description', '').lower()
        tags = requirement.get('tags', [])
        
        # UI/UX related requirements need design context
        ui_indicators = ['ui', 'interface', 'form', 'button', 'screen', 'page', 'frontend']
        
        return (req_type == 'user_story' or
                'ui' in tags or
                any(indicator in description for indicator in ui_indicators))
    
    def _embed_design_context(self, requirement: Dict, token_limit: int) -> str:
        """
        Extract and embed relevant design context
        """
        # Find related design sections
        design_sections = self._find_related_design(requirement)
        
        if not design_sections:
            return "No specific design requirements identified."
        
        context = "### Design Context\n\n"
        
        for section in design_sections[:2]:  # Limit to top 2 relevant sections
            context += f"""#### {section['title']}
{section['content'][:300]}

**UI Components**:
{self._extract_ui_components(section)}

"""
        
        return self.token_manager.trim_to_tokens(context, token_limit)
    
    def _find_related_design(self, requirement: Dict) -> List[Dict]:
        """Find design sections related to the requirement"""
        related_sections = []
        req_keywords = self._extract_keywords(requirement.get('description', ''))
        
        for doc_name, doc_data in self.documents.items():
            if doc_data['type'] != 'design':
                continue
            
            for section in doc_data['sections']:
                section_keywords = self._extract_keywords(section['content'])
                
                # Calculate relevance score
                overlap = len(req_keywords.intersection(section_keywords))
                if overlap >= 1:  # At least 1 common keyword for design
                    related_sections.append({
                        'title': section['title'],
                        'content': section['content'],
                        'document': doc_name,
                        'relevance': overlap
                    })
        
        # Sort by relevance
        related_sections.sort(key=lambda x: x['relevance'], reverse=True)
        return related_sections
    
    def _extract_ui_components(self, section: Dict) -> str:
        """Extract UI components from design section"""
        content = section['content']
        
        # Look for UI component patterns
        ui_patterns = [
            r'(\w+Button)',
            r'(\w+Form)', 
            r'(\w+Modal)',
            r'(\w+Dialog)',
            r'(\w+Panel)',
            r'(\w+Menu)',
            r'(\w+Input)'
        ]
        
        components = set()
        for pattern in ui_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            components.update(matches)
        
        if components:
            return "- " + "\n- ".join(sorted(components))
        return "UI components to be designed during implementation"
    
    def _embed_dependencies(self, requirement: Dict, token_limit: int) -> List[str]:
        """
        Extract and embed dependency information
        """
        dependencies = []
        
        # Get explicit dependencies from requirement
        req_deps = requirement.get('dependencies', [])
        dependencies.extend(req_deps)
        
        # Analyze content for implied dependencies
        description = requirement.get('description', '').lower()
        
        # Common dependency patterns
        if 'authentication' in description or 'login' in description:
            dependencies.append("User authentication system")
        if 'database' in description or 'data' in description:
            dependencies.append("Database connection and schema")
        if 'api' in description:
            dependencies.append("API framework and routing")
        if 'email' in description or 'notification' in description:
            dependencies.append("Email/notification service")
        
        # Remove duplicates and limit
        dependencies = list(set(dependencies))[:5]
        
        return dependencies
    
    def _find_related_requirements(self, requirement: Dict, token_limit: int) -> List[str]:
        """Find requirements related to the current one"""
        related = []
        
        req_keywords = self._extract_keywords(requirement.get('description', ''))
        current_id = requirement.get('id')
        
        # Check all requirements for similarity
        all_requirements = []
        if 'requirements' in self.requirements:
            all_requirements.extend(self.requirements['requirements'])
        if 'stories' in self.requirements:
            all_requirements.extend(self.requirements['stories'])
        
        for other_req in all_requirements:
            if other_req.get('id') == current_id:
                continue
            
            other_keywords = self._extract_keywords(other_req.get('description', ''))
            overlap = len(req_keywords.intersection(other_keywords))
            
            if overlap >= 2:  # Significant overlap
                related.append(f"{other_req.get('id')}: {other_req.get('title', 'Unknown')}")
        
        return related[:3]  # Limit to 3 most related
    
    def _generate_implementation_hints(self, 
                                     requirement: Dict, 
                                     context: ContextWindow,
                                     token_limit: int) -> str:
        """
        Generate specific implementation guidance based on requirement and context
        """
        hints = "### Implementation Guidance\n\n"
        
        # Add step-by-step approach
        hints += "#### Suggested Approach\n"
        steps = self._generate_implementation_steps(requirement, context)
        for i, step in enumerate(steps, 1):
            hints += f"{i}. {step}\n"
        
        # Add technical considerations
        hints += "\n#### Technical Considerations\n"
        considerations = self._extract_technical_considerations(requirement)
        for consideration in considerations:
            hints += f"- {consideration}\n"
        
        # Add code patterns if applicable
        patterns = self._find_relevant_patterns(requirement)
        if patterns:
            hints += "\n#### Relevant Patterns\n"
            for pattern in patterns[:2]:  # Limit to 2 patterns
                hints += f"""**{pattern['name']}**
```{pattern.get('language', 'python')}
{pattern['code'][:300]}
```

"""
        
        # Add testing guidance
        hints += "#### Testing Requirements\n"
        hints += self._generate_testing_guidance(requirement)
        
        return self.token_manager.trim_to_tokens(hints, token_limit)
    
    def _generate_implementation_steps(self, requirement: Dict, context: ContextWindow) -> List[str]:
        """Generate step-by-step implementation approach"""
        req_type = requirement.get('type', 'functional')
        complexity = requirement.get('complexity', 'medium')
        
        if req_type == 'user_story':
            return [
                "Create user interface mockups/wireframes",
                "Set up backend API endpoints",
                "Implement data models and database schema",
                "Build frontend components",
                "Add input validation and error handling",
                "Implement business logic",
                "Add unit and integration tests",
                "Perform user acceptance testing"
            ]
        elif req_type == 'technical':
            return [
                "Research technical requirements and constraints",
                "Design system architecture",
                "Set up development environment and tooling",
                "Implement core functionality",
                "Add configuration and environment handling",
                "Implement monitoring and logging",
                "Add comprehensive testing",
                "Document implementation and deployment"
            ]
        elif req_type == 'epic':
            return [
                "Break down epic into manageable user stories",
                "Prioritize stories based on dependencies",
                "Create overall architecture plan",
                "Implement foundational components first",
                "Develop stories incrementally",
                "Integrate components as stories complete",
                "Conduct end-to-end testing",
                "Prepare for production deployment"
            ]
        else:
            return [
                "Analyze requirement details and acceptance criteria",
                "Design solution approach",
                "Identify required components and dependencies",
                "Implement core functionality",
                "Add validation and error handling",
                "Create comprehensive tests",
                "Review and refactor code",
                "Document solution"
            ]
    
    def _extract_technical_considerations(self, requirement: Dict) -> List[str]:
        """Extract technical considerations for the requirement"""
        considerations = []
        description = requirement.get('description', '').lower()
        req_type = requirement.get('type', '')
        
        # Security considerations
        if any(word in description for word in ['password', 'login', 'auth', 'security']):
            considerations.append("Implement proper security measures (authentication, authorization)")
        
        # Performance considerations
        if any(word in description for word in ['large', 'scale', 'many', 'fast', 'performance']):
            considerations.append("Consider performance implications and optimization")
        
        # Database considerations
        if 'database' in description or 'data' in description:
            considerations.append("Design efficient database schema and queries")
        
        # API considerations
        if 'api' in description:
            considerations.append("Design RESTful API with proper status codes and documentation")
        
        # UI considerations
        if req_type == 'user_story':
            considerations.append("Ensure responsive design and accessibility")
            considerations.append("Implement proper form validation and user feedback")
        
        # Error handling
        considerations.append("Add comprehensive error handling and logging")
        
        # Testing
        considerations.append("Include unit tests, integration tests, and appropriate end-to-end tests")
        
        return considerations
    
    def _find_relevant_patterns(self, requirement: Dict) -> List[Dict]:
        """Find relevant code patterns for the requirement"""
        patterns = []
        description = requirement.get('description', '').lower()
        
        # This would ideally integrate with the knowledge base
        # For now, provide common patterns based on requirement content
        
        if 'authentication' in description or 'login' in description:
            patterns.append({
                'name': 'Authentication Pattern',
                'language': 'python',
                'code': '''
def authenticate_user(username, password):
    """Authenticate user with secure password hashing"""
    user = User.get_by_username(username)
    if user and bcrypt.check_password_hash(user.password_hash, password):
        token = jwt.encode({'user_id': user.id}, app.secret_key)
        return {'success': True, 'token': token}
    return {'success': False, 'message': 'Invalid credentials'}
'''
            })
        
        if 'api' in description:
            patterns.append({
                'name': 'REST API Pattern',
                'language': 'python',
                'code': '''
@app.route('/api/resource', methods=['GET', 'POST'])
def handle_resource():
    """Handle resource CRUD operations"""
    if request.method == 'GET':
        return jsonify(get_resources())
    elif request.method == 'POST':
        data = request.get_json()
        result = create_resource(data)
        return jsonify(result), 201
'''
            })
        
        return patterns
    
    def _generate_testing_guidance(self, requirement: Dict) -> str:
        """Generate testing guidance for the requirement"""
        req_type = requirement.get('type', '')
        
        if req_type == 'user_story':
            return """- Unit tests for business logic components
- Integration tests for API endpoints
- End-to-end tests for user workflows
- Accessibility testing for UI components
- Cross-browser compatibility testing"""
        
        elif req_type == 'technical':
            return """- Unit tests for all functions and methods
- Integration tests for component interactions  
- Performance tests for scalability requirements
- Security tests for vulnerabilities
- Configuration tests for different environments"""
        
        else:
            return """- Unit tests covering core functionality
- Integration tests for component interactions
- Edge case testing for error conditions
- Regression tests to prevent future issues
- Documentation verification"""
    
    def _handle_overflow(self, context: ContextWindow) -> ContextWindow:
        """
        Handle context that exceeds token limits
        """
        total_tokens = self._count_context_tokens(context)
        
        if total_tokens <= self.MAX_TOKENS:
            context.overflow_strategy = "none"
            return context
        
        # Try progressive strategies
        strategies = [
            ('trim_implementation', self._trim_implementation_hints),
            ('trim_design', self._trim_design_context),
            ('trim_architecture', self._trim_architecture_context),
            ('trim_related', self._trim_related_requirements)
        ]
        
        for strategy_name, strategy_func in strategies:
            modified_context = strategy_func(context)
            if self._count_context_tokens(modified_context) <= self.MAX_TOKENS:
                modified_context.overflow_strategy = strategy_name
                return modified_context
        
        # Last resort: hard truncate
        context.overflow_strategy = "hard_truncate"
        return self._hard_truncate(context)
    
    def _count_context_tokens(self, context: ContextWindow) -> int:
        """Count total tokens in a context window"""
        total = 0
        total += self.token_manager.count_tokens(context.primary_requirement)
        total += self.token_manager.count_tokens(context.architecture_context)
        total += self.token_manager.count_tokens(context.design_context)
        total += self.token_manager.count_tokens(context.implementation_hints)
        total += sum(self.token_manager.count_tokens(dep) for dep in context.dependencies)
        total += sum(self.token_manager.count_tokens(req) for req in context.related_requirements)
        total += sum(self.token_manager.count_tokens(ac) for ac in context.acceptance_criteria)
        return total
    
    def _trim_implementation_hints(self, context: ContextWindow) -> ContextWindow:
        """Trim implementation hints to reduce token count"""
        new_context = context
        new_context.implementation_hints = self.token_manager.trim_to_tokens(
            context.implementation_hints, 
            self.token_manager.count_tokens(context.implementation_hints) // 2
        )
        return new_context
    
    def _trim_design_context(self, context: ContextWindow) -> ContextWindow:
        """Trim design context to reduce token count"""
        new_context = context
        new_context.design_context = self.token_manager.trim_to_tokens(
            context.design_context,
            self.token_manager.count_tokens(context.design_context) // 2
        )
        return new_context
    
    def _trim_architecture_context(self, context: ContextWindow) -> ContextWindow:
        """Trim architecture context to reduce token count"""
        new_context = context
        new_context.architecture_context = self.token_manager.trim_to_tokens(
            context.architecture_context,
            self.token_manager.count_tokens(context.architecture_context) // 2
        )
        return new_context
    
    def _trim_related_requirements(self, context: ContextWindow) -> ContextWindow:
        """Trim related requirements to reduce token count"""
        new_context = context
        # Keep only top 2 related requirements
        new_context.related_requirements = context.related_requirements[:2]
        return new_context
    
    def _hard_truncate(self, context: ContextWindow) -> ContextWindow:
        """Hard truncate all context components to fit within limits"""
        # Allocate tokens proportionally
        allocation = {
            'primary_requirement': 400,
            'architecture_context': 300,
            'implementation_hints': 400,
            'design_context': 200,
            'related_requirements': 100,
            'dependencies': 100,
            'acceptance_criteria': 200
        }
        
        context.primary_requirement = self.token_manager.trim_to_tokens(
            context.primary_requirement, allocation['primary_requirement'])
        context.architecture_context = self.token_manager.trim_to_tokens(
            context.architecture_context, allocation['architecture_context'])
        context.implementation_hints = self.token_manager.trim_to_tokens(
            context.implementation_hints, allocation['implementation_hints'])
        context.design_context = self.token_manager.trim_to_tokens(
            context.design_context, allocation['design_context'])
        
        # Trim lists
        context.related_requirements = context.related_requirements[:1]
        context.dependencies = context.dependencies[:3]
        context.acceptance_criteria = context.acceptance_criteria[:5]
        
        return context
    
    def _create_empty_context_window(self) -> ContextWindow:
        """Create an empty context window for missing requirements"""
        return ContextWindow(
            primary_requirement="Requirement not found",
            related_requirements=[],
            architecture_context="",
            design_context="",
            dependencies=[],
            implementation_hints="",
            acceptance_criteria=[],
            estimated_tokens=0,
            overflow_strategy="none"
        )


class ContextTemplates:
    """
    Templates for different types of context embedding
    """
    
    @staticmethod
    def issue_with_full_context(requirement: Dict, context: ContextWindow) -> str:
        """
        Generate complete issue body with embedded context
        """
        req_id = requirement.get('id', 'Unknown')
        req_title = requirement.get('title', 'Unknown Requirement')
        
        return f"""# {req_title}

{context.primary_requirement}

---

## Context

{context.architecture_context}

{context.design_context if context.design_context else ''}

### Dependencies
{chr(10).join(f'- {dep}' for dep in context.dependencies) if context.dependencies else 'No specific dependencies identified'}

### Related Requirements
{chr(10).join(f'- {req}' for req in context.related_requirements) if context.related_requirements else 'No related requirements identified'}

---

## Implementation

{context.implementation_hints}

---

## Acceptance Criteria
{chr(10).join(f'- [ ] {criterion}' for criterion in context.acceptance_criteria) if context.acceptance_criteria else '- [ ] Implementation meets requirement specification'}

---

## Metadata
- **Requirement ID**: {req_id}
- **Type**: {requirement.get('type', 'functional')}
- **Complexity**: {requirement.get('complexity', 'medium')}
- **Priority**: {requirement.get('priority', 'medium')}
- **Estimated Tokens**: {context.estimated_tokens}
- **Context Strategy**: {context.overflow_strategy}

---

*This issue contains embedded context from project documentation. All necessary information for implementation is included above.*
"""
    
    @staticmethod
    def epic_with_context(epic: Dict, context: ContextWindow) -> str:
        """Generate epic issue with embedded context"""
        return f"""# Epic: {epic.get('title', 'Unknown Epic')}

{context.primary_requirement}

---

## Epic Overview

**Estimated Effort**: {epic.get('estimated_effort', 'TBD')}
**Story Count**: {len(epic.get('stories', []))}

### Context

{context.architecture_context}

### Implementation Strategy

{context.implementation_hints}

---

## Stories in this Epic
{chr(10).join(f'- [ ] {story_id}' for story_id in epic.get('stories', [])) if epic.get('stories') else 'Stories to be defined'}

---

## Acceptance Criteria
{chr(10).join(f'- [ ] {criterion}' for criterion in context.acceptance_criteria) if context.acceptance_criteria else '- [ ] All stories in epic completed successfully'}

---

**Epic ID**: {epic.get('id')}
**Estimated Tokens**: {context.estimated_tokens}
**Context Strategy**: {context.overflow_strategy}

---

*This epic contains embedded context and implementation guidance.*
"""
    
    @staticmethod
    def user_story_with_context(story: Dict, context: ContextWindow) -> str:
        """Generate user story with embedded context"""
        return f"""# User Story: {story.get('title', 'Unknown Story')}

{context.primary_requirement}

---

## Story Details

**Actor**: {story.get('actor', 'User')}
**Story Points**: {story.get('estimated_points', 'TBD')}

### Context

{context.design_context}
{context.architecture_context}

### Implementation

{context.implementation_hints}

---

## Acceptance Criteria
{chr(10).join(f'- [ ] {criterion}' for criterion in context.acceptance_criteria) if context.acceptance_criteria else '- [ ] Story requirements met'}

---

**Story ID**: {story.get('id')}
**Estimated Tokens**: {context.estimated_tokens}
**Context Strategy**: {context.overflow_strategy}

---

*This user story contains complete implementation context.*
"""


def main():
    """Main function for CLI usage"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Embed context into requirements for issue generation")
    parser.add_argument("--xml-file", help="Path to flattened documentation XML file")
    parser.add_argument("--requirements-file", help="Path to requirements JSON file")
    parser.add_argument("--requirement-id", help="Specific requirement ID to process")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--format", choices=['issue', 'json'], default='issue', help="Output format")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize context embedder
        embedder = ContextEmbedder(
            xml_file_path=args.xml_file,
            requirements_file_path=args.requirements_file
        )
        
        if args.requirement_id:
            # Process single requirement
            context = embedder.embed_context_for_requirement(args.requirement_id)
            requirement = embedder._get_requirement(args.requirement_id)
            
            if requirement:
                if args.format == 'issue':
                    output = ContextTemplates.issue_with_full_context(requirement, context)
                else:
                    output = json.dumps({
                        'requirement': requirement,
                        'context': {
                            'primary_requirement': context.primary_requirement,
                            'architecture_context': context.architecture_context,
                            'design_context': context.design_context,
                            'implementation_hints': context.implementation_hints,
                            'dependencies': context.dependencies,
                            'related_requirements': context.related_requirements,
                            'acceptance_criteria': context.acceptance_criteria,
                            'estimated_tokens': context.estimated_tokens,
                            'overflow_strategy': context.overflow_strategy,
                            'metadata': context.context_metadata
                        }
                    }, indent=2)
                
                if args.output:
                    Path(args.output).write_text(output, encoding='utf-8')
                    print(f"Context embedded and saved to: {args.output}")
                else:
                    print(output)
            else:
                print(f"Requirement {args.requirement_id} not found", file=sys.stderr)
                sys.exit(1)
        else:
            print("Please specify --requirement-id to process", file=sys.stderr)
            sys.exit(1)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()