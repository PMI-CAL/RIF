"""
Claude Code Knowledge MCP Server - Integration with Existing Knowledge Graph

This MCP server acts as a thin query interface over the existing RIF knowledge graph
system, providing accurate Claude Code capability information, compatibility checking,
and implementation pattern recommendations.

Architecture: Lightweight interface -> RIF Knowledge Graph (DuckDB + ChromaDB)
Dependencies: RIFDatabase, HybridKnowledgeSystem, VectorSearchEngine
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# MCP imports (would be installed separately)
try:
    from mcp import MCPServer, Tool, ResourcePattern
    from mcp.types import ToolRequest, ToolResponse
except ImportError:
    # Fallback for development/testing
    class MCPServer:
        def __init__(self, name: str, version: str): pass
        def add_tool(self, tool): pass
        def run(self): pass
    
    class Tool:
        def __init__(self, name: str, description: str, handler): pass
    
    class ToolRequest:
        def __init__(self, **kwargs): 
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class ToolResponse:
        def __init__(self, content: Any):
            self.content = content

# RIF Knowledge Graph imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from knowledge.database.database_interface import RIFDatabase
from knowledge.integration.hybrid_knowledge_system import HybridKnowledgeSystem
from knowledge.database.vector_search import VectorSearchResult, SearchQuery


@dataclass
class CompatibilityReport:
    """Compatibility analysis result."""
    compatible: bool
    confidence: float
    issues: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    alternatives: List[Dict[str, Any]]


@dataclass 
class PatternRecommendation:
    """Implementation pattern recommendation."""
    pattern_id: str
    name: str
    description: str
    technology: str
    task_type: str
    code_example: str
    confidence: float
    supporting_capabilities: List[str]


class ClaudeCodeKnowledgeServer:
    """
    MCP Server providing Claude Code capability knowledge through existing RIF knowledge graph.
    
    Integrates with:
    - RIFDatabase: Core database operations
    - HybridKnowledgeSystem: Advanced search and query capabilities  
    - VectorSearchEngine: Semantic similarity for pattern matching
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize MCP server
        self.server = MCPServer(
            name="Claude Code Knowledge Server",
            version="1.0.0"
        )
        
        # Initialize RIF knowledge graph connection
        self.rif_db = None
        self.hybrid_system = None
        self._setup_knowledge_graph()
        
        # Register MCP tools
        self._register_tools()
        
        self.logger.info("Claude Code Knowledge MCP Server initialized")
    
    def _setup_knowledge_graph(self):
        """Initialize connection to existing RIF knowledge graph."""
        try:
            # Use existing RIF database interface
            self.rif_db = RIFDatabase()
            
            # Initialize hybrid knowledge system for advanced queries
            hybrid_config = {
                'memory_limit_mb': 256,  # Lightweight for MCP server
                'cpu_cores': 1,
                'performance_mode': 'FAST',  # Optimize for query speed
                'database_path': 'knowledge/hybrid_knowledge.duckdb'
            }
            
            self.hybrid_system = HybridKnowledgeSystem(hybrid_config)
            if not self.hybrid_system.initialize():
                raise RuntimeError("Failed to initialize hybrid knowledge system")
            
            # Verify Claude Code knowledge is loaded
            self._verify_capability_data()
            
            self.logger.info("Connected to RIF knowledge graph successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup knowledge graph connection: {e}")
            raise
    
    def _verify_capability_data(self):
        """Verify Claude Code capability data is present in knowledge graph."""
        # Check for Claude Code capability entities
        capabilities = self.rif_db.search_entities(
            entity_types=['claude_capability', 'claude_limitation', 'implementation_pattern'],
            limit=10
        )
        
        if not capabilities:
            self.logger.warning("No Claude Code capability data found - may need seeding")
            # Could trigger automatic data seeding here
        else:
            self.logger.info(f"Found {len(capabilities)} Claude Code knowledge entities")
    
    def _register_tools(self):
        """Register MCP tools that provide Claude Code knowledge."""
        
        # Tool 1: Compatibility Checking
        check_compatibility_tool = Tool(
            name="check_compatibility",
            description="Validates proposed solution against Claude Code capabilities and limitations",
            handler=self._handle_check_compatibility
        )
        self.server.add_tool(check_compatibility_tool)
        
        # Tool 2: Pattern Recommendations  
        get_patterns_tool = Tool(
            name="get_patterns", 
            description="Returns correct implementation patterns for specific technology and task type",
            handler=self._handle_get_patterns
        )
        self.server.add_tool(get_patterns_tool)
        
        # Tool 3: Alternative Suggestions
        suggest_alternatives_tool = Tool(
            name="suggest_alternatives",
            description="Proposes compatible solutions when incompatible approach is detected", 
            handler=self._handle_suggest_alternatives
        )
        self.server.add_tool(suggest_alternatives_tool)
        
        # Tool 4: Architecture Validation
        validate_architecture_tool = Tool(
            name="validate_architecture",
            description="Reviews system design against Claude Code architectural constraints",
            handler=self._handle_validate_architecture
        )
        self.server.add_tool(validate_architecture_tool)
        
        # Tool 5: Limitation Queries
        get_limitations_tool = Tool(
            name="get_limitations",
            description="Returns known limitations for specific capability area",
            handler=self._handle_get_limitations
        )
        self.server.add_tool(get_limitations_tool)
        
        self.logger.info("Registered 5 MCP tools successfully")
    
    async def _handle_check_compatibility(self, request: ToolRequest) -> ToolResponse:
        """
        Check compatibility of proposed solution against Claude Code capabilities.
        
        Uses knowledge graph queries to:
        1. Find limitations that conflict with proposed approach
        2. Check capability requirements are available  
        3. Identify potential integration issues
        4. Generate compatibility report with recommendations
        """
        try:
            issue_description = request.arguments.get('issue_description', '')
            
            if not issue_description:
                return ToolResponse({
                    'error': 'issue_description parameter is required'
                })
            
            # Step 1: Extract key concepts from description using vector search
            concepts = await self._extract_concepts(issue_description)
            
            # Step 2: Find conflicting limitations using relationship queries
            conflicts = await self._find_conflicting_limitations(concepts)
            
            # Step 3: Check required capabilities are available
            capability_gaps = await self._check_capability_requirements(concepts)
            
            # Step 4: Generate compatibility report
            report = CompatibilityReport(
                compatible=len(conflicts) == 0 and len(capability_gaps) == 0,
                confidence=self._calculate_confidence(conflicts, capability_gaps),
                issues=conflicts + capability_gaps,
                recommendations=await self._generate_recommendations(conflicts, capability_gaps),
                alternatives=await self._find_alternatives(conflicts) if conflicts else []
            )
            
            return ToolResponse({
                'compatibility_report': {
                    'compatible': report.compatible,
                    'confidence': report.confidence,
                    'summary': f"Found {len(report.issues)} potential issues",
                    'issues': report.issues,
                    'recommendations': report.recommendations,
                    'alternatives': report.alternatives
                }
            })
            
        except Exception as e:
            self.logger.error(f"Compatibility check failed: {e}")
            return ToolResponse({'error': str(e)})
    
    async def _handle_get_patterns(self, request: ToolRequest) -> ToolResponse:
        """
        Get implementation patterns for specific technology and task type.
        
        Uses hybrid search (text + vector) to find relevant patterns:
        1. Text search for technology and task type
        2. Vector similarity for semantic pattern matching
        3. Relationship filtering for supported patterns only
        4. Ranking by relevance and confidence
        """
        try:
            technology = request.arguments.get('technology', '')
            task_type = request.arguments.get('task_type', '')
            limit = request.arguments.get('limit', 5)
            
            # Hybrid search combining text and semantic similarity
            search_query = f"{technology} {task_type}"
            
            # Use existing hybrid search capabilities
            results = self.rif_db.hybrid_search(
                text_query=search_query,
                entity_types=['implementation_pattern'],
                limit=limit
            )
            
            # Transform results into pattern recommendations
            patterns = []
            for result in results:
                entity = self.rif_db.get_entity(str(result.entity_id))
                if entity:
                    pattern = PatternRecommendation(
                        pattern_id=entity['id'],
                        name=entity['name'],
                        description=entity['metadata'].get('description', ''),
                        technology=entity['metadata'].get('technology', technology),
                        task_type=entity['metadata'].get('task_type', task_type),
                        code_example=entity['metadata'].get('code_example', ''),
                        confidence=result.similarity_score,
                        supporting_capabilities=await self._get_supporting_capabilities(entity['id'])
                    )
                    patterns.append(pattern)
            
            return ToolResponse({
                'patterns': [
                    {
                        'pattern_id': p.pattern_id,
                        'name': p.name,
                        'description': p.description,
                        'technology': p.technology,
                        'task_type': p.task_type,
                        'code_example': p.code_example,
                        'confidence': p.confidence,
                        'supporting_capabilities': p.supporting_capabilities
                    }
                    for p in patterns
                ]
            })
            
        except Exception as e:
            self.logger.error(f"Pattern search failed: {e}")
            return ToolResponse({'error': str(e)})
    
    async def _handle_suggest_alternatives(self, request: ToolRequest) -> ToolResponse:
        """
        Suggest compatible alternatives for incompatible approaches.
        
        Uses relationship traversal to:
        1. Find patterns marked as alternatives to incompatible approach
        2. Search for similar patterns using vector similarity
        3. Filter by compatibility with available capabilities
        4. Rank by contextual relevance
        """
        try:
            incompatible_approach = request.arguments.get('incompatible_approach', '')
            
            # Find alternative patterns using relationship traversal
            alternatives = await self._find_relationship_alternatives(incompatible_approach)
            
            # Also find similar compatible patterns using vector search
            similar_alternatives = await self._find_similar_alternatives(incompatible_approach)
            
            # Combine and deduplicate
            all_alternatives = self._merge_alternatives(alternatives, similar_alternatives)
            
            return ToolResponse({
                'alternatives': all_alternatives,
                'count': len(all_alternatives),
                'recommendation': 'Consider these compatible approaches instead'
            })
            
        except Exception as e:
            self.logger.error(f"Alternative suggestion failed: {e}")
            return ToolResponse({'error': str(e)})
    
    async def _handle_validate_architecture(self, request: ToolRequest) -> ToolResponse:
        """
        Validate system design against Claude Code architectural constraints.
        
        Multi-entity validation across:
        1. Orchestration patterns (check for unsupported Task() assumptions)  
        2. Integration patterns (verify MCP/subagent usage)
        3. State management (check for session scope limits)
        4. Performance patterns (validate against response time limits)
        """
        try:
            system_design = request.arguments.get('system_design', '')
            
            # Extract architectural components from design
            components = await self._extract_architectural_components(system_design)
            
            # Validate each component against architectural constraints
            validation_results = []
            
            for component in components:
                component_validation = await self._validate_component(component)
                validation_results.append(component_validation)
            
            # Generate overall architecture assessment
            overall_valid = all(result['valid'] for result in validation_results)
            
            return ToolResponse({
                'architecture_validation': {
                    'valid': overall_valid,
                    'component_results': validation_results,
                    'summary': f"Validated {len(components)} architectural components",
                    'recommendations': await self._generate_architectural_recommendations(validation_results)
                }
            })
            
        except Exception as e:
            self.logger.error(f"Architecture validation failed: {e}")
            return ToolResponse({'error': str(e)})
    
    async def _handle_get_limitations(self, request: ToolRequest) -> ToolResponse:
        """
        Get known limitations for specific capability area.
        
        Direct entity search with metadata filtering:
        1. Query limitation entities by category
        2. Filter by capability area
        3. Include workarounds and alternatives
        4. Sort by severity and relevance
        """
        try:
            capability_area = request.arguments.get('capability_area', '')
            
            # Search for limitation entities in the specified area
            limitations = self.rif_db.search_entities(
                query=capability_area,
                entity_types=['claude_limitation'],
                limit=20
            )
            
            # Format limitation information
            limitation_details = []
            for limitation in limitations:
                entity = self.rif_db.get_entity(limitation['id'])
                if entity:
                    detail = {
                        'limitation_id': entity['id'],
                        'name': entity['name'],
                        'category': entity['metadata'].get('category', ''),
                        'description': entity['metadata'].get('description', ''),
                        'severity': entity['metadata'].get('severity', 'medium'),
                        'workarounds': entity['metadata'].get('workarounds', []),
                        'alternatives': await self._get_limitation_alternatives(entity['id'])
                    }
                    limitation_details.append(detail)
            
            return ToolResponse({
                'limitations': limitation_details,
                'capability_area': capability_area,
                'count': len(limitation_details)
            })
            
        except Exception as e:
            self.logger.error(f"Limitation query failed: {e}")
            return ToolResponse({'error': str(e)})
    
    # Helper methods for knowledge graph queries
    
    async def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text using NLP or keyword extraction."""
        # Simplified implementation - in production could use more sophisticated NLP
        import re
        
        # Extract technical terms and concepts
        concepts = []
        
        # Look for technical patterns
        patterns = [
            r'Task\(\)',
            r'orchestrat\w+',
            r'agent\w*',
            r'MCP\s+server',
            r'subagent\w*',
            r'background\s+process',
            r'parallel\s+execution',
            r'state\s+management',
            r'hook\w*',
            r'GitHub\s+\w+'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                concepts.append(match.group().lower())
        
        return list(set(concepts))  # Deduplicate
    
    async def _find_conflicting_limitations(self, concepts: List[str]) -> List[Dict[str, Any]]:
        """Find limitations that conflict with the proposed concepts."""
        conflicts = []
        
        for concept in concepts:
            # Search for limitations related to this concept
            limitations = self.rif_db.search_entities(
                query=concept,
                entity_types=['claude_limitation'],
                limit=10
            )
            
            for limitation in limitations:
                entity = self.rif_db.get_entity(limitation['id'])
                if entity:
                    conflicts.append({
                        'type': 'limitation_conflict',
                        'concept': concept,
                        'limitation': entity['name'],
                        'description': entity['metadata'].get('description', ''),
                        'severity': entity['metadata'].get('severity', 'medium')
                    })
        
        return conflicts
    
    async def _check_capability_requirements(self, concepts: List[str]) -> List[Dict[str, Any]]:
        """Check if required capabilities are available for the concepts."""
        gaps = []
        
        # This would implement more sophisticated capability requirement checking
        # For now, simplified version
        known_unavailable = ['task_parallel_execution', 'background_scheduling', 'persistent_agents']
        
        for concept in concepts:
            if any(unavailable in concept for unavailable in known_unavailable):
                gaps.append({
                    'type': 'capability_gap',
                    'concept': concept,
                    'issue': f'Capability not available in Claude Code',
                    'severity': 'high'
                })
        
        return gaps
    
    def _calculate_confidence(self, conflicts: List, capability_gaps: List) -> float:
        """Calculate confidence score for compatibility assessment."""
        total_issues = len(conflicts) + len(capability_gaps)
        
        if total_issues == 0:
            return 1.0
        elif total_issues <= 2:
            return 0.7
        elif total_issues <= 5:
            return 0.4
        else:
            return 0.1
    
    async def _generate_recommendations(self, conflicts: List, capability_gaps: List) -> List[Dict[str, Any]]:
        """Generate recommendations to resolve compatibility issues."""
        recommendations = []
        
        for conflict in conflicts:
            recommendations.append({
                'type': 'avoid_pattern',
                'issue': conflict['limitation'],
                'recommendation': f"Avoid using {conflict['concept']} - not supported in Claude Code",
                'alternatives': await self._get_concept_alternatives(conflict['concept'])
            })
        
        for gap in capability_gaps:
            recommendations.append({
                'type': 'use_alternative', 
                'issue': gap['concept'],
                'recommendation': f"Use Claude Code native alternatives instead",
                'alternatives': await self._get_concept_alternatives(gap['concept'])
            })
        
        return recommendations
    
    async def _get_concept_alternatives(self, concept: str) -> List[str]:
        """Get alternative approaches for a concept."""
        # Simplified mapping - in production would query knowledge graph
        alternatives_map = {
            'task()': ['subagent delegation', 'MCP server integration'],
            'orchestration': ['hook-based automation', 'direct tool usage'],
            'background process': ['hook triggers', 'session-based workflow'],
            'parallel execution': ['sequential subagent calls', 'MCP coordination']
        }
        
        for key, alts in alternatives_map.items():
            if key in concept.lower():
                return alts
        
        return []
    
    async def _find_alternatives(self, conflicts: List) -> List[Dict[str, Any]]:
        """Find alternative patterns for conflicting approaches."""
        alternatives = []
        
        for conflict in conflicts:
            # Use relationship traversal to find alternatives
            relationships = self.rif_db.get_entity_relationships(
                conflict.get('entity_id', ''),
                direction='both'
            )
            
            for rel in relationships:
                if rel['relationship_type'] == 'alternative_to':
                    alternative_entity = self.rif_db.get_entity(rel['target_id'])
                    if alternative_entity:
                        alternatives.append({
                            'alternative_id': alternative_entity['id'],
                            'name': alternative_entity['name'],
                            'description': alternative_entity['metadata'].get('description', ''),
                            'confidence': rel['confidence']
                        })
        
        return alternatives
    
    async def _get_supporting_capabilities(self, pattern_id: str) -> List[str]:
        """Get capabilities that support a specific pattern."""
        relationships = self.rif_db.get_entity_relationships(pattern_id, direction='incoming')
        
        capabilities = []
        for rel in relationships:
            if rel['relationship_type'] == 'supports':
                capability = self.rif_db.get_entity(rel['source_id'])
                if capability and capability['type'] == 'claude_capability':
                    capabilities.append(capability['name'])
        
        return capabilities
    
    # Additional helper methods would be implemented here...
    
    def run(self):
        """Start the MCP server."""
        try:
            self.logger.info("Starting Claude Code Knowledge MCP Server...")
            self.server.run()
        except Exception as e:
            self.logger.error(f"Failed to start MCP server: {e}")
            raise
        finally:
            # Cleanup
            if self.hybrid_system:
                self.hybrid_system.shutdown()
            if self.rif_db:
                self.rif_db.close()


def main():
    """Main entry point for the MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Claude Code Knowledge MCP Server')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Create and run server
    server = ClaudeCodeKnowledgeServer(config)
    server.run()


if __name__ == '__main__':
    main()