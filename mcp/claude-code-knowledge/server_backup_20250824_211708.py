#!/usr/bin/env python3
"""
Claude Code Knowledge MCP Server - Phase 2 Implementation

A lightweight MCP server providing Claude Code capability knowledge through 
the existing RIF knowledge graph system. This server acts as a read-only 
query interface over the extended knowledge graph from Phase 1.

Architecture:
- JSON-RPC 2.0 protocol compliance
- Direct integration with RIFDatabase 
- Semantic search capabilities
- <200ms query response target
- Zero impact on existing RIF operations

Features:
- 5 core tools for Claude Code knowledge
- Connection to extended knowledge graph from Phase 1
- Vector similarity search for pattern matching
- Comprehensive error handling and graceful degradation
- Performance monitoring and caching
"""

import json
import asyncio
import logging
import sys
import os
import warnings
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from datetime import datetime, timezone

# Suppress all warnings before any imports
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add RIF root to Python path
rif_root = Path(__file__).parents[2]
sys.path.insert(0, str(rif_root))

# Suppress stderr during imports
old_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
try:
    from knowledge.database.database_interface import RIFDatabase
    from knowledge.database.vector_search import VectorSearchResult, SearchQuery
except ImportError as e:
    sys.stderr = old_stderr
    sys.exit(1)
finally:
    sys.stderr = old_stderr


@dataclass
class MCPRequest:
    """MCP request structure following JSON-RPC 2.0 spec."""
    id: Union[str, int]
    method: str
    params: Dict[str, Any]
    jsonrpc: str = "2.0"


@dataclass 
class MCPResponse:
    """MCP response structure following JSON-RPC 2.0 spec."""
    id: Union[str, int]
    jsonrpc: str = "2.0"
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


@dataclass
class CompatibilityCheck:
    """Compatibility analysis result."""
    compatible: bool
    confidence: float
    issues: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    execution_time_ms: float


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
    supporting_tools: List[str]
    usage_count: int


@dataclass
class ArchitectureValidation:
    """Architecture validation result."""
    valid: bool
    components_analyzed: int
    issues_found: List[Dict[str, Any]]
    recommendations: List[str]
    confidence: float


class QueryCache:
    """Simple in-memory cache for frequent queries."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl_seconds:
                return entry['data']
            else:
                del self.cache[key]
        return None
        
    def set(self, key: str, data: Any):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
            
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }


class ClaudeCodeKnowledgeServer:
    """
    MCP Server providing Claude Code capability knowledge.
    
    Implements 5 core tools:
    1. check_compatibility - Validate approach compatibility
    2. recommend_pattern - Suggest implementation patterns  
    3. find_alternatives - Provide alternative approaches
    4. validate_architecture - Check design alignment
    5. query_limitations - Get specific limitations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Initialize components
        self.rif_db: Optional[RIFDatabase] = None
        self.query_cache = QueryCache(
            max_size=self.config.get('cache_size', 100),
            ttl_seconds=self.config.get('cache_ttl', 300)
        )
        
        # Performance tracking
        self.request_count = 0
        self.total_response_time = 0.0
        self.error_count = 0
        
        # Tool registry
        self.tools = {
            "check_compatibility": self._check_compatibility,
            "recommend_pattern": self._recommend_pattern, 
            "find_alternatives": self._find_alternatives,
            "validate_architecture": self._validate_architecture,
            "query_limitations": self._query_limitations
        }
        
        self.logger.info("Claude Code Knowledge MCP Server initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        
        if not logger.handlers:
            # For MCP, log to file only, never to stderr
            if self.config.get('log_to_file', False):
                log_dir = Path(__file__).parent / 'logs'
                log_dir.mkdir(exist_ok=True)
                handler = logging.FileHandler(log_dir / 'server.log')
            else:
                # Use NullHandler to suppress all output
                handler = logging.NullHandler()
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        log_level = self.config.get('log_level', 'INFO')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        return logger
    
    async def initialize(self) -> bool:
        """Initialize connection to RIF knowledge graph."""
        try:
            self.logger.info("Initializing connection to RIF knowledge graph...")
            
            # Initialize RIF database interface
            self.rif_db = RIFDatabase()
            
            # Verify Claude Code knowledge is available
            await self._verify_claude_knowledge()
            
            self.logger.info("Successfully connected to RIF knowledge graph")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize knowledge graph connection: {e}")
            if self.config.get('graceful_degradation', True):
                self.logger.info("Enabling graceful degradation mode")
                # Initialize graceful degradation handler
                from safety import GracefulDegradation
                self.graceful_handler = GracefulDegradation()
                return True  # Continue operating with fallback responses
            return False
    
    async def _verify_claude_knowledge(self):
        """Verify Claude Code knowledge entities are present."""
        claude_entities = self.rif_db.search_entities(
            entity_types=['claude_capability', 'claude_limitation', 'implementation_pattern'],
            limit=5
        )
        
        if not claude_entities:
            self.logger.warning("No Claude Code knowledge found - may need to run Phase 1 seeding")
            raise RuntimeError("Claude Code knowledge not found in database")
        
        self.logger.info(f"Found {len(claude_entities)} Claude Code knowledge entities")
    
    async def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP request."""
        start_time = time.time()
        
        try:
            # Parse request
            request = MCPRequest(**request_data)
            self.request_count += 1
            
            self.logger.debug(f"Handling request: {request.method}")
            
            # Route to appropriate tool
            if request.method in self.tools:
                result = await self.tools[request.method](request.params)
                
                response = MCPResponse(
                    id=request.id,
                    result=result
                )
            else:
                response = MCPResponse(
                    id=request.id,
                    error={
                        "code": -32601,
                        "message": f"Method not found: {request.method}",
                        "data": {"available_methods": list(self.tools.keys())}
                    }
                )
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Request handling failed: {e}")
            
            response = MCPResponse(
                id=request_data.get('id', 0),
                error={
                    "code": -32000,
                    "message": "Internal server error",
                    "data": {"error": str(e)}
                }
            )
        
        # Track performance
        response_time = (time.time() - start_time) * 1000  # ms
        self.total_response_time += response_time
        
        if response_time > 200:  # Target <200ms
            self.logger.warning(f"Slow response: {response_time:.1f}ms for {request_data.get('method')}")
        
        return asdict(response)
    
    async def _check_compatibility(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool 1: Check compatibility of proposed solution.
        
        Validates approach against Claude Code capabilities and limitations.
        Returns compatibility report with issues and recommendations.
        """
        start_time = time.time()
        
        try:
            issue_description = params.get('issue_description', '')
            approach = params.get('approach', '')
            
            if not issue_description:
                return {"error": "issue_description parameter is required"}
            
            # Check if we have a working knowledge graph
            if not self.rif_db or hasattr(self, 'graceful_handler'):
                # Use graceful degradation
                if hasattr(self, 'graceful_handler'):
                    return self.graceful_handler.get_fallback_response("check_compatibility", params)
                else:
                    return {"error": "Knowledge graph not available and graceful degradation not initialized"}
            
            # Create cache key
            cache_key = f"compat:{hash(issue_description + approach)}"
            cached_result = self.query_cache.get(cache_key)
            if cached_result:
                self.logger.debug("Returning cached compatibility result")
                return cached_result
            
            # Extract technical concepts from description
            concepts = await self._extract_technical_concepts(issue_description + " " + approach)
            
            # Find conflicting limitations
            conflicts = []
            for concept in concepts:
                concept_conflicts = await self._find_concept_conflicts(concept)
                conflicts.extend(concept_conflicts)
            
            # Check capability requirements
            capability_gaps = await self._check_capability_gaps(concepts)
            
            # Calculate compatibility
            total_issues = len(conflicts) + len(capability_gaps)
            compatible = total_issues == 0
            confidence = max(0.1, 1.0 - (total_issues * 0.2))
            
            # Generate recommendations
            recommendations = []
            if conflicts:
                recommendations.extend(await self._generate_conflict_recommendations(conflicts))
            if capability_gaps:
                recommendations.extend(await self._generate_capability_recommendations(capability_gaps))
            
            result = {
                "compatible": compatible,
                "confidence": confidence,
                "concepts_analyzed": len(concepts),
                "issues": conflicts + capability_gaps,
                "recommendations": recommendations,
                "execution_time_ms": (time.time() - start_time) * 1000
            }
            
            # Cache result
            self.query_cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Compatibility check failed: {e}")
            # Try graceful degradation on error
            if hasattr(self, 'graceful_handler'):
                return self.graceful_handler.get_fallback_response("check_compatibility", params)
            return {"error": f"Compatibility check failed: {str(e)}"}
    
    async def _recommend_pattern(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool 2: Recommend implementation patterns.
        
        Returns suitable patterns for given technology and task type.
        Uses vector similarity for semantic pattern matching.
        """
        try:
            technology = params.get('technology', '')
            task_type = params.get('task_type', '')
            limit = min(params.get('limit', 5), 10)  # Cap at 10
            
            # Create search query
            search_query = f"{technology} {task_type}"
            
            # Use hybrid search for patterns
            results = self.rif_db.hybrid_search(
                text_query=search_query,
                entity_types=['implementation_pattern'],
                limit=limit * 2  # Get extra for filtering
            )
            
            # Transform and rank results
            patterns = []
            for result in results[:limit]:
                entity = self.rif_db.get_entity(str(result.entity_id))
                if entity:
                    # Get supporting tools
                    supporting_tools = await self._get_supporting_tools(entity['id'])
                    
                    pattern = PatternRecommendation(
                        pattern_id=entity['id'],
                        name=entity['name'],
                        description=entity.get('metadata', {}).get('description', ''),
                        technology=entity.get('metadata', {}).get('technology', technology),
                        task_type=entity.get('metadata', {}).get('task_type', task_type),
                        code_example=entity.get('metadata', {}).get('code_example', ''),
                        confidence=result.similarity_score,
                        supporting_tools=supporting_tools,
                        usage_count=entity.get('metadata', {}).get('usage_count', 0)
                    )
                    patterns.append(asdict(pattern))
            
            return {
                "patterns": patterns,
                "search_query": search_query,
                "total_found": len(patterns)
            }
            
        except Exception as e:
            self.logger.error(f"Pattern recommendation failed: {e}")
            return {"error": f"Pattern recommendation failed: {str(e)}"}
    
    async def _find_alternatives(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool 3: Find alternatives to problematic approaches.
        
        Uses relationship traversal and vector similarity to find alternatives.
        """
        try:
            problematic_approach = params.get('problematic_approach', '')
            
            if not problematic_approach:
                return {"error": "problematic_approach parameter is required"}
            
            # Find relationship-based alternatives
            relationship_alternatives = await self._find_relationship_alternatives(problematic_approach)
            
            # Find similarity-based alternatives
            similarity_alternatives = await self._find_similarity_alternatives(problematic_approach)
            
            # Merge and deduplicate
            all_alternatives = {}
            
            # Add relationship alternatives with higher weight
            for alt in relationship_alternatives:
                alt['source'] = 'relationship'
                alt['confidence'] *= 1.2  # Boost relationship-based matches
                all_alternatives[alt['id']] = alt
            
            # Add similarity alternatives
            for alt in similarity_alternatives:
                if alt['id'] not in all_alternatives:
                    alt['source'] = 'similarity'
                    all_alternatives[alt['id']] = alt
            
            # Sort by confidence
            sorted_alternatives = sorted(
                all_alternatives.values(), 
                key=lambda x: x['confidence'], 
                reverse=True
            )
            
            return {
                "alternatives": sorted_alternatives[:5],  # Top 5
                "total_found": len(all_alternatives),
                "search_approach": problematic_approach
            }
            
        except Exception as e:
            self.logger.error(f"Alternative finding failed: {e}")
            return {"error": f"Alternative finding failed: {str(e)}"}
    
    async def _validate_architecture(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool 4: Validate architecture design.
        
        Checks system design against Claude Code architectural constraints.
        """
        try:
            system_design = params.get('system_design', '')
            
            if not system_design:
                return {"error": "system_design parameter is required"}
            
            # Extract architectural components
            components = await self._extract_architectural_components(system_design)
            
            # Validate each component
            validation_results = []
            issues_found = []
            
            for component in components:
                component_result = await self._validate_component(component)
                validation_results.append(component_result)
                
                if not component_result['valid']:
                    issues_found.extend(component_result.get('issues', []))
            
            # Calculate overall validation
            valid_components = sum(1 for r in validation_results if r['valid'])
            overall_valid = len(issues_found) == 0
            confidence = valid_components / len(validation_results) if validation_results else 0.0
            
            # Generate recommendations
            recommendations = await self._generate_architectural_recommendations(validation_results)
            
            return {
                "valid": overall_valid,
                "confidence": confidence,
                "components_analyzed": len(components),
                "validation_results": validation_results,
                "issues_found": issues_found,
                "recommendations": recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Architecture validation failed: {e}")
            return {"error": f"Architecture validation failed: {str(e)}"}
    
    async def _query_limitations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool 5: Query specific limitations.
        
        Returns known limitations for capability area with workarounds.
        """
        try:
            capability_area = params.get('capability_area', '')
            severity_filter = params.get('severity', '')  # optional filter
            
            if not capability_area:
                return {"error": "capability_area parameter is required"}
            
            # Search for limitations
            limitations = self.rif_db.search_entities(
                query=capability_area,
                entity_types=['claude_limitation'],
                limit=20
            )
            
            # Format limitation details
            limitation_details = []
            for limitation in limitations:
                entity = self.rif_db.get_entity(limitation['id'])
                if entity:
                    metadata = entity.get('metadata', {})
                    severity = metadata.get('severity', 'medium')
                    
                    # Apply severity filter if specified
                    if severity_filter and severity != severity_filter:
                        continue
                    
                    # Get alternative approaches
                    alternatives = await self._get_limitation_alternatives(entity['id'])
                    
                    detail = {
                        'limitation_id': entity['id'],
                        'name': entity['name'],
                        'category': metadata.get('category', ''),
                        'description': metadata.get('description', ''),
                        'severity': severity,
                        'impact': metadata.get('impact', ''),
                        'workarounds': metadata.get('workarounds', []),
                        'alternatives': alternatives,
                        'documentation_link': metadata.get('documentation_link', '')
                    }
                    limitation_details.append(detail)
            
            # Sort by severity (high -> medium -> low)
            severity_order = {'high': 3, 'medium': 2, 'low': 1}
            limitation_details.sort(
                key=lambda x: severity_order.get(x['severity'], 0), 
                reverse=True
            )
            
            return {
                "limitations": limitation_details,
                "capability_area": capability_area,
                "total_found": len(limitation_details),
                "severity_filter": severity_filter or "all"
            }
            
        except Exception as e:
            self.logger.error(f"Limitation query failed: {e}")
            return {"error": f"Limitation query failed: {str(e)}"}
    
    # Helper methods for knowledge graph operations
    
    async def _extract_technical_concepts(self, text: str) -> List[str]:
        """Extract technical concepts from text."""
        import re
        
        concepts = set()
        
        # Common Claude Code patterns to detect
        patterns = [
            r'Task\(\)',
            r'orchestrat\w+',
            r'agent\s+\w+',
            r'MCP\s+server',
            r'subagent',
            r'background\s+process',
            r'parallel\s+execution', 
            r'state\s+management',
            r'hook\w*',
            r'GitHub\s+\w+',
            r'Tool\(\)',
            r'Read\(\)',
            r'Write\(\)',
            r'Edit\(\)',
            r'Bash\(\)'
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            concepts.update(matches)
        
        # Also extract general technical terms
        tech_keywords = [
            'api', 'database', 'server', 'client', 'framework',
            'library', 'service', 'component', 'module', 'function'
        ]
        
        words = re.findall(r'\b\w+\b', text_lower)
        for word in words:
            if word in tech_keywords:
                concepts.add(word)
        
        return list(concepts)
    
    async def _find_concept_conflicts(self, concept: str) -> List[Dict[str, Any]]:
        """Find limitations that conflict with a concept."""
        conflicts = []
        
        # Search for relevant limitations
        limitations = self.rif_db.search_entities(
            query=concept,
            entity_types=['claude_limitation'],
            limit=10
        )
        
        for limitation in limitations:
            entity = self.rif_db.get_entity(limitation['id'])
            if entity:
                metadata = entity.get('metadata', {})
                conflicts.append({
                    'type': 'limitation_conflict',
                    'concept': concept,
                    'limitation_id': entity['id'],
                    'limitation_name': entity['name'],
                    'description': metadata.get('description', ''),
                    'severity': metadata.get('severity', 'medium'),
                    'category': metadata.get('category', 'general')
                })
        
        return conflicts
    
    async def _check_capability_gaps(self, concepts: List[str]) -> List[Dict[str, Any]]:
        """Check for capability gaps in the concepts."""
        gaps = []
        
        # Known problematic patterns for Claude Code
        problematic_patterns = {
            'task()': 'Task() orchestration not supported - use direct tool calls',
            'background process': 'Background processes not supported - use hooks',
            'persistent state': 'Persistent state limited to session scope',
            'parallel execution': 'True parallel execution not available',
            'file system persistence': 'File system changes require explicit tools'
        }
        
        for concept in concepts:
            for pattern, issue in problematic_patterns.items():
                if pattern in concept.lower():
                    gaps.append({
                        'type': 'capability_gap',
                        'concept': concept,
                        'issue': issue,
                        'severity': 'high'
                    })
        
        return gaps
    
    async def _generate_conflict_recommendations(self, conflicts: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for resolving conflicts."""
        recommendations = []
        
        for conflict in conflicts:
            concept = conflict['concept']
            severity = conflict['severity']
            
            if 'task()' in concept.lower():
                recommendations.append(
                    "Replace Task() orchestration with direct tool usage and subagent delegation"
                )
            elif 'orchestrat' in concept.lower():
                recommendations.append(
                    "Use hook-based automation instead of centralized orchestration"
                )
            elif 'background' in concept.lower():
                recommendations.append(
                    "Replace background processes with event-driven hook triggers"
                )
            else:
                recommendations.append(
                    f"Review {concept} approach - conflicts with Claude Code {severity} limitation"
                )
        
        return list(set(recommendations))  # Deduplicate
    
    async def _generate_capability_recommendations(self, gaps: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for capability gaps."""
        recommendations = []
        
        for gap in gaps:
            concept = gap['concept']
            issue = gap['issue']
            
            recommendations.append(f"For {concept}: {issue}")
        
        return recommendations
    
    async def _get_supporting_tools(self, pattern_id: str) -> List[str]:
        """Get tools that support a specific pattern."""
        relationships = self.rif_db.get_entity_relationships(pattern_id, direction='incoming')
        
        tools = []
        for rel in relationships:
            if rel['relationship_type'] == 'supports':
                tool_entity = self.rif_db.get_entity(rel['source_id'])
                if tool_entity and tool_entity['type'] == 'claude_tool':
                    tools.append(tool_entity['name'])
        
        return tools
    
    async def _find_relationship_alternatives(self, approach: str) -> List[Dict[str, Any]]:
        """Find alternatives using relationship traversal."""
        alternatives = []
        
        # First find entities related to the problematic approach
        related_entities = self.rif_db.search_entities(
            query=approach,
            entity_types=['anti_pattern', 'implementation_pattern'],
            limit=10
        )
        
        for entity_data in related_entities:
            entity = self.rif_db.get_entity(entity_data['id'])
            if entity:
                # Find alternative relationships
                relationships = self.rif_db.get_entity_relationships(entity['id'])
                
                for rel in relationships:
                    if rel['relationship_type'] == 'alternative_to':
                        alternative_entity = self.rif_db.get_entity(rel['target_id'])
                        if alternative_entity:
                            alternatives.append({
                                'id': alternative_entity['id'],
                                'name': alternative_entity['name'],
                                'description': alternative_entity.get('metadata', {}).get('description', ''),
                                'confidence': rel.get('confidence', 0.8),
                                'technology': alternative_entity.get('metadata', {}).get('technology', '')
                            })
        
        return alternatives
    
    async def _find_similarity_alternatives(self, approach: str) -> List[Dict[str, Any]]:
        """Find alternatives using vector similarity."""
        alternatives = []
        
        # Use vector search to find similar patterns
        results = self.rif_db.hybrid_search(
            text_query=f"alternative to {approach}",
            entity_types=['implementation_pattern'],
            limit=10
        )
        
        for result in results:
            entity = self.rif_db.get_entity(str(result.entity_id))
            if entity:
                alternatives.append({
                    'id': entity['id'],
                    'name': entity['name'],
                    'description': entity.get('metadata', {}).get('description', ''),
                    'confidence': result.similarity_score,
                    'technology': entity.get('metadata', {}).get('technology', '')
                })
        
        return alternatives
    
    async def _extract_architectural_components(self, design: str) -> List[Dict[str, Any]]:
        """Extract architectural components from system design."""
        components = []
        
        # Simple component extraction - could be enhanced with NLP
        import re
        
        # Look for common architectural patterns
        patterns = {
            'orchestrator': r'orchestrator|coordinator|manager',
            'agent': r'agent|service|worker',
            'database': r'database|storage|repository',
            'api': r'api|endpoint|interface',
            'queue': r'queue|buffer|pipeline',
            'cache': r'cache|memory|store'
        }
        
        design_lower = design.lower()
        for component_type, pattern in patterns.items():
            matches = re.findall(pattern, design_lower)
            if matches:
                components.append({
                    'type': component_type,
                    'mentions': len(matches),
                    'context': design[:200]  # First 200 chars for context
                })
        
        return components
    
    async def _validate_component(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single architectural component."""
        component_type = component['type']
        issues = []
        
        # Component-specific validation rules
        if component_type == 'orchestrator':
            issues.append({
                'issue': 'Orchestrator patterns not supported in Claude Code',
                'recommendation': 'Use direct tool calls and hook-based automation',
                'severity': 'high'
            })
        elif component_type == 'queue':
            issues.append({
                'issue': 'Persistent queues not available',
                'recommendation': 'Use session-based processing or MCP coordination',
                'severity': 'medium'
            })
        
        return {
            'component_type': component_type,
            'valid': len(issues) == 0,
            'issues': issues,
            'mentions': component.get('mentions', 0)
        }
    
    async def _generate_architectural_recommendations(self, validation_results: List[Dict[str, Any]]) -> List[str]:
        """Generate architectural recommendations based on validation results."""
        recommendations = []
        
        for result in validation_results:
            if not result['valid']:
                for issue in result['issues']:
                    recommendations.append(issue['recommendation'])
        
        # Add general recommendations
        if any(not r['valid'] for r in validation_results):
            recommendations.extend([
                "Consider using MCP servers for complex integrations",
                "Leverage Claude Code's built-in tools instead of custom orchestration",
                "Use hook-based automation for workflow management"
            ])
        
        return list(set(recommendations))  # Deduplicate
    
    async def _get_limitation_alternatives(self, limitation_id: str) -> List[Dict[str, Any]]:
        """Get alternative approaches for a limitation."""
        relationships = self.rif_db.get_entity_relationships(limitation_id)
        
        alternatives = []
        for rel in relationships:
            if rel['relationship_type'] == 'alternative_to':
                alternative = self.rif_db.get_entity(rel['target_id'])
                if alternative:
                    alternatives.append({
                        'id': alternative['id'],
                        'name': alternative['name'],
                        'description': alternative.get('metadata', {}).get('description', ''),
                        'confidence': rel.get('confidence', 0.7)
                    })
        
        return alternatives
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server performance statistics."""
        avg_response_time = (
            self.total_response_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            'request_count': self.request_count,
            'error_count': self.error_count,
            'avg_response_time_ms': avg_response_time,
            'cache_size': len(self.query_cache.cache),
            'available_tools': list(self.tools.keys()),
            'uptime_seconds': time.time() - getattr(self, 'start_time', time.time())
        }
    
    async def shutdown(self):
        """Cleanup resources."""
        try:
            if self.rif_db:
                # Close database connections
                # self.rif_db.close()  # If this method exists
                pass
            
            self.logger.info("Claude Code Knowledge MCP Server shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


def create_server_config() -> Dict[str, Any]:
    """Create default server configuration."""
    return {
        'log_level': 'INFO',
        'cache_size': 100,
        'cache_ttl': 300,  # 5 minutes
        'max_request_size': 1024 * 1024,  # 1MB
        'timeout_seconds': 30
    }


async def main():
    """Main entry point for the MCP server."""
    import argparse
    import warnings
    
    # Suppress all warnings that could pollute stdout
    warnings.filterwarnings('ignore')
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    parser = argparse.ArgumentParser(description='Claude Code Knowledge MCP Server')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = create_server_config()
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    if args.debug:
        config['log_level'] = 'DEBUG'
    
    # Disable all stdout/stderr output during initialization
    config['log_level'] = 'ERROR'  # Only log errors
    config['log_to_file'] = True   # Log to file instead of stderr
    
    # Create and initialize server
    server = ClaudeCodeKnowledgeServer(config)
    
    if not await server.initialize():
        # Silent failure - no stderr output for MCP protocol
        return 1
    
    server.start_time = time.time()
    
    try:
        # MCP Protocol: Read JSON-RPC from stdin, write to stdout
        while True:
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            
            try:
                request = json.loads(line.strip())
                
                # Handle special MCP protocol methods
                if request.get('method') == 'initialize':
                    response = {
                        'jsonrpc': '2.0',
                        'id': request.get('id'),
                        'result': {
                            'protocolVersion': '2024-11-05',
                            'capabilities': {
                                'tools': {}
                            },
                            'serverInfo': {
                                'name': 'claude-code-knowledge',
                                'version': '1.0.0'
                            }
                        }
                    }
                elif request.get('method') == 'tools/list':
                    response = {
                        'jsonrpc': '2.0',
                        'id': request.get('id'),
                        'result': {
                            'tools': [
                                {
                                    'name': tool_name,
                                    'description': f'Claude Code knowledge tool: {tool_name}',
                                    'inputSchema': {'type': 'object'}
                                }
                                for tool_name in server.tools.keys()
                            ]
                        }
                    }
                elif request.get('method') == 'tools/call':
                    # Delegate to server's handle_request
                    response = await server.handle_request(request)
                else:
                    # Let server handle other requests
                    response = await server.handle_request(request)
                
                # Add JSON-RPC fields if not present
                if 'jsonrpc' not in response:
                    response['jsonrpc'] = '2.0'
                if 'id' not in response:
                    response['id'] = request.get('id')
                
                # Write response to stdout
                print(json.dumps(response))
                sys.stdout.flush()
                
            except json.JSONDecodeError:
                error_response = {
                    'jsonrpc': '2.0',
                    'id': None,
                    'error': {
                        'code': -32700,
                        'message': 'Parse error'
                    }
                }
                print(json.dumps(error_response))
                sys.stdout.flush()
            except Exception as e:
                error_response = {
                    'jsonrpc': '2.0',
                    'id': request.get('id') if 'request' in locals() else None,
                    'error': {
                        'code': -32603,
                        'message': f'Internal error: {str(e)}'
                    }
                }
                print(json.dumps(error_response))
                sys.stdout.flush()
            
    except KeyboardInterrupt:
        pass  # Silent shutdown
    finally:
        await server.shutdown()
    
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))