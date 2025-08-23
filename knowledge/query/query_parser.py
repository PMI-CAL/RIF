"""
Query Parser for Hybrid Search System - Issue #33
Parses natural language queries into structured search plans
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum


class QueryIntent(Enum):
    """Primary query intent classification"""
    ENTITY_SEARCH = "entity_search"          # Find specific functions, classes, etc.
    SIMILARITY_SEARCH = "similarity_search"  # Find similar code patterns
    DEPENDENCY_ANALYSIS = "dependency_analysis"  # What calls/uses this
    IMPACT_ANALYSIS = "impact_analysis"      # What breaks if changed
    HYBRID_SEARCH = "hybrid_search"          # Complex multi-modal search


class SearchStrategy(Enum):
    """Search execution strategy"""
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID_PARALLEL = "hybrid_parallel"
    SEQUENTIAL = "sequential"


@dataclass
class QueryEntity:
    """Extracted entity from natural language query"""
    name: str
    entity_type: Optional[str] = None  # function, class, module, etc.
    file_path: Optional[str] = None
    confidence: float = 1.0


@dataclass
class QueryFilters:
    """Filters to apply during search"""
    entity_types: Set[str] = field(default_factory=set)
    file_patterns: List[str] = field(default_factory=list)
    complexity_levels: Set[str] = field(default_factory=set)
    time_range: Optional[Tuple[str, str]] = None  # (start, end) timestamps


@dataclass
class QueryIntent:
    """Structured representation of query intent"""
    primary_intent: QueryIntent
    secondary_intents: List[QueryIntent] = field(default_factory=list)
    entities: List[QueryEntity] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    filters: QueryFilters = field(default_factory=QueryFilters)
    confidence: float = 1.0
    
    # Search strategy hints
    requires_semantic_search: bool = False
    requires_structural_search: bool = False
    requires_exact_match: bool = False


@dataclass
class StructuredQuery:
    """Final structured query for execution"""
    original_query: str
    intent: QueryIntent
    execution_strategy: SearchStrategy
    vector_query: Optional[str] = None
    graph_query: Optional[Dict[str, Any]] = None
    direct_lookup: Optional[Dict[str, Any]] = None
    result_limit: int = 20
    timeout_ms: int = 5000


class QueryParser:
    """
    Natural language query parser for hybrid search system.
    
    Converts natural language queries into structured search plans that can be
    executed by the hybrid search engine.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Query pattern definitions
        self.intent_patterns = {
            QueryIntent.ENTITY_SEARCH: [
                r'\b(find|show|get|locate|search for)\s+(function|class|module|variable|method)\s+(\w+)',
                r'\b(where is|what is)\s+(\w+)',
                r'\b(find|locate)\s+(\w+)\s+(function|class|method)',
                r'^(\w+)$',  # Single word entity search
            ],
            QueryIntent.SIMILARITY_SEARCH: [
                r'\b(similar|like|patterns?|examples?)\b',
                r'\b(find\s+code\s+like|show\s+examples?)\b',
                r'\b(compare|match|resembles?)\b',
                r'\b(other\s+functions?\s+that|alternatives?)\b',
            ],
            QueryIntent.DEPENDENCY_ANALYSIS: [
                r'\b(what\s+(calls|uses|imports|depends\s+on))\b',
                r'\b(who\s+(calls|uses|references))\b',
                r'\b(dependencies?\s+of|depends?\s+on)\b',
                r'\b(calls?\s+to|references?\s+to)\b',
            ],
            QueryIntent.IMPACT_ANALYSIS: [
                r'\b(what\s+(breaks|fails)\s+if)\b',
                r'\b(impact\s+of\s+chang(ing|e))\b',
                r'\b(affected\s+by|influences?)\b',
                r'\b(side\s+effects?\s+of)\b',
            ]
        }
        
        # Entity type patterns
        self.entity_type_patterns = {
            'function': [r'\b(function|method|def|func)\b', r'\(\)'],
            'class': [r'\b(class|type|object)\b'],
            'module': [r'\b(module|file|script)\b', r'\.py\b', r'\.js\b'],
            'variable': [r'\b(variable|var|const|let)\b'],
            'interface': [r'\b(interface|contract|api)\b'],
            'enum': [r'\b(enum|enumeration|constants?)\b'],
        }
        
        # File pattern indicators  
        self.file_patterns = {
            r'\.py$': 'python',
            r'\.js$': 'javascript', 
            r'\.ts$': 'typescript',
            r'\.java$': 'java',
            r'\.go$': 'golang',
            r'\.rs$': 'rust',
        }
        
        # Semantic search indicators
        self.semantic_indicators = [
            'similar', 'like', 'patterns', 'examples', 'alternatives',
            'comparable', 'equivalent', 'matching', 'resembling'
        ]
        
        # Structural search indicators
        self.structural_indicators = [
            'calls', 'uses', 'imports', 'depends', 'references', 
            'extends', 'implements', 'inherits', 'breaks', 'affects'
        ]
    
    def parse_query(self, query: str) -> StructuredQuery:
        """
        Parse natural language query into structured format.
        
        Args:
            query: Natural language query string
            
        Returns:
            StructuredQuery with execution plan
        """
        self.logger.debug(f"Parsing query: {query}")
        
        # Clean and normalize query
        normalized_query = self._normalize_query(query)
        
        # Extract query intent
        intent = self._extract_intent(normalized_query)
        
        # Determine execution strategy
        strategy = self._select_execution_strategy(intent)
        
        # Generate execution queries
        vector_query = self._generate_vector_query(intent, normalized_query)
        graph_query = self._generate_graph_query(intent)
        direct_lookup = self._generate_direct_lookup(intent)
        
        structured_query = StructuredQuery(
            original_query=query,
            intent=intent,
            execution_strategy=strategy,
            vector_query=vector_query,
            graph_query=graph_query, 
            direct_lookup=direct_lookup,
            result_limit=self._estimate_result_limit(intent),
            timeout_ms=self._estimate_timeout(intent, strategy)
        )
        
        self.logger.debug(f"Parsed query strategy: {strategy.value}")
        return structured_query
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query text for processing"""
        # Convert to lowercase for pattern matching
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Handle common query prefixes
        normalized = re.sub(r'^(can you |please |help me )', '', normalized)
        
        return normalized
    
    def _extract_intent(self, query: str) -> QueryIntent:
        """Extract structured intent from normalized query"""
        # Initialize intent structure
        intent = QueryIntent(
            primary_intent=QueryIntent.ENTITY_SEARCH,  # Default
            entities=[],
            concepts=[],
            filters=QueryFilters()
        )
        
        # Classify primary intent
        intent_scores = {}
        for query_intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches
            intent_scores[query_intent] = score
        
        # Select primary intent
        if intent_scores:
            intent.primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            intent.confidence = max(intent_scores.values()) / len(query.split())
        
        # Extract entities
        intent.entities = self._extract_entities(query)
        
        # Extract concepts (keywords that aren't entities)
        intent.concepts = self._extract_concepts(query, intent.entities)
        
        # Extract filters
        intent.filters = self._extract_filters(query)
        
        # Set search requirements
        intent.requires_semantic_search = any(
            indicator in query for indicator in self.semantic_indicators
        )
        intent.requires_structural_search = any(
            indicator in query for indicator in self.structural_indicators  
        )
        intent.requires_exact_match = self._has_exact_match_indicators(query)
        
        # Adjust primary intent for hybrid queries
        if intent.requires_semantic_search and intent.requires_structural_search:
            intent.primary_intent = QueryIntent.HYBRID_SEARCH
        
        return intent
    
    def _extract_entities(self, query: str) -> List[QueryEntity]:
        """Extract code entities mentioned in query"""
        entities = []
        
        # Look for quoted strings (exact entity names)
        quoted_entities = re.findall(r'"([^"]+)"', query)
        for entity_name in quoted_entities:
            entities.append(QueryEntity(
                name=entity_name,
                confidence=1.0
            ))
        
        # Look for camelCase/PascalCase identifiers
        camel_case_entities = re.findall(r'\b[a-z]+[A-Z][a-zA-Z0-9]*\b', query)
        for entity_name in camel_case_entities:
            if entity_name not in [e.name for e in entities]:  # Avoid duplicates
                entities.append(QueryEntity(
                    name=entity_name,
                    confidence=0.8
                ))
        
        # Look for snake_case identifiers
        snake_case_entities = re.findall(r'\b[a-z][a-z0-9_]*[a-z0-9]\b', query)
        for entity_name in snake_case_entities:
            if (entity_name not in [e.name for e in entities] and 
                entity_name not in {'function', 'class', 'method', 'variable', 'module'}):
                entities.append(QueryEntity(
                    name=entity_name,
                    confidence=0.6
                ))
        
        # Infer entity types
        for entity in entities:
            entity.entity_type = self._infer_entity_type(entity.name, query)
        
        return entities
    
    def _infer_entity_type(self, entity_name: str, query: str) -> Optional[str]:
        """Infer entity type from context"""
        for entity_type, patterns in self.entity_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return entity_type
        
        # Heuristics based on naming conventions
        if entity_name.endswith('()'):
            return 'function'
        elif entity_name[0].isupper():
            return 'class'
        elif '.' in entity_name and entity_name.split('.')[-1] in ['py', 'js', 'ts']:
            return 'module'
        
        return None
    
    def _extract_concepts(self, query: str, entities: List[QueryEntity]) -> List[str]:
        """Extract conceptual keywords not covered by entities"""
        entity_names = {e.name.lower() for e in entities}
        
        # Remove common stop words and entity names
        stop_words = {
            'find', 'show', 'get', 'what', 'where', 'how', 'why', 'when',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'function', 'class', 'method', 'variable', 'module'
        }
        
        words = query.lower().split()
        concepts = []
        
        for word in words:
            cleaned_word = re.sub(r'[^\w]', '', word)
            if (cleaned_word and 
                cleaned_word not in stop_words and
                cleaned_word not in entity_names and
                len(cleaned_word) > 2):
                concepts.append(cleaned_word)
        
        return concepts
    
    def _extract_filters(self, query: str) -> QueryFilters:
        """Extract search filters from query"""
        filters = QueryFilters()
        
        # Extract entity type filters
        for entity_type in ['function', 'class', 'module', 'variable', 'method']:
            if entity_type in query.lower():
                filters.entity_types.add(entity_type)
        
        # Extract file pattern filters  
        for pattern, lang in self.file_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                filters.file_patterns.append(pattern)
        
        # Extract language hints
        lang_patterns = {
            'python': r'\b(python|\.py)\b',
            'javascript': r'\b(javascript|js|\.js)\b', 
            'typescript': r'\b(typescript|ts|\.ts)\b',
            'java': r'\b(java|\.java)\b',
        }
        
        for lang, pattern in lang_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                filters.file_patterns.append(f'*.{lang}')
        
        return filters
    
    def _has_exact_match_indicators(self, query: str) -> bool:
        """Check if query requires exact matching"""
        exact_indicators = [
            'exact', 'exactly', 'precise', 'specific',
            '"', "'",  # Quoted strings indicate exact match
        ]
        
        return any(indicator in query.lower() for indicator in exact_indicators)
    
    def _select_execution_strategy(self, intent: QueryIntent) -> SearchStrategy:
        """Select optimal execution strategy based on intent"""
        if intent.primary_intent == QueryIntent.HYBRID_SEARCH:
            return SearchStrategy.HYBRID_PARALLEL
        elif intent.requires_semantic_search and intent.requires_structural_search:
            return SearchStrategy.HYBRID_PARALLEL
        elif intent.requires_semantic_search:
            return SearchStrategy.VECTOR_ONLY
        elif intent.requires_structural_search:
            return SearchStrategy.GRAPH_ONLY
        elif intent.entities and intent.requires_exact_match:
            return SearchStrategy.GRAPH_ONLY  # Direct lookup via graph
        else:
            # Default to hybrid for best results
            return SearchStrategy.HYBRID_PARALLEL
    
    def _generate_vector_query(self, intent: QueryIntent, query: str) -> Optional[str]:
        """Generate vector similarity search query"""
        if not intent.requires_semantic_search and intent.primary_intent != QueryIntent.HYBRID_SEARCH:
            return None
        
        # Build semantic query combining entities and concepts
        query_parts = []
        
        # Add entity names
        for entity in intent.entities:
            query_parts.append(entity.name)
        
        # Add conceptual terms
        query_parts.extend(intent.concepts)
        
        # If no specific entities/concepts, use original query
        if not query_parts:
            query_parts.append(query)
        
        return ' '.join(query_parts)
    
    def _generate_graph_query(self, intent: QueryIntent) -> Optional[Dict[str, Any]]:
        """Generate graph traversal query"""
        if not intent.requires_structural_search and intent.primary_intent not in [
            QueryIntent.DEPENDENCY_ANALYSIS, QueryIntent.IMPACT_ANALYSIS, QueryIntent.HYBRID_SEARCH
        ]:
            return None
        
        query = {
            'start_entities': [],
            'relationship_types': [],
            'direction': 'both',  # outgoing, incoming, both
            'max_depth': 3,
            'min_confidence': 0.5
        }
        
        # Add start entities
        for entity in intent.entities:
            query['start_entities'].append({
                'name': entity.name,
                'type': entity.entity_type,
                'confidence': entity.confidence
            })
        
        # Set relationship types based on intent
        if intent.primary_intent == QueryIntent.DEPENDENCY_ANALYSIS:
            query['relationship_types'] = ['calls', 'uses', 'imports', 'references']
            query['direction'] = 'outgoing'
        elif intent.primary_intent == QueryIntent.IMPACT_ANALYSIS:
            query['relationship_types'] = ['calls', 'uses', 'imports', 'references']
            query['direction'] = 'incoming'
        else:
            # Hybrid - include all relationship types
            query['relationship_types'] = ['calls', 'uses', 'imports', 'references', 'extends', 'implements']
        
        return query
    
    def _generate_direct_lookup(self, intent: QueryIntent) -> Optional[Dict[str, Any]]:
        """Generate direct entity lookup query"""
        if not intent.entities or not intent.requires_exact_match:
            return None
        
        return {
            'entities': [
                {
                    'name': entity.name,
                    'type': entity.entity_type,
                    'file_path': entity.file_path
                }
                for entity in intent.entities
            ],
            'exact_match': True,
            'filters': {
                'entity_types': list(intent.filters.entity_types),
                'file_patterns': intent.filters.file_patterns
            }
        }
    
    def _estimate_result_limit(self, intent: QueryIntent) -> int:
        """Estimate appropriate result limit based on query intent"""
        if intent.primary_intent == QueryIntent.ENTITY_SEARCH:
            return 10  # Precise search, fewer results
        elif intent.primary_intent == QueryIntent.SIMILARITY_SEARCH:
            return 20  # Semantic search, more examples
        elif intent.primary_intent in [QueryIntent.DEPENDENCY_ANALYSIS, QueryIntent.IMPACT_ANALYSIS]:
            return 30  # Relationship analysis, comprehensive results
        else:
            return 25  # Hybrid search, balanced results
    
    def _estimate_timeout(self, intent: QueryIntent, strategy: SearchStrategy) -> int:
        """Estimate appropriate timeout based on complexity"""
        base_timeout = 1000  # 1 second base
        
        # Strategy complexity multiplier
        strategy_multiplier = {
            SearchStrategy.VECTOR_ONLY: 1.0,
            SearchStrategy.GRAPH_ONLY: 1.5,
            SearchStrategy.HYBRID_PARALLEL: 2.0,
            SearchStrategy.SEQUENTIAL: 3.0
        }
        
        # Intent complexity multiplier
        intent_multiplier = {
            QueryIntent.ENTITY_SEARCH: 1.0,
            QueryIntent.SIMILARITY_SEARCH: 1.5,
            QueryIntent.DEPENDENCY_ANALYSIS: 2.0,
            QueryIntent.IMPACT_ANALYSIS: 2.5,
            QueryIntent.HYBRID_SEARCH: 3.0
        }
        
        timeout = int(
            base_timeout * 
            strategy_multiplier[strategy] * 
            intent_multiplier[intent.primary_intent]
        )
        
        # Cap at 5 seconds
        return min(timeout, 5000)


def parse_query(query: str) -> StructuredQuery:
    """Convenience function for parsing queries"""
    parser = QueryParser()
    return parser.parse_query(query)


# Example usage and testing
if __name__ == "__main__":
    parser = QueryParser()
    
    test_queries = [
        "find function authenticateUser",
        "show me code similar to error handling in auth.py", 
        "what calls the processPayment function",
        "what breaks if I change the User class",
        "find authentication patterns in Python files"
    ]
    
    for query in test_queries:
        result = parser.parse_query(query)
        print(f"Query: {query}")
        print(f"Intent: {result.intent.primary_intent.value}")
        print(f"Strategy: {result.execution_strategy.value}")
        print(f"Entities: {[e.name for e in result.intent.entities]}")
        print("---")