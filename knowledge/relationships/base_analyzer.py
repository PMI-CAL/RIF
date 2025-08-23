"""
Base classes for relationship analyzers.
"""

from abc import ABC, abstractmethod
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from uuid import UUID
import time

from tree_sitter import Node, Tree

from .relationship_types import CodeRelationship, RelationshipDetectionResult, RelationshipType
from ..extraction.entity_types import CodeEntity
from ..parsing.parser_manager import ParserManager


class BaseRelationshipAnalyzer(ABC):
    """
    Base class for all relationship analyzers.
    
    Each analyzer is responsible for detecting a specific type of relationship
    (imports, calls, inheritance, etc.) from AST nodes.
    """
    
    def __init__(self, parser_manager: ParserManager):
        self.parser_manager = parser_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self.supported_languages: Set[str] = set()
    
    @abstractmethod
    def get_supported_relationship_types(self) -> Set[RelationshipType]:
        """Return the relationship types this analyzer can detect."""
        pass
    
    @abstractmethod
    def analyze_ast(self, tree: Tree, file_path: str, language: str, 
                   entities: List[CodeEntity]) -> List[CodeRelationship]:
        """
        Analyze AST tree to find relationships.
        
        Args:
            tree: Parsed AST tree
            file_path: Path to the source file
            language: Programming language
            entities: List of entities found in this file
            
        Returns:
            List of detected relationships
        """
        pass
    
    def can_analyze(self, language: str) -> bool:
        """Check if this analyzer supports the given language."""
        return language in self.supported_languages
    
    def create_relationship(self, source_entity: CodeEntity, target_entity: CodeEntity,
                          relationship_type: RelationshipType, confidence: float = 1.0,
                          line_number: Optional[int] = None, 
                          source_code: Optional[str] = None,
                          **metadata) -> CodeRelationship:
        """Helper method to create a relationship with proper metadata."""
        
        from .relationship_types import RelationshipContext
        
        context = RelationshipContext(
            line_number=line_number,
            source_code=source_code
        )
        
        return CodeRelationship(
            source_id=source_entity.id,
            target_id=target_entity.id,
            relationship_type=relationship_type,
            confidence=confidence,
            context=context,
            metadata=metadata
        )
    
    def find_entity_by_name(self, entities: List[CodeEntity], name: str, 
                           entity_type: Optional[str] = None) -> Optional[CodeEntity]:
        """Find an entity by name and optionally type."""
        for entity in entities:
            if entity.name == name:
                if entity_type is None or entity.type.value == entity_type:
                    return entity
        return None
    
    def extract_node_text(self, node: Node, source_code: bytes) -> str:
        """Extract text content from an AST node."""
        return source_code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
    
    def get_node_line_number(self, node: Node) -> int:
        """Get line number for an AST node (1-based)."""
        return node.start_point[0] + 1
    
    def find_nodes_by_type(self, root: Node, node_type: str) -> List[Node]:
        """Find all nodes of a specific type in the AST."""
        results = []
        
        def traverse(node: Node):
            if node.type == node_type:
                results.append(node)
            for child in node.children:
                traverse(child)
        
        traverse(root)
        return results
    
    def find_nodes_by_types(self, root: Node, node_types: Set[str]) -> List[Node]:
        """Find all nodes of any of the specified types."""
        results = []
        
        def traverse(node: Node):
            if node.type in node_types:
                results.append(node)
            for child in node.children:
                traverse(child)
        
        traverse(root)
        return results


class CrossFileReferenceResolver:
    """
    Resolves cross-file references for relationships.
    
    This class helps analyzers resolve references to entities in other files
    by maintaining a registry of all known entities and their locations.
    """
    
    def __init__(self):
        self.entity_registry: Dict[str, List[CodeEntity]] = {}  # name -> entities
        self.file_entities: Dict[str, List[CodeEntity]] = {}    # file -> entities
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register_entities(self, entities: List[CodeEntity]):
        """Register entities for cross-file reference resolution."""
        for entity in entities:
            # Index by name for quick lookup
            if entity.name not in self.entity_registry:
                self.entity_registry[entity.name] = []
            self.entity_registry[entity.name].append(entity)
            
            # Index by file for file-based lookups
            if entity.file_path not in self.file_entities:
                self.file_entities[entity.file_path] = []
            self.file_entities[entity.file_path].append(entity)
    
    def find_entity_by_name(self, name: str, entity_type: Optional[str] = None,
                           exclude_file: Optional[str] = None) -> List[CodeEntity]:
        """Find entities by name, optionally filtering by type and excluding a file."""
        candidates = self.entity_registry.get(name, [])
        
        results = []
        for entity in candidates:
            # Filter by type if specified
            if entity_type and entity.type.value != entity_type:
                continue
            
            # Exclude entities from specified file
            if exclude_file and entity.file_path == exclude_file:
                continue
            
            results.append(entity)
        
        return results
    
    def find_module_entity(self, module_name: str, from_file: str) -> Optional[CodeEntity]:
        """Find a module entity by name, excluding the current file."""
        candidates = self.find_entity_by_name(module_name, "module", exclude_file=from_file)
        
        # Prefer exact module name matches
        for entity in candidates:
            if entity.name == module_name:
                return entity
        
        # Try to find by file path patterns
        for entity in candidates:
            if module_name in entity.file_path:
                return entity
        
        return candidates[0] if candidates else None
    
    def resolve_import_target(self, import_path: str, from_file: str) -> Optional[CodeEntity]:
        """Resolve an import statement to a target entity."""
        
        # Handle relative imports (./file, ../file) 
        if import_path.startswith('.'):
            # TODO: Implement relative import resolution
            return None
        
        # Handle direct module names
        parts = import_path.split('.')
        module_name = parts[-1] if parts else import_path
        
        return self.find_module_entity(module_name, from_file)
    
    def resolve_function_call(self, function_name: str, from_file: str) -> List[CodeEntity]:
        """Resolve a function call to possible target functions."""
        return self.find_entity_by_name(function_name, "function", exclude_file=from_file)
    
    def resolve_class_reference(self, class_name: str, from_file: str) -> List[CodeEntity]:
        """Resolve a class reference to possible target classes."""
        return self.find_entity_by_name(class_name, "class", exclude_file=from_file)
    
    def get_file_summary(self) -> Dict[str, Any]:
        """Get summary statistics about registered entities."""
        return {
            'total_entities': sum(len(entities) for entities in self.entity_registry.values()),
            'unique_names': len(self.entity_registry),
            'files_covered': len(self.file_entities),
            'entity_breakdown': {
                name: len(entities) for name, entities in self.entity_registry.items() 
                if len(entities) > 1  # Only show names with multiple entities
            }
        }


class RelationshipAnalysisContext:
    """
    Context object that provides shared state and utilities for relationship analysis.
    """
    
    def __init__(self, parser_manager: ParserManager):
        self.parser_manager = parser_manager
        self.cross_file_resolver = CrossFileReferenceResolver()
        self.current_file: Optional[str] = None
        self.current_language: Optional[str] = None
        self.current_entities: List[CodeEntity] = []
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance tracking
        self.analysis_start_time: Optional[float] = None
        self.relationships_detected: int = 0
    
    def start_file_analysis(self, file_path: str, language: str, entities: List[CodeEntity]):
        """Begin analysis of a new file."""
        self.current_file = file_path
        self.current_language = language  
        self.current_entities = entities
        self.analysis_start_time = time.time()
        
        # Register entities for cross-file resolution
        self.cross_file_resolver.register_entities(entities)
        
        self.logger.debug(f"Starting relationship analysis for {file_path} ({language})")
    
    def finish_file_analysis(self) -> float:
        """Complete analysis of current file and return analysis time."""
        if self.analysis_start_time is None:
            return 0.0
        
        analysis_time = time.time() - self.analysis_start_time
        self.logger.debug(
            f"Completed analysis for {self.current_file}: "
            f"{self.relationships_detected} relationships in {analysis_time:.3f}s"
        )
        
        self.relationships_detected = 0
        self.analysis_start_time = None
        return analysis_time
    
    def record_relationship(self, relationship: CodeRelationship):
        """Record that a relationship was detected."""
        self.relationships_detected += 1
    
    def get_current_entities(self) -> List[CodeEntity]:
        """Get entities for the current file being analyzed."""
        return self.current_entities
    
    def resolve_cross_file_reference(self, reference_name: str, 
                                    entity_type: Optional[str] = None) -> List[CodeEntity]:
        """Resolve a reference to entities in other files."""
        return self.cross_file_resolver.find_entity_by_name(
            reference_name, entity_type, exclude_file=self.current_file
        )