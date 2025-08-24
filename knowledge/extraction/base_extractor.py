"""
Base class for language-specific entity extractors.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import tree_sitter

from .entity_types import CodeEntity, EntityType, SourceLocation, ExtractionResult


class BaseLanguageExtractor(ABC):
    """
    Base class for language-specific AST entity extractors.
    
    Each supported language will have its own extractor that inherits from this class
    and implements the language-specific extraction logic.
    """
    
    def __init__(self, language: str):
        self.language = language
        self._supported_entity_types = [
            EntityType.FUNCTION,
            EntityType.CLASS, 
            EntityType.MODULE,
            EntityType.VARIABLE
        ]
    
    @abstractmethod
    def extract_entities(self, tree: tree_sitter.Tree, file_path: str, source_code: bytes) -> List[CodeEntity]:
        """
        Extract code entities from an AST tree.
        
        Args:
            tree: Parsed AST tree from tree-sitter
            file_path: Path to the source file
            source_code: Raw source code as bytes
            
        Returns:
            List of extracted CodeEntity objects
        """
        pass
    
    def get_supported_entity_types(self) -> List[EntityType]:
        """Get the entity types this extractor can handle."""
        return self._supported_entity_types
    
    def _get_node_text(self, node: tree_sitter.Node, source_code: bytes) -> str:
        """Extract text content from an AST node."""
        return source_code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
    
    def _get_source_location(self, node: tree_sitter.Node) -> SourceLocation:
        """Get source location from an AST node."""
        return SourceLocation(
            line_start=node.start_point[0] + 1,  # tree-sitter uses 0-based lines
            line_end=node.end_point[0] + 1,
            column_start=node.start_point[1],
            column_end=node.end_point[1]
        )
    
    def _extract_function_signature(self, node: tree_sitter.Node, source_code: bytes) -> Dict[str, Any]:
        """
        Extract function signature information (name, parameters, return type).
        This is a default implementation that can be overridden by language-specific extractors.
        """
        metadata = {}
        
        # Find function name
        name_node = self._find_child_by_type(node, 'identifier')
        if name_node:
            metadata['name'] = self._get_node_text(name_node, source_code)
        
        # Find parameters
        params_node = self._find_child_by_type(node, 'parameters')
        if params_node:
            parameters = []
            for param_node in params_node.children:
                if param_node.type == 'identifier':
                    parameters.append(self._get_node_text(param_node, source_code))
            metadata['parameters'] = parameters
        
        return metadata
    
    def _extract_class_info(self, node: tree_sitter.Node, source_code: bytes) -> Dict[str, Any]:
        """
        Extract class information (name, methods, inheritance).
        This is a default implementation that can be overridden by language-specific extractors.
        """
        metadata = {}
        
        # Find class name
        name_node = self._find_child_by_type(node, 'identifier')
        if name_node:
            metadata['name'] = self._get_node_text(name_node, source_code)
        
        # Find methods
        methods = []
        for child in node.children:
            if child.type in ['method_definition', 'function_definition']:
                method_name_node = self._find_child_by_type(child, 'identifier')
                if method_name_node:
                    methods.append(self._get_node_text(method_name_node, source_code))
        metadata['methods'] = methods
        
        return metadata
    
    def _find_child_by_type(self, node: tree_sitter.Node, node_type: str) -> Optional[tree_sitter.Node]:
        """Find first child node of specific type."""
        for child in node.children:
            if child.type == node_type:
                return child
        return None
    
    def _find_children_by_type(self, node: tree_sitter.Node, node_type: str) -> List[tree_sitter.Node]:
        """Find all child nodes of specific type."""
        return [child for child in node.children if child.type == node_type]
    
    def _calculate_complexity(self, node: tree_sitter.Node) -> int:
        """
        Calculate cyclomatic complexity for functions.
        Basic implementation counting decision points.
        """
        complexity = 1  # Base complexity
        
        def count_decision_points(n):
            nonlocal complexity
            if n.type in ['if_statement', 'while_statement', 'for_statement', 
                         'switch_statement', 'conditional_expression']:
                complexity += 1
            for child in n.children:
                count_decision_points(child)
        
        count_decision_points(node)
        return complexity