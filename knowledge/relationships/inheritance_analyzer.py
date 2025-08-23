"""
Inheritance relationship analyzer for detecting class hierarchies and interface implementations.

This analyzer detects inheritance patterns, interface implementations, and other
type hierarchy relationships in object-oriented languages.
"""

import re
from typing import List, Dict, Set, Optional, Any
from tree_sitter import Node, Tree

from .base_analyzer import BaseRelationshipAnalyzer
from .relationship_types import CodeRelationship, RelationshipType, RelationshipContext
from ..extraction.entity_types import CodeEntity, EntityType


class InheritanceAnalyzer(BaseRelationshipAnalyzer):
    """
    Analyzes inheritance and interface implementation relationships.
    
    Supports:
    - Class inheritance (extends)
    - Interface implementation (implements)
    - Trait usage (Rust)
    - Mixin patterns
    - Abstract class relationships
    """
    
    def __init__(self, parser_manager):
        super().__init__(parser_manager)
        self.supported_languages = {"javascript", "python", "go", "rust"}
        
        # Language-specific AST node types for inheritance
        self.inheritance_node_types = {
            "javascript": {
                "class_declaration",
                "class_expression",
                "interface_declaration"  # TypeScript
            },
            "python": {
                "class_definition"
            },
            "go": {
                "type_declaration",
                "interface_type",
                "struct_type"
            },
            "rust": {
                "impl_item",
                "trait_item",
                "struct_item"
            }
        }
    
    def get_supported_relationship_types(self) -> Set[RelationshipType]:
        """Inheritance analyzer supports extends and implements relationship types."""
        return {RelationshipType.EXTENDS, RelationshipType.IMPLEMENTS}
    
    def analyze_ast(self, tree: Tree, file_path: str, language: str, 
                   entities: List[CodeEntity]) -> List[CodeRelationship]:
        """Analyze AST tree to find inheritance relationships."""
        
        if language not in self.supported_languages:
            return []
        
        relationships = []
        source_code = self.parser_manager.get_file_content(file_path)
        
        # Find inheritance-related nodes based on language
        inheritance_nodes = self.find_inheritance_nodes(tree.root_node, language)
        
        for node in inheritance_nodes:
            try:
                detected_relationships = self.analyze_inheritance_node(
                    node, source_code, file_path, language, entities
                )
                relationships.extend(detected_relationships)
            except Exception as e:
                self.logger.warning(f"Error analyzing inheritance node in {file_path}: {e}")
        
        self.logger.debug(f"Found {len(relationships)} inheritance relationships in {file_path}")
        return relationships
    
    def find_inheritance_nodes(self, root: Node, language: str) -> List[Node]:
        """Find all inheritance related nodes in the AST."""
        node_types = self.inheritance_node_types.get(language, set())
        return self.find_nodes_by_types(root, node_types)
    
    def analyze_inheritance_node(self, node: Node, source_code: bytes, file_path: str,
                               language: str, entities: List[CodeEntity]) -> List[CodeRelationship]:
        """Analyze a single inheritance node to extract relationships."""
        
        relationships = []
        
        if language == "javascript":
            relationships.extend(self.analyze_javascript_inheritance(node, source_code, file_path, entities))
        elif language == "python":
            relationships.extend(self.analyze_python_inheritance(node, source_code, file_path, entities))
        elif language == "go":
            relationships.extend(self.analyze_go_inheritance(node, source_code, file_path, entities))
        elif language == "rust":
            relationships.extend(self.analyze_rust_inheritance(node, source_code, file_path, entities))
        
        return relationships
    
    def analyze_javascript_inheritance(self, node: Node, source_code: bytes, file_path: str,
                                     entities: List[CodeEntity]) -> List[CodeRelationship]:
        """Analyze JavaScript/TypeScript class inheritance and interface implementation."""
        relationships = []
        
        if node.type in ["class_declaration", "class_expression"]:
            class_info = self.parse_javascript_class(node, source_code)
            if class_info:
                relationships.extend(self.create_inheritance_relationships(
                    class_info, file_path, entities, node, source_code
                ))
        
        elif node.type == "interface_declaration":
            # TypeScript interface inheritance
            interface_info = self.parse_typescript_interface(node, source_code)
            if interface_info:
                relationships.extend(self.create_inheritance_relationships(
                    interface_info, file_path, entities, node, source_code
                ))
        
        return relationships
    
    def analyze_python_inheritance(self, node: Node, source_code: bytes, file_path: str,
                                 entities: List[CodeEntity]) -> List[CodeRelationship]:
        """Analyze Python class inheritance."""
        relationships = []
        
        if node.type == "class_definition":
            class_info = self.parse_python_class(node, source_code)
            if class_info:
                relationships.extend(self.create_inheritance_relationships(
                    class_info, file_path, entities, node, source_code
                ))
        
        return relationships
    
    def analyze_go_inheritance(self, node: Node, source_code: bytes, file_path: str,
                             entities: List[CodeEntity]) -> List[CodeRelationship]:
        """Analyze Go interface implementation (structs implementing interfaces)."""
        relationships = []
        
        # Go uses implicit interface implementation
        # We need to analyze method signatures to determine implementation
        if node.type == "type_declaration":
            type_info = self.parse_go_type_declaration(node, source_code)
            if type_info:
                relationships.extend(self.create_inheritance_relationships(
                    type_info, file_path, entities, node, source_code
                ))
        
        return relationships
    
    def analyze_rust_inheritance(self, node: Node, source_code: bytes, file_path: str,
                               entities: List[CodeEntity]) -> List[CodeRelationship]:
        """Analyze Rust trait implementation."""
        relationships = []
        
        if node.type == "impl_item":
            impl_info = self.parse_rust_impl(node, source_code)
            if impl_info:
                relationships.extend(self.create_inheritance_relationships(
                    impl_info, file_path, entities, node, source_code
                ))
        
        elif node.type == "trait_item":
            trait_info = self.parse_rust_trait(node, source_code)
            if trait_info:
                # Trait inheritance (trait A: B)
                relationships.extend(self.create_inheritance_relationships(
                    trait_info, file_path, entities, node, source_code
                ))
        
        return relationships
    
    def parse_javascript_class(self, node: Node, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Parse JavaScript class declaration for inheritance information."""
        node_text = self.extract_node_text(node, source_code)
        class_name = None
        extends_class = None
        implements_interfaces = []
        
        # Extract class name
        for child in node.children:
            if child.type == "identifier" and not class_name:
                class_name = self.extract_node_text(child, source_code)
            elif child.type == "class_heritage":
                # Parse extends and implements clauses
                heritage_text = self.extract_node_text(child, source_code)
                
                # Extract extends clause
                extends_match = re.search(r'extends\s+([^\s,{]+)', heritage_text)
                if extends_match:
                    extends_class = extends_match.group(1)
                
                # Extract implements clause (TypeScript)
                implements_match = re.search(r'implements\s+([^{]+)', heritage_text)
                if implements_match:
                    implements_list = implements_match.group(1).strip()
                    implements_interfaces = [iface.strip() for iface in implements_list.split(',')]
        
        if class_name:
            return {
                'type': 'class',
                'name': class_name,
                'extends': extends_class,
                'implements': implements_interfaces
            }
        
        return None
    
    def parse_typescript_interface(self, node: Node, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Parse TypeScript interface declaration for inheritance."""
        interface_name = None
        extends_interfaces = []
        
        # Extract interface name and extends clause
        for child in node.children:
            if child.type == "identifier" and not interface_name:
                interface_name = self.extract_node_text(child, source_code)
            elif child.type == "extends_clause":
                extends_text = self.extract_node_text(child, source_code)
                # Extract interface names after "extends"
                extends_match = re.search(r'extends\s+(.+)', extends_text)
                if extends_match:
                    extends_list = extends_match.group(1).strip()
                    extends_interfaces = [iface.strip() for iface in extends_list.split(',')]
        
        if interface_name:
            return {
                'type': 'interface',
                'name': interface_name,
                'extends': extends_interfaces,
                'implements': []
            }
        
        return None
    
    def parse_python_class(self, node: Node, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Parse Python class definition for inheritance."""
        node_text = self.extract_node_text(node, source_code)
        
        # Parse class definition: class ClassName(BaseClass, Interface):
        pattern = re.compile(r'class\s+(\w+)(?:\s*\(\s*([^)]*)\s*\))?')
        match = pattern.search(node_text)
        
        if match:
            class_name = match.group(1)
            base_classes = []
            
            if match.group(2):
                base_list = match.group(2).strip()
                if base_list:
                    base_classes = [base.strip() for base in base_list.split(',')]
            
            return {
                'type': 'class',
                'name': class_name,
                'extends': base_classes[0] if base_classes else None,
                'implements': base_classes[1:] if len(base_classes) > 1 else []
            }
        
        return None
    
    def parse_go_type_declaration(self, node: Node, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Parse Go type declaration."""
        # Go doesn't have explicit inheritance, but we can detect interface implementation
        # This would require more sophisticated analysis of method signatures
        node_text = self.extract_node_text(node, source_code)
        
        # Basic parsing for type declarations
        pattern = re.compile(r'type\s+(\w+)\s+(struct|interface)')
        match = pattern.search(node_text)
        
        if match:
            type_name = match.group(1)
            type_kind = match.group(2)
            
            return {
                'type': type_kind,
                'name': type_name,
                'extends': None,
                'implements': []  # Would need method signature analysis
            }
        
        return None
    
    def parse_rust_impl(self, node: Node, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Parse Rust impl block for trait implementation."""
        node_text = self.extract_node_text(node, source_code)
        
        # Parse impl patterns:
        # impl TraitName for StructName
        # impl StructName
        patterns = [
            re.compile(r'impl\s+(\w+)\s+for\s+(\w+)'),  # trait implementation
            re.compile(r'impl\s+(\w+)')  # inherent implementation
        ]
        
        for pattern in patterns:
            match = pattern.search(node_text)
            if match:
                if len(match.groups()) == 2:
                    # Trait implementation
                    trait_name = match.group(1)
                    struct_name = match.group(2)
                    return {
                        'type': 'impl',
                        'name': struct_name,
                        'extends': None,
                        'implements': [trait_name]
                    }
                else:
                    # Inherent implementation
                    struct_name = match.group(1)
                    return {
                        'type': 'impl',
                        'name': struct_name,
                        'extends': None,
                        'implements': []
                    }
        
        return None
    
    def parse_rust_trait(self, node: Node, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Parse Rust trait declaration for trait inheritance."""
        node_text = self.extract_node_text(node, source_code)
        
        # Parse trait inheritance: trait A: B + C
        pattern = re.compile(r'trait\s+(\w+)(?:\s*:\s*([^{]+))?')
        match = pattern.search(node_text)
        
        if match:
            trait_name = match.group(1)
            super_traits = []
            
            if match.group(2):
                super_trait_list = match.group(2).strip()
                super_traits = [trait.strip() for trait in super_trait_list.split('+')]
            
            return {
                'type': 'trait',
                'name': trait_name,
                'extends': super_traits,
                'implements': []
            }
        
        return None
    
    def create_inheritance_relationships(self, inheritance_info: Dict[str, Any], file_path: str,
                                       entities: List[CodeEntity], node: Node, source_code: bytes) -> List[CodeRelationship]:
        """Create inheritance relationships from parsed inheritance information."""
        relationships = []
        
        entity_name = inheritance_info['name']
        entity_type = inheritance_info['type']
        
        # Find the source entity (class, interface, trait, etc.)
        source_entity = self.find_entity_by_name(entities, entity_name, "class")
        if not source_entity:
            # Try to find as interface or other type
            source_entity = self.find_entity_by_name(entities, entity_name)
        
        if not source_entity:
            return relationships
        
        # Create extends relationships
        extends_targets = inheritance_info.get('extends', [])
        if isinstance(extends_targets, str):
            extends_targets = [extends_targets] if extends_targets else []
        
        for target_name in extends_targets:
            if target_name:
                target_entity = self.find_or_create_class_entity(target_name, file_path, entity_type)
                if target_entity:
                    relationships.append(self.create_inheritance_relationship(
                        source_entity, target_entity, RelationshipType.EXTENDS,
                        node, source_code, {'inheritance_type': 'extends'}
                    ))
        
        # Create implements relationships
        implements_targets = inheritance_info.get('implements', [])
        for target_name in implements_targets:
            if target_name:
                target_entity = self.find_or_create_class_entity(target_name, file_path, 'interface')
                if target_entity:
                    relationships.append(self.create_inheritance_relationship(
                        source_entity, target_entity, RelationshipType.IMPLEMENTS,
                        node, source_code, {'inheritance_type': 'implements'}
                    ))
        
        return relationships
    
    def create_inheritance_relationship(self, source_entity: CodeEntity, target_entity: CodeEntity,
                                      relationship_type: RelationshipType, node: Node, source_code: bytes,
                                      extra_metadata: Dict[str, Any] = None) -> CodeRelationship:
        """Create a single inheritance relationship."""
        
        context = RelationshipContext(
            line_number=self.get_node_line_number(node),
            source_code=self.extract_node_text(node, source_code)
        )
        
        metadata = extra_metadata or {}
        metadata.update({
            'source_class': source_entity.name,
            'target_class': target_entity.name
        })
        
        # Calculate confidence based on explicitness of inheritance
        confidence = 0.98  # Inheritance is usually very explicit
        if target_entity.metadata.get('needs_resolution'):
            confidence = 0.85  # Reduce confidence for unresolved references
        
        return CodeRelationship(
            source_id=source_entity.id,
            target_id=target_entity.id,
            relationship_type=relationship_type,
            confidence=confidence,
            context=context,
            metadata=metadata
        )
    
    def find_or_create_class_entity(self, class_name: str, from_file: str, 
                                   entity_type: str = 'class') -> Optional[CodeEntity]:
        """Find existing class entity or create placeholder for cross-file resolution."""
        
        # In a full implementation, this would use the cross-file resolver
        # For now, create a placeholder entity
        from uuid import uuid4
        from ..extraction.entity_types import EntityType
        
        # Map entity types
        entity_type_map = {
            'class': EntityType.CLASS,
            'interface': EntityType.INTERFACE,
            'trait': EntityType.INTERFACE,  # Treat traits as interfaces
            'struct': EntityType.CLASS      # Treat structs as classes
        }
        
        return CodeEntity(
            id=uuid4(),
            type=entity_type_map.get(entity_type, EntityType.CLASS),
            name=class_name,
            file_path=f"<unresolved>:{class_name}",
            location=None,
            metadata={
                'original_type': entity_type,
                'created_by': 'inheritance_analyzer',
                'needs_resolution': True,
                'referenced_from': from_file
            }
        )
    
    def calculate_inheritance_confidence(self, source_entity: CodeEntity, target_entity: CodeEntity,
                                       relationship_type: RelationshipType) -> float:
        """Calculate confidence score for inheritance relationship."""
        
        base_confidence = 0.95
        
        # Inheritance is usually explicit and unambiguous
        if relationship_type == RelationshipType.EXTENDS:
            base_confidence = 0.98
        elif relationship_type == RelationshipType.IMPLEMENTS:
            base_confidence = 0.95
        
        # Reduce confidence for cross-file references
        if source_entity.file_path != target_entity.file_path:
            base_confidence *= 0.9
        
        # Reduce confidence for unresolved entities
        if target_entity.metadata.get('needs_resolution'):
            base_confidence *= 0.85
        
        return min(base_confidence, 1.0)