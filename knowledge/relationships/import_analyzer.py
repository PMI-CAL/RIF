"""
Import/Export relationship analyzer for detecting module dependencies.

This analyzer detects import and export statements in source code and creates
relationships between modules and the entities they import/export.
"""

import re
from typing import List, Dict, Set, Optional, Any
from tree_sitter import Node, Tree

from .base_analyzer import BaseRelationshipAnalyzer
from .relationship_types import CodeRelationship, RelationshipType, RelationshipContext
from ..extraction.entity_types import CodeEntity, EntityType


class ImportExportAnalyzer(BaseRelationshipAnalyzer):
    """
    Analyzes import and export statements to build module dependency relationships.
    
    Supports:
    - ES6 imports/exports (JavaScript/TypeScript)
    - Python imports (from x import y, import x)
    - Go imports and packages
    - Rust use statements and module declarations
    """
    
    def __init__(self, parser_manager):
        super().__init__(parser_manager)
        self.supported_languages = {"javascript", "python", "go", "rust"}
        
        # Language-specific AST node types for import/export statements
        self.import_node_types = {
            "javascript": {
                "import_statement",
                "import_clause", 
                "export_statement"
            },
            "python": {
                "import_statement",
                "import_from_statement"
            },
            "go": {
                "import_declaration",
                "import_spec"
            },
            "rust": {
                "use_declaration",
                "mod_item"
            }
        }
    
    def get_supported_relationship_types(self) -> Set[RelationshipType]:
        """Import analyzer supports imports relationship type."""
        return {RelationshipType.IMPORTS}
    
    def analyze_ast(self, tree: Tree, file_path: str, language: str, 
                   entities: List[CodeEntity]) -> List[CodeRelationship]:
        """Analyze AST tree to find import/export relationships."""
        
        if language not in self.supported_languages:
            return []
        
        relationships = []
        source_code = self.parser_manager.get_file_content(file_path)
        
        # Find import/export nodes based on language
        import_nodes = self.find_import_export_nodes(tree.root_node, language)
        
        for node in import_nodes:
            try:
                detected_relationships = self.analyze_import_node(
                    node, source_code, file_path, language, entities
                )
                relationships.extend(detected_relationships)
            except Exception as e:
                self.logger.warning(f"Error analyzing import node in {file_path}: {e}")
        
        self.logger.debug(f"Found {len(relationships)} import relationships in {file_path}")
        return relationships
    
    def find_import_export_nodes(self, root: Node, language: str) -> List[Node]:
        """Find all import/export related nodes in the AST."""
        node_types = self.import_node_types.get(language, set())
        return self.find_nodes_by_types(root, node_types)
    
    def analyze_import_node(self, node: Node, source_code: bytes, file_path: str,
                           language: str, entities: List[CodeEntity]) -> List[CodeRelationship]:
        """Analyze a single import/export node to extract relationships."""
        
        relationships = []
        
        if language == "javascript":
            relationships.extend(self.analyze_javascript_import(node, source_code, file_path, entities))
        elif language == "python":
            relationships.extend(self.analyze_python_import(node, source_code, file_path, entities))
        elif language == "go":
            relationships.extend(self.analyze_go_import(node, source_code, file_path, entities))
        elif language == "rust":
            relationships.extend(self.analyze_rust_import(node, source_code, file_path, entities))
        
        return relationships
    
    def analyze_javascript_import(self, node: Node, source_code: bytes, file_path: str,
                                entities: List[CodeEntity]) -> List[CodeRelationship]:
        """Analyze JavaScript/TypeScript import statements."""
        relationships = []
        line_number = self.get_node_line_number(node)
        node_text = self.extract_node_text(node, source_code)
        
        # Parse different import patterns
        import_info = self.parse_javascript_import(node, source_code)
        if not import_info:
            return relationships
        
        source_module = import_info['module_path']
        imported_names = import_info['imported_names']
        import_type = import_info['import_type']  # 'default', 'named', 'namespace', 'side-effect'
        
        # Find or create module entity for the imported module
        target_module = self.find_or_create_module_entity(source_module, file_path)
        current_module = self.find_current_module_entity(entities)
        
        if current_module and target_module:
            # Create module-to-module relationship
            context = RelationshipContext(
                line_number=line_number,
                source_code=node_text,
                import_alias=import_info.get('alias')
            )
            
            relationship = CodeRelationship(
                source_id=current_module.id,
                target_id=target_module.id,
                relationship_type=RelationshipType.IMPORTS,
                confidence=0.95,  # High confidence for explicit imports
                context=context,
                metadata={
                    'import_type': import_type,
                    'imported_names': imported_names,
                    'module_path': source_module
                }
            )
            relationships.append(relationship)
        
        return relationships
    
    def analyze_python_import(self, node: Node, source_code: bytes, file_path: str,
                            entities: List[CodeEntity]) -> List[CodeRelationship]:
        """Analyze Python import statements."""
        relationships = []
        line_number = self.get_node_line_number(node)
        node_text = self.extract_node_text(node, source_code)
        
        # Parse Python import patterns
        import_info = self.parse_python_import(node, source_code)
        if not import_info:
            return relationships
        
        current_module = self.find_current_module_entity(entities)
        
        for module_path, imported_names, alias in import_info:
            target_module = self.find_or_create_module_entity(module_path, file_path)
            
            if current_module and target_module:
                context = RelationshipContext(
                    line_number=line_number,
                    source_code=node_text,
                    import_alias=alias
                )
                
                relationship = CodeRelationship(
                    source_id=current_module.id,
                    target_id=target_module.id,
                    relationship_type=RelationshipType.IMPORTS,
                    confidence=0.95,
                    context=context,
                    metadata={
                        'imported_names': imported_names,
                        'module_path': module_path,
                        'is_from_import': node.type == 'import_from_statement'
                    }
                )
                relationships.append(relationship)
        
        return relationships
    
    def analyze_go_import(self, node: Node, source_code: bytes, file_path: str,
                         entities: List[CodeEntity]) -> List[CodeRelationship]:
        """Analyze Go import declarations."""
        relationships = []
        line_number = self.get_node_line_number(node)
        node_text = self.extract_node_text(node, source_code)
        
        # Go imports are package imports
        import_path = self.parse_go_import(node, source_code)
        if not import_path:
            return relationships
        
        current_module = self.find_current_module_entity(entities)
        target_module = self.find_or_create_module_entity(import_path, file_path)
        
        if current_module and target_module:
            context = RelationshipContext(
                line_number=line_number,
                source_code=node_text
            )
            
            relationship = CodeRelationship(
                source_id=current_module.id,
                target_id=target_module.id,
                relationship_type=RelationshipType.IMPORTS,
                confidence=0.98,  # Very high confidence for Go imports
                context=context,
                metadata={
                    'import_path': import_path,
                    'language': 'go'
                }
            )
            relationships.append(relationship)
        
        return relationships
    
    def analyze_rust_import(self, node: Node, source_code: bytes, file_path: str,
                          entities: List[CodeEntity]) -> List[CodeRelationship]:
        """Analyze Rust use declarations and mod items."""
        relationships = []
        line_number = self.get_node_line_number(node)
        node_text = self.extract_node_text(node, source_code)
        
        if node.type == "use_declaration":
            use_info = self.parse_rust_use(node, source_code)
            if use_info:
                current_module = self.find_current_module_entity(entities)
                target_module = self.find_or_create_module_entity(use_info['module_path'], file_path)
                
                if current_module and target_module:
                    context = RelationshipContext(
                        line_number=line_number,
                        source_code=node_text
                    )
                    
                    relationship = CodeRelationship(
                        source_id=current_module.id,
                        target_id=target_module.id,
                        relationship_type=RelationshipType.IMPORTS,
                        confidence=0.95,
                        context=context,
                        metadata={
                            'use_path': use_info['module_path'],
                            'imported_items': use_info['imported_items'],
                            'language': 'rust'
                        }
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def parse_javascript_import(self, node: Node, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Parse JavaScript import statement details."""
        node_text = self.extract_node_text(node, source_code)
        
        # Regex patterns for different import types
        patterns = {
            'default': re.compile(r'import\s+(\w+)\s+from\s+[\'"]([^\'"]+)[\'"]'),
            'named': re.compile(r'import\s+\{([^}]+)\}\s+from\s+[\'"]([^\'"]+)[\'"]'),
            'namespace': re.compile(r'import\s+\*\s+as\s+(\w+)\s+from\s+[\'"]([^\'"]+)[\'"]'),
            'side_effect': re.compile(r'import\s+[\'"]([^\'"]+)[\'"]')
        }
        
        for import_type, pattern in patterns.items():
            match = pattern.search(node_text)
            if match:
                if import_type == 'default':
                    return {
                        'import_type': 'default',
                        'module_path': match.group(2),
                        'imported_names': [match.group(1)],
                        'alias': match.group(1)
                    }
                elif import_type == 'named':
                    names = [name.strip() for name in match.group(1).split(',')]
                    return {
                        'import_type': 'named',
                        'module_path': match.group(2),
                        'imported_names': names
                    }
                elif import_type == 'namespace':
                    return {
                        'import_type': 'namespace', 
                        'module_path': match.group(2),
                        'imported_names': ['*'],
                        'alias': match.group(1)
                    }
                elif import_type == 'side_effect':
                    return {
                        'import_type': 'side_effect',
                        'module_path': match.group(1),
                        'imported_names': []
                    }
        
        return None
    
    def parse_python_import(self, node: Node, source_code: bytes) -> List[tuple]:
        """Parse Python import statement details."""
        node_text = self.extract_node_text(node, source_code)
        results = []
        
        if node.type == 'import_statement':
            # Regular import: import module1, module2 as alias
            pattern = re.compile(r'import\s+(.+)')
            match = pattern.search(node_text)
            if match:
                modules = match.group(1).split(',')
                for module in modules:
                    module = module.strip()
                    if ' as ' in module:
                        module_name, alias = module.split(' as ')
                        results.append((module_name.strip(), [], alias.strip()))
                    else:
                        results.append((module, [], None))
        
        elif node.type == 'import_from_statement':
            # From import: from module import item1, item2 as alias
            pattern = re.compile(r'from\s+([^\s]+)\s+import\s+(.+)')
            match = pattern.search(node_text)
            if match:
                module_name = match.group(1)
                imports = match.group(2)
                
                if imports.strip() == '*':
                    imported_names = ['*']
                else:
                    imported_names = []
                    for item in imports.split(','):
                        item = item.strip()
                        if ' as ' in item:
                            original, alias = item.split(' as ')
                            imported_names.append(original.strip())
                        else:
                            imported_names.append(item)
                
                results.append((module_name, imported_names, None))
        
        return results
    
    def parse_go_import(self, node: Node, source_code: bytes) -> Optional[str]:
        """Parse Go import declaration."""
        node_text = self.extract_node_text(node, source_code)
        
        # Go imports are quoted strings
        pattern = re.compile(r'[\'"]([^\'"]+)[\'"]')
        match = pattern.search(node_text)
        if match:
            return match.group(1)
        
        return None
    
    def parse_rust_use(self, node: Node, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Parse Rust use declaration."""
        node_text = self.extract_node_text(node, source_code)
        
        # Basic use statement parsing
        pattern = re.compile(r'use\s+([^;]+);?')
        match = pattern.search(node_text)
        if match:
            use_path = match.group(1).strip()
            
            # Handle use statements with braces: use std::{io, fs}
            if '{' in use_path:
                base_path, items = use_path.split('{', 1)
                base_path = base_path.rstrip('::').strip()
                items = items.rstrip('}').strip()
                imported_items = [item.strip() for item in items.split(',')]
                
                return {
                    'module_path': base_path,
                    'imported_items': imported_items
                }
            else:
                return {
                    'module_path': use_path,
                    'imported_items': [use_path.split('::')[-1]]
                }
        
        return None
    
    def find_current_module_entity(self, entities: List[CodeEntity]) -> Optional[CodeEntity]:
        """Find the module entity for the current file."""
        for entity in entities:
            if entity.type == EntityType.MODULE:
                return entity
        return None
    
    def find_or_create_module_entity(self, module_path: str, from_file: str) -> Optional[CodeEntity]:
        """Find existing module entity or create a placeholder for cross-file resolution."""
        # This is a placeholder - in practice, we would use the cross-file resolver
        # to find the actual module entity from the entity registry
        
        # For now, create a placeholder entity that can be resolved later
        from uuid import uuid4
        from ..extraction.entity_types import CodeEntity, EntityType, SourceLocation
        
        # Convert module path to a reasonable entity name
        module_name = module_path.split('/')[-1].replace('.js', '').replace('.py', '')
        
        return CodeEntity(
            id=uuid4(),
            type=EntityType.MODULE,
            name=module_name,
            file_path=f"<unresolved>:{module_path}",  # Mark as unresolved
            location=None,
            metadata={
                'original_import_path': module_path,
                'created_by': 'import_analyzer',
                'needs_resolution': True,
                'referenced_from': from_file
            }
        )