"""
Function call relationship analyzer for detecting function and method invocations.

This analyzer detects function calls, method calls, and other invocations in source code
and creates call relationships between calling and called functions.
"""

import re
from typing import List, Dict, Set, Optional, Any
from tree_sitter import Node, Tree

from .base_analyzer import BaseRelationshipAnalyzer
from .relationship_types import CodeRelationship, RelationshipType, RelationshipContext
from ..extraction.entity_types import CodeEntity, EntityType


class FunctionCallAnalyzer(BaseRelationshipAnalyzer):
    """
    Analyzes function calls and method invocations to build call relationships.
    
    Supports:
    - Function calls (foo(), bar.baz())
    - Method calls (obj.method(), this.method())
    - Constructor calls (new Class())
    - Static method calls (Class.method())
    - Async/await calls
    """
    
    def __init__(self, parser_manager):
        super().__init__(parser_manager)
        self.supported_languages = {"javascript", "python", "go", "rust"}
        
        # Language-specific AST node types for function calls
        self.call_node_types = {
            "javascript": {
                "call_expression",
                "method_definition",
                "new_expression",
                "await_expression"
            },
            "python": {
                "call",
                "attribute"  # For method calls like obj.method()
            },
            "go": {
                "call_expression",
                "selector_expression"  # For method calls
            },
            "rust": {
                "call_expression",
                "method_call_expression"
            }
        }
    
    def get_supported_relationship_types(self) -> Set[RelationshipType]:
        """Call analyzer supports calls relationship type."""
        return {RelationshipType.CALLS}
    
    def analyze_ast(self, tree: Tree, file_path: str, language: str, 
                   entities: List[CodeEntity]) -> List[CodeRelationship]:
        """Analyze AST tree to find function call relationships."""
        
        if language not in self.supported_languages:
            return []
        
        relationships = []
        source_code = self.parser_manager.get_file_content(file_path)
        
        # Find call-related nodes based on language
        call_nodes = self.find_call_nodes(tree.root_node, language)
        
        for node in call_nodes:
            try:
                detected_relationships = self.analyze_call_node(
                    node, source_code, file_path, language, entities
                )
                relationships.extend(detected_relationships)
            except Exception as e:
                self.logger.warning(f"Error analyzing call node in {file_path}: {e}")
        
        self.logger.debug(f"Found {len(relationships)} call relationships in {file_path}")
        return relationships
    
    def find_call_nodes(self, root: Node, language: str) -> List[Node]:
        """Find all function call related nodes in the AST."""
        node_types = self.call_node_types.get(language, set())
        return self.find_nodes_by_types(root, node_types)
    
    def analyze_call_node(self, node: Node, source_code: bytes, file_path: str,
                         language: str, entities: List[CodeEntity]) -> List[CodeRelationship]:
        """Analyze a single call node to extract relationships."""
        
        relationships = []
        
        if language == "javascript":
            relationships.extend(self.analyze_javascript_call(node, source_code, file_path, entities))
        elif language == "python":
            relationships.extend(self.analyze_python_call(node, source_code, file_path, entities))
        elif language == "go":
            relationships.extend(self.analyze_go_call(node, source_code, file_path, entities))
        elif language == "rust":
            relationships.extend(self.analyze_rust_call(node, source_code, file_path, entities))
        
        return relationships
    
    def analyze_javascript_call(self, node: Node, source_code: bytes, file_path: str,
                              entities: List[CodeEntity]) -> List[CodeRelationship]:
        """Analyze JavaScript function and method calls."""
        relationships = []
        
        if node.type == "call_expression":
            call_info = self.parse_javascript_call(node, source_code)
            if call_info:
                relationships.extend(self.create_call_relationships(
                    call_info, file_path, entities, node, source_code
                ))
        
        elif node.type == "new_expression":
            # Constructor calls
            constructor_info = self.parse_javascript_constructor(node, source_code)
            if constructor_info:
                relationships.extend(self.create_call_relationships(
                    constructor_info, file_path, entities, node, source_code
                ))
        
        elif node.type == "await_expression":
            # Async function calls
            await_call = self.find_call_in_await(node)
            if await_call:
                call_info = self.parse_javascript_call(await_call, source_code)
                if call_info:
                    relationships.extend(self.create_call_relationships(
                        call_info, file_path, entities, node, source_code, is_async=True
                    ))
        
        return relationships
    
    def analyze_python_call(self, node: Node, source_code: bytes, file_path: str,
                          entities: List[CodeEntity]) -> List[CodeRelationship]:
        """Analyze Python function and method calls."""
        relationships = []
        
        if node.type == "call":
            call_info = self.parse_python_call(node, source_code)
            if call_info:
                relationships.extend(self.create_call_relationships(
                    call_info, file_path, entities, node, source_code
                ))
        
        return relationships
    
    def analyze_go_call(self, node: Node, source_code: bytes, file_path: str,
                       entities: List[CodeEntity]) -> List[CodeRelationship]:
        """Analyze Go function calls."""
        relationships = []
        
        if node.type == "call_expression":
            call_info = self.parse_go_call(node, source_code)
            if call_info:
                relationships.extend(self.create_call_relationships(
                    call_info, file_path, entities, node, source_code
                ))
        
        return relationships
    
    def analyze_rust_call(self, node: Node, source_code: bytes, file_path: str,
                         entities: List[CodeEntity]) -> List[CodeRelationship]:
        """Analyze Rust function and method calls."""
        relationships = []
        
        if node.type == "call_expression":
            call_info = self.parse_rust_call(node, source_code)
            if call_info:
                relationships.extend(self.create_call_relationships(
                    call_info, file_path, entities, node, source_code
                ))
        
        elif node.type == "method_call_expression":
            method_info = self.parse_rust_method_call(node, source_code)
            if method_info:
                relationships.extend(self.create_call_relationships(
                    method_info, file_path, entities, node, source_code
                ))
        
        return relationships
    
    def parse_javascript_call(self, node: Node, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Parse JavaScript call expression details."""
        node_text = self.extract_node_text(node, source_code)
        
        # Find the function being called
        if node.children and node.children[0]:
            callee_node = node.children[0]
            callee_text = self.extract_node_text(callee_node, source_code)
            
            # Determine call type and extract function name
            if '.' in callee_text:
                # Method call: obj.method() or Class.method()
                parts = callee_text.split('.')
                return {
                    'call_type': 'method',
                    'object': '.'.join(parts[:-1]),
                    'function_name': parts[-1],
                    'full_call': callee_text,
                    'arguments': self.extract_call_arguments(node, source_code)
                }
            else:
                # Regular function call: func()
                return {
                    'call_type': 'function',
                    'function_name': callee_text,
                    'full_call': callee_text,
                    'arguments': self.extract_call_arguments(node, source_code)
                }
        
        return None
    
    def parse_javascript_constructor(self, node: Node, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Parse JavaScript constructor call (new expression)."""
        node_text = self.extract_node_text(node, source_code)
        
        # Extract constructor name from "new ClassName()"
        pattern = re.compile(r'new\s+([^(]+)')
        match = pattern.search(node_text)
        if match:
            constructor_name = match.group(1).strip()
            return {
                'call_type': 'constructor',
                'function_name': constructor_name,
                'full_call': constructor_name,
                'arguments': self.extract_call_arguments(node, source_code)
            }
        
        return None
    
    def parse_python_call(self, node: Node, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Parse Python call expression details."""
        if not node.children:
            return None
        
        func_node = node.children[0]
        func_text = self.extract_node_text(func_node, source_code)
        
        # Handle different call patterns
        if '.' in func_text:
            # Method call: obj.method()
            parts = func_text.rsplit('.', 1)
            return {
                'call_type': 'method',
                'object': parts[0],
                'function_name': parts[1],
                'full_call': func_text,
                'arguments': self.extract_call_arguments(node, source_code)
            }
        else:
            # Function call: func()
            return {
                'call_type': 'function',
                'function_name': func_text,
                'full_call': func_text,
                'arguments': self.extract_call_arguments(node, source_code)
            }
    
    def parse_go_call(self, node: Node, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Parse Go function call details."""
        if not node.children:
            return None
        
        func_node = node.children[0]
        func_text = self.extract_node_text(func_node, source_code)
        
        # Handle package.Function() calls
        if '.' in func_text:
            parts = func_text.rsplit('.', 1)
            return {
                'call_type': 'qualified',
                'package': parts[0],
                'function_name': parts[1],
                'full_call': func_text,
                'arguments': self.extract_call_arguments(node, source_code)
            }
        else:
            return {
                'call_type': 'function',
                'function_name': func_text,
                'full_call': func_text,
                'arguments': self.extract_call_arguments(node, source_code)
            }
    
    def parse_rust_call(self, node: Node, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Parse Rust function call details."""
        if not node.children:
            return None
        
        func_node = node.children[0]
        func_text = self.extract_node_text(func_node, source_code)
        
        # Handle module::function() calls
        if '::' in func_text:
            parts = func_text.rsplit('::', 1)
            return {
                'call_type': 'qualified',
                'module': parts[0],
                'function_name': parts[1],
                'full_call': func_text,
                'arguments': self.extract_call_arguments(node, source_code)
            }
        else:
            return {
                'call_type': 'function',
                'function_name': func_text,
                'full_call': func_text,
                'arguments': self.extract_call_arguments(node, source_code)
            }
    
    def parse_rust_method_call(self, node: Node, source_code: bytes) -> Optional[Dict[str, Any]]:
        """Parse Rust method call details."""
        node_text = self.extract_node_text(node, source_code)
        
        # Method calls: obj.method()
        pattern = re.compile(r'(.+)\.([^.]+)\s*\(')
        match = pattern.search(node_text)
        if match:
            return {
                'call_type': 'method',
                'object': match.group(1),
                'function_name': match.group(2),
                'full_call': f"{match.group(1)}.{match.group(2)}",
                'arguments': self.extract_call_arguments(node, source_code)
            }
        
        return None
    
    def extract_call_arguments(self, node: Node, source_code: bytes) -> List[str]:
        """Extract function call arguments as strings."""
        arguments = []
        
        # Find argument list node
        for child in node.children:
            if child.type in ["arguments", "argument_list"]:
                for arg_child in child.children:
                    if arg_child.type not in [",", "(", ")"]:
                        arg_text = self.extract_node_text(arg_child, source_code)
                        arguments.append(arg_text)
                break
        
        return arguments
    
    def find_call_in_await(self, await_node: Node) -> Optional[Node]:
        """Find the call expression inside an await expression."""
        for child in await_node.children:
            if child.type == "call_expression":
                return child
        return None
    
    def create_call_relationships(self, call_info: Dict[str, Any], file_path: str,
                                entities: List[CodeEntity], node: Node, source_code: bytes,
                                is_async: bool = False) -> List[CodeRelationship]:
        """Create call relationships from parsed call information."""
        relationships = []
        
        function_name = call_info['function_name']
        call_type = call_info['call_type']
        
        # Find the calling function (current context)
        calling_function = self.find_containing_function(node, entities)
        if not calling_function:
            return relationships
        
        # Find potential target functions
        target_functions = self.find_target_functions(
            function_name, call_type, call_info, entities, file_path
        )
        
        for target_function in target_functions:
            context = RelationshipContext(
                line_number=self.get_node_line_number(node),
                source_code=self.extract_node_text(node, source_code),
                call_arguments=call_info.get('arguments', [])
            )
            
            # Calculate confidence based on call type and context
            confidence = self.calculate_call_confidence(call_info, calling_function, target_function)
            
            relationship = CodeRelationship(
                source_id=calling_function.id,
                target_id=target_function.id,
                relationship_type=RelationshipType.CALLS,
                confidence=confidence,
                context=context,
                metadata={
                    'call_type': call_type,
                    'function_name': function_name,
                    'is_async': is_async,
                    'argument_count': len(call_info.get('arguments', [])),
                    'full_call': call_info['full_call']
                }
            )
            relationships.append(relationship)
        
        return relationships
    
    def find_containing_function(self, call_node: Node, entities: List[CodeEntity]) -> Optional[CodeEntity]:
        """Find the function that contains this function call."""
        
        # Walk up the AST to find containing function
        current = call_node.parent
        while current:
            if current.type in ["function_declaration", "method_definition", 
                               "function_definition", "function_expression"]:
                # Extract function name and find corresponding entity
                func_name = self.extract_function_name_from_node(current)
                if func_name:
                    return self.find_entity_by_name(entities, func_name, "function")
            current = current.parent
        
        return None
    
    def extract_function_name_from_node(self, func_node: Node) -> Optional[str]:
        """Extract function name from a function declaration node."""
        for child in func_node.children:
            if child.type == "identifier":
                return child.text.decode('utf-8')
        return None
    
    def find_target_functions(self, function_name: str, call_type: str, 
                            call_info: Dict[str, Any], entities: List[CodeEntity],
                            file_path: str) -> List[CodeEntity]:
        """Find potential target functions for a function call."""
        targets = []
        
        # First, look for functions in the current file
        local_targets = self.find_entity_by_name(entities, function_name, "function")
        if local_targets:
            targets.append(local_targets)
        
        # For method calls, also look for methods in classes
        if call_type == "method" and "object" in call_info:
            # This would require more sophisticated analysis to determine the object's type
            pass
        
        # For qualified calls (package.func, module::func), this would require cross-file resolution
        if call_type in ["qualified", "method"] and not targets:
            # Create placeholder for cross-file resolution
            targets.append(self.create_placeholder_function(function_name, call_info, file_path))
        
        return targets
    
    def create_placeholder_function(self, function_name: str, call_info: Dict[str, Any], 
                                  from_file: str) -> CodeEntity:
        """Create placeholder function entity for cross-file resolution."""
        from uuid import uuid4
        from ..extraction.entity_types import EntityType
        
        return CodeEntity(
            id=uuid4(),
            type=EntityType.FUNCTION,
            name=function_name,
            file_path=f"<unresolved>:{call_info['full_call']}",
            location=None,
            metadata={
                'call_info': call_info,
                'created_by': 'call_analyzer',
                'needs_resolution': True,
                'referenced_from': from_file
            }
        )
    
    def calculate_call_confidence(self, call_info: Dict[str, Any], 
                                calling_function: CodeEntity, target_function: CodeEntity) -> float:
        """Calculate confidence score for a function call relationship."""
        
        base_confidence = 0.8
        
        # Boost confidence for local calls (same file)
        if calling_function.file_path == target_function.file_path:
            base_confidence = 0.95
        
        # Reduce confidence for method calls (more ambiguous)
        if call_info['call_type'] == 'method':
            base_confidence *= 0.9
        
        # Reduce confidence if target is a placeholder
        if target_function.metadata.get('needs_resolution'):
            base_confidence *= 0.7
        
        return min(base_confidence, 1.0)