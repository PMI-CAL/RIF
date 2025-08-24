"""
Language-specific entity extractors for supported programming languages.
"""

from typing import List, Dict, Any, Optional
import tree_sitter

from .base_extractor import BaseLanguageExtractor
from .entity_types import CodeEntity, EntityType, SourceLocation


class JavaScriptExtractor(BaseLanguageExtractor):
    """Entity extractor for JavaScript and TypeScript files."""
    
    def __init__(self):
        super().__init__('javascript')
        # JavaScript supports additional entity types
        self._supported_entity_types.extend([EntityType.INTERFACE, EntityType.ENUM])
    
    def extract_entities(self, tree: tree_sitter.Tree, file_path: str, source_code: bytes) -> List[CodeEntity]:
        """Extract entities from JavaScript/TypeScript AST."""
        entities = []
        
        if not tree.root_node:
            return entities
        
        # Extract module-level entity first
        module_entity = self._extract_module_entity(tree.root_node, file_path, source_code)
        entities.append(module_entity)
        
        # Recursively extract other entities
        self._extract_from_node(tree.root_node, file_path, source_code, entities)
        
        return entities
    
    def _extract_module_entity(self, root_node: tree_sitter.Node, file_path: str, source_code: bytes) -> CodeEntity:
        """Extract module-level information."""
        imports = []
        exports = []
        
        # Find import/export statements
        for child in root_node.children:
            if child.type == 'import_statement':
                imports.extend(self._extract_imports(child, source_code))
            elif child.type in ['export_statement', 'export_default_declaration']:
                exports.extend(self._extract_exports(child, source_code))
        
        # Get module name from file path
        module_name = file_path.split('/')[-1].replace('.js', '').replace('.ts', '').replace('.jsx', '').replace('.tsx', '')
        
        return CodeEntity(
            type=EntityType.MODULE,
            name=module_name,
            file_path=file_path,
            location=SourceLocation(line_start=1, line_end=root_node.end_point[0] + 1),
            metadata={
                'imports': imports,
                'exports': exports,
                'language': self.language
            }
        )
    
    def _extract_from_node(self, node: tree_sitter.Node, file_path: str, source_code: bytes, entities: List[CodeEntity]):
        """Recursively extract entities from AST nodes."""
        
        # Function declarations
        if node.type in ['function_declaration', 'function_expression', 'arrow_function', 'method_definition']:
            entity = self._extract_function(node, file_path, source_code)
            if entity:
                entities.append(entity)
        
        # Class declarations
        elif node.type == 'class_declaration':
            entity = self._extract_class(node, file_path, source_code)
            if entity:
                entities.append(entity)
        
        # Variable declarations
        elif node.type in ['variable_declaration', 'lexical_declaration']:
            variable_entities = self._extract_variables(node, file_path, source_code)
            entities.extend(variable_entities)
        
        # Interface declarations (TypeScript)
        elif node.type == 'interface_declaration':
            entity = self._extract_interface(node, file_path, source_code)
            if entity:
                entities.append(entity)
        
        # Enum declarations (TypeScript)
        elif node.type == 'enum_declaration':
            entity = self._extract_enum(node, file_path, source_code)
            if entity:
                entities.append(entity)
        
        # Recursively process children
        for child in node.children:
            self._extract_from_node(child, file_path, source_code, entities)
    
    def _extract_function(self, node: tree_sitter.Node, file_path: str, source_code: bytes) -> Optional[CodeEntity]:
        """Extract function entity."""
        name_node = self._find_child_by_type(node, 'identifier')
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, source_code)
        
        # Extract parameters
        parameters = []
        params_node = self._find_child_by_type(node, 'formal_parameters')
        if params_node:
            for param in params_node.children:
                if param.type == 'identifier':
                    parameters.append(self._get_node_text(param, source_code))
                elif param.type == 'required_parameter':
                    param_name = self._find_child_by_type(param, 'identifier')
                    if param_name:
                        parameters.append(self._get_node_text(param_name, source_code))
        
        # Extract return type (TypeScript)
        return_type = None
        type_annotation = self._find_child_by_type(node, 'type_annotation')
        if type_annotation:
            return_type = self._get_node_text(type_annotation, source_code)
        
        return CodeEntity(
            type=EntityType.FUNCTION,
            name=name,
            file_path=file_path,
            location=self._get_source_location(node),
            metadata={
                'parameters': parameters,
                'return_type': return_type,
                'is_async': 'async' in self._get_node_text(node, source_code)[:20],
                'complexity': self._calculate_complexity(node),
                'language': self.language
            }
        )
    
    def _extract_class(self, node: tree_sitter.Node, file_path: str, source_code: bytes) -> Optional[CodeEntity]:
        """Extract class entity."""
        name_node = self._find_child_by_type(node, 'identifier')
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, source_code)
        
        # Extract methods
        methods = []
        for child in node.children:
            if child.type in ['method_definition', 'field_definition']:
                method_name_node = self._find_child_by_type(child, 'property_identifier')
                if not method_name_node:
                    method_name_node = self._find_child_by_type(child, 'identifier')
                if method_name_node:
                    methods.append(self._get_node_text(method_name_node, source_code))
        
        # Extract inheritance
        heritage = None
        heritage_clause = self._find_child_by_type(node, 'class_heritage')
        if heritage_clause:
            extends_clause = self._find_child_by_type(heritage_clause, 'extends_clause')
            if extends_clause:
                identifier = self._find_child_by_type(extends_clause, 'identifier')
                if identifier:
                    heritage = self._get_node_text(identifier, source_code)
        
        return CodeEntity(
            type=EntityType.CLASS,
            name=name,
            file_path=file_path,
            location=self._get_source_location(node),
            metadata={
                'methods': methods,
                'extends': heritage,
                'language': self.language
            }
        )
    
    def _extract_variables(self, node: tree_sitter.Node, file_path: str, source_code: bytes) -> List[CodeEntity]:
        """Extract variable entities."""
        entities = []
        
        # Find variable declarators
        declarators = self._find_children_by_type(node, 'variable_declarator')
        
        for declarator in declarators:
            name_node = self._find_child_by_type(declarator, 'identifier')
            if name_node:
                name = self._get_node_text(name_node, source_code)
                
                # Determine if it's a constant
                declaration_kind = 'let'  # default
                if node.children and node.children[0].type in ['const', 'var', 'let']:
                    declaration_kind = node.children[0].type
                
                entity_type = EntityType.CONSTANT if declaration_kind == 'const' else EntityType.VARIABLE
                
                entities.append(CodeEntity(
                    type=entity_type,
                    name=name,
                    file_path=file_path,
                    location=self._get_source_location(declarator),
                    metadata={
                        'declaration_kind': declaration_kind,
                        'language': self.language
                    }
                ))
        
        return entities
    
    def _extract_interface(self, node: tree_sitter.Node, file_path: str, source_code: bytes) -> Optional[CodeEntity]:
        """Extract TypeScript interface entity."""
        name_node = self._find_child_by_type(node, 'type_identifier')
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, source_code)
        
        # Extract interface members
        members = []
        body = self._find_child_by_type(node, 'object_type')
        if body:
            for child in body.children:
                if child.type == 'property_signature':
                    member_name = self._find_child_by_type(child, 'property_identifier')
                    if member_name:
                        members.append(self._get_node_text(member_name, source_code))
        
        return CodeEntity(
            type=EntityType.INTERFACE,
            name=name,
            file_path=file_path,
            location=self._get_source_location(node),
            metadata={
                'members': members,
                'language': self.language
            }
        )
    
    def _extract_enum(self, node: tree_sitter.Node, file_path: str, source_code: bytes) -> Optional[CodeEntity]:
        """Extract TypeScript enum entity."""
        name_node = self._find_child_by_type(node, 'identifier')
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, source_code)
        
        # Extract enum values
        values = []
        body = self._find_child_by_type(node, 'enum_body')
        if body:
            for child in body.children:
                if child.type == 'property_identifier':
                    values.append(self._get_node_text(child, source_code))
        
        return CodeEntity(
            type=EntityType.ENUM,
            name=name,
            file_path=file_path,
            location=self._get_source_location(node),
            metadata={
                'values': values,
                'language': self.language
            }
        )
    
    def _extract_imports(self, import_node: tree_sitter.Node, source_code: bytes) -> List[str]:
        """Extract import statements."""
        imports = []
        
        # Find import specifiers
        import_clause = self._find_child_by_type(import_node, 'import_clause')
        if import_clause:
            # Named imports
            named_imports = self._find_child_by_type(import_clause, 'named_imports')
            if named_imports:
                for child in named_imports.children:
                    if child.type == 'import_specifier':
                        name_node = self._find_child_by_type(child, 'identifier')
                        if name_node:
                            imports.append(self._get_node_text(name_node, source_code))
        
        # Default import
        default_import = self._find_child_by_type(import_node, 'identifier')
        if default_import:
            imports.append(self._get_node_text(default_import, source_code))
        
        return imports
    
    def _extract_exports(self, export_node: tree_sitter.Node, source_code: bytes) -> List[str]:
        """Extract export statements."""
        exports = []
        
        # Handle different export types
        if export_node.type == 'export_statement':
            declaration = export_node.children[-1]  # Last child is usually the declaration
            if declaration.type in ['function_declaration', 'class_declaration']:
                name_node = self._find_child_by_type(declaration, 'identifier')
                if name_node:
                    exports.append(self._get_node_text(name_node, source_code))
        
        return exports


class PythonExtractor(BaseLanguageExtractor):
    """Entity extractor for Python files."""
    
    def __init__(self):
        super().__init__('python')
    
    def extract_entities(self, tree: tree_sitter.Tree, file_path: str, source_code: bytes) -> List[CodeEntity]:
        """Extract entities from Python AST."""
        entities = []
        
        if not tree.root_node:
            return entities
        
        # Extract module-level entity
        module_entity = self._extract_module_entity(tree.root_node, file_path, source_code)
        entities.append(module_entity)
        
        # Extract other entities
        self._extract_from_node(tree.root_node, file_path, source_code, entities)
        
        return entities
    
    def _extract_module_entity(self, root_node: tree_sitter.Node, file_path: str, source_code: bytes) -> CodeEntity:
        """Extract Python module information."""
        imports = []
        
        # Find import statements
        for child in root_node.children:
            if child.type in ['import_statement', 'import_from_statement']:
                imports.extend(self._extract_imports(child, source_code))
        
        module_name = file_path.split('/')[-1].replace('.py', '')
        
        return CodeEntity(
            type=EntityType.MODULE,
            name=module_name,
            file_path=file_path,
            location=SourceLocation(line_start=1, line_end=root_node.end_point[0] + 1),
            metadata={
                'imports': imports,
                'language': self.language
            }
        )
    
    def _extract_from_node(self, node: tree_sitter.Node, file_path: str, source_code: bytes, entities: List[CodeEntity]):
        """Recursively extract entities from Python AST nodes."""
        
        # Function definitions
        if node.type == 'function_definition':
            entity = self._extract_function(node, file_path, source_code)
            if entity:
                entities.append(entity)
        
        # Class definitions
        elif node.type == 'class_definition':
            entity = self._extract_class(node, file_path, source_code)
            if entity:
                entities.append(entity)
        
        # Variable assignments (simple heuristic for module-level variables)
        elif node.type == 'assignment' and self._is_module_level(node):
            variable_entities = self._extract_assignment(node, file_path, source_code)
            entities.extend(variable_entities)
        
        # Recursively process children
        for child in node.children:
            self._extract_from_node(child, file_path, source_code, entities)
    
    def _extract_function(self, node: tree_sitter.Node, file_path: str, source_code: bytes) -> Optional[CodeEntity]:
        """Extract Python function entity."""
        name_node = self._find_child_by_type(node, 'identifier')
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, source_code)
        
        # Extract parameters
        parameters = []
        params_node = self._find_child_by_type(node, 'parameters')
        if params_node:
            for child in params_node.children:
                if child.type == 'identifier':
                    parameters.append(self._get_node_text(child, source_code))
                elif child.type == 'default_parameter':
                    param_name = self._find_child_by_type(child, 'identifier')
                    if param_name:
                        parameters.append(self._get_node_text(param_name, source_code))
        
        # Extract decorators
        decorators = []
        for child in node.children:
            if child.type == 'decorator':
                decorator_name = self._find_child_by_type(child, 'identifier')
                if decorator_name:
                    decorators.append(self._get_node_text(decorator_name, source_code))
        
        return CodeEntity(
            type=EntityType.FUNCTION,
            name=name,
            file_path=file_path,
            location=self._get_source_location(node),
            metadata={
                'parameters': parameters,
                'decorators': decorators,
                'complexity': self._calculate_complexity(node),
                'language': self.language
            }
        )
    
    def _extract_class(self, node: tree_sitter.Node, file_path: str, source_code: bytes) -> Optional[CodeEntity]:
        """Extract Python class entity."""
        name_node = self._find_child_by_type(node, 'identifier')
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, source_code)
        
        # Extract methods
        methods = []
        body = self._find_child_by_type(node, 'block')
        if body:
            for child in body.children:
                if child.type == 'function_definition':
                    method_name_node = self._find_child_by_type(child, 'identifier')
                    if method_name_node:
                        methods.append(self._get_node_text(method_name_node, source_code))
        
        # Extract inheritance (argument list in class definition)
        inheritance = []
        argument_list = self._find_child_by_type(node, 'argument_list')
        if argument_list:
            for child in argument_list.children:
                if child.type == 'identifier':
                    inheritance.append(self._get_node_text(child, source_code))
        
        return CodeEntity(
            type=EntityType.CLASS,
            name=name,
            file_path=file_path,
            location=self._get_source_location(node),
            metadata={
                'methods': methods,
                'inheritance': inheritance,
                'language': self.language
            }
        )
    
    def _extract_assignment(self, node: tree_sitter.Node, file_path: str, source_code: bytes) -> List[CodeEntity]:
        """Extract variable assignments."""
        entities = []
        
        # Find assignment targets
        left_side = node.children[0] if node.children else None
        if left_side and left_side.type == 'identifier':
            name = self._get_node_text(left_side, source_code)
            
            # Check if it looks like a constant (all uppercase)
            entity_type = EntityType.CONSTANT if name.isupper() else EntityType.VARIABLE
            
            entities.append(CodeEntity(
                type=entity_type,
                name=name,
                file_path=file_path,
                location=self._get_source_location(node),
                metadata={
                    'language': self.language
                }
            ))
        
        return entities
    
    def _extract_imports(self, import_node: tree_sitter.Node, source_code: bytes) -> List[str]:
        """Extract Python import statements."""
        imports = []
        
        if import_node.type == 'import_statement':
            # import module
            for child in import_node.children:
                if child.type == 'dotted_name':
                    imports.append(self._get_node_text(child, source_code))
        elif import_node.type == 'import_from_statement':
            # from module import name
            module_name = None
            for child in import_node.children:
                if child.type == 'dotted_name':
                    module_name = self._get_node_text(child, source_code)
                elif child.type == 'import_list':
                    for import_child in child.children:
                        if import_child.type == 'identifier':
                            imported_name = self._get_node_text(import_child, source_code)
                            if module_name:
                                imports.append(f"{module_name}.{imported_name}")
                            else:
                                imports.append(imported_name)
        
        return imports
    
    def _is_module_level(self, node: tree_sitter.Node) -> bool:
        """Check if a node is at module level (simple heuristic)."""
        # This is a simplified check - in a full implementation, 
        # we'd track the scope depth more carefully
        return True  # For now, assume all assignments we see are module-level


class GoExtractor(BaseLanguageExtractor):
    """Entity extractor for Go files."""
    
    def __init__(self):
        super().__init__('go')
        self._supported_entity_types.extend([EntityType.INTERFACE])
    
    def extract_entities(self, tree: tree_sitter.Tree, file_path: str, source_code: bytes) -> List[CodeEntity]:
        """Extract entities from Go AST."""
        entities = []
        
        if not tree.root_node:
            return entities
        
        # Extract package-level entity
        package_entity = self._extract_package_entity(tree.root_node, file_path, source_code)
        entities.append(package_entity)
        
        # Extract other entities
        self._extract_from_node(tree.root_node, file_path, source_code, entities)
        
        return entities
    
    def _extract_package_entity(self, root_node: tree_sitter.Node, file_path: str, source_code: bytes) -> CodeEntity:
        """Extract Go package information."""
        package_name = "main"  # default
        imports = []
        
        # Find package declaration
        for child in root_node.children:
            if child.type == 'package_clause':
                name_node = self._find_child_by_type(child, 'package_identifier')
                if name_node:
                    package_name = self._get_node_text(name_node, source_code)
            elif child.type == 'import_declaration':
                imports.extend(self._extract_imports(child, source_code))
        
        return CodeEntity(
            type=EntityType.MODULE,
            name=package_name,
            file_path=file_path,
            location=SourceLocation(line_start=1, line_end=root_node.end_point[0] + 1),
            metadata={
                'imports': imports,
                'package': package_name,
                'language': self.language
            }
        )
    
    def _extract_from_node(self, node: tree_sitter.Node, file_path: str, source_code: bytes, entities: List[CodeEntity]):
        """Recursively extract entities from Go AST nodes."""
        
        # Function declarations
        if node.type == 'function_declaration':
            entity = self._extract_function(node, file_path, source_code)
            if entity:
                entities.append(entity)
        
        # Method declarations
        elif node.type == 'method_declaration':
            entity = self._extract_method(node, file_path, source_code)
            if entity:
                entities.append(entity)
        
        # Type declarations (structs, interfaces)
        elif node.type == 'type_declaration':
            type_entities = self._extract_type_declaration(node, file_path, source_code)
            entities.extend(type_entities)
        
        # Variable declarations
        elif node.type == 'var_declaration':
            variable_entities = self._extract_var_declaration(node, file_path, source_code)
            entities.extend(variable_entities)
        
        # Constant declarations
        elif node.type == 'const_declaration':
            constant_entities = self._extract_const_declaration(node, file_path, source_code)
            entities.extend(constant_entities)
        
        # Recursively process children
        for child in node.children:
            self._extract_from_node(child, file_path, source_code, entities)
    
    def _extract_function(self, node: tree_sitter.Node, file_path: str, source_code: bytes) -> Optional[CodeEntity]:
        """Extract Go function entity."""
        name_node = self._find_child_by_type(node, 'identifier')
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, source_code)
        
        # Extract parameters
        parameters = []
        param_list = self._find_child_by_type(node, 'parameter_list')
        if param_list:
            for child in param_list.children:
                if child.type == 'parameter_declaration':
                    param_name = self._find_child_by_type(child, 'identifier')
                    if param_name:
                        parameters.append(self._get_node_text(param_name, source_code))
        
        return CodeEntity(
            type=EntityType.FUNCTION,
            name=name,
            file_path=file_path,
            location=self._get_source_location(node),
            metadata={
                'parameters': parameters,
                'complexity': self._calculate_complexity(node),
                'language': self.language
            }
        )
    
    def _extract_method(self, node: tree_sitter.Node, file_path: str, source_code: bytes) -> Optional[CodeEntity]:
        """Extract Go method entity."""
        name_node = self._find_child_by_type(node, 'field_identifier')
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, source_code)
        
        # Extract receiver type
        receiver = None
        receiver_node = self._find_child_by_type(node, 'parameter_list')
        if receiver_node:
            type_node = self._find_child_by_type(receiver_node, 'type_identifier')
            if type_node:
                receiver = self._get_node_text(type_node, source_code)
        
        return CodeEntity(
            type=EntityType.FUNCTION,
            name=f"{receiver}.{name}" if receiver else name,
            file_path=file_path,
            location=self._get_source_location(node),
            metadata={
                'receiver': receiver,
                'is_method': True,
                'language': self.language
            }
        )
    
    def _extract_type_declaration(self, node: tree_sitter.Node, file_path: str, source_code: bytes) -> List[CodeEntity]:
        """Extract Go type declarations (structs, interfaces)."""
        entities = []
        
        type_spec = self._find_child_by_type(node, 'type_spec')
        if not type_spec:
            return entities
        
        name_node = self._find_child_by_type(type_spec, 'type_identifier')
        if not name_node:
            return entities
        
        name = self._get_node_text(name_node, source_code)
        
        # Check if it's a struct or interface
        type_def = type_spec.children[-1]  # Last child should be the type definition
        
        if type_def.type == 'struct_type':
            entities.append(self._extract_struct(name, type_def, file_path, source_code, node))
        elif type_def.type == 'interface_type':
            entities.append(self._extract_interface_go(name, type_def, file_path, source_code, node))
        
        return entities
    
    def _extract_struct(self, name: str, struct_node: tree_sitter.Node, file_path: str, source_code: bytes, declaration_node: tree_sitter.Node) -> CodeEntity:
        """Extract Go struct entity."""
        fields = []
        
        field_list = self._find_child_by_type(struct_node, 'field_declaration_list')
        if field_list:
            for child in field_list.children:
                if child.type == 'field_declaration':
                    field_name = self._find_child_by_type(child, 'field_identifier')
                    if field_name:
                        fields.append(self._get_node_text(field_name, source_code))
        
        return CodeEntity(
            type=EntityType.CLASS,  # Treating struct as class-like
            name=name,
            file_path=file_path,
            location=self._get_source_location(declaration_node),
            metadata={
                'fields': fields,
                'is_struct': True,
                'language': self.language
            }
        )
    
    def _extract_interface_go(self, name: str, interface_node: tree_sitter.Node, file_path: str, source_code: bytes, declaration_node: tree_sitter.Node) -> CodeEntity:
        """Extract Go interface entity."""
        methods = []
        
        method_spec_list = self._find_child_by_type(interface_node, 'method_spec_list')
        if method_spec_list:
            for child in method_spec_list.children:
                if child.type == 'method_spec':
                    method_name = self._find_child_by_type(child, 'field_identifier')
                    if method_name:
                        methods.append(self._get_node_text(method_name, source_code))
        
        return CodeEntity(
            type=EntityType.INTERFACE,
            name=name,
            file_path=file_path,
            location=self._get_source_location(declaration_node),
            metadata={
                'methods': methods,
                'language': self.language
            }
        )
    
    def _extract_var_declaration(self, node: tree_sitter.Node, file_path: str, source_code: bytes) -> List[CodeEntity]:
        """Extract Go variable declarations."""
        entities = []
        
        var_spec = self._find_child_by_type(node, 'var_spec')
        if var_spec:
            name_node = self._find_child_by_type(var_spec, 'identifier')
            if name_node:
                name = self._get_node_text(name_node, source_code)
                entities.append(CodeEntity(
                    type=EntityType.VARIABLE,
                    name=name,
                    file_path=file_path,
                    location=self._get_source_location(node),
                    metadata={'language': self.language}
                ))
        
        return entities
    
    def _extract_const_declaration(self, node: tree_sitter.Node, file_path: str, source_code: bytes) -> List[CodeEntity]:
        """Extract Go constant declarations."""
        entities = []
        
        const_spec = self._find_child_by_type(node, 'const_spec')
        if const_spec:
            name_node = self._find_child_by_type(const_spec, 'identifier')
            if name_node:
                name = self._get_node_text(name_node, source_code)
                entities.append(CodeEntity(
                    type=EntityType.CONSTANT,
                    name=name,
                    file_path=file_path,
                    location=self._get_source_location(node),
                    metadata={'language': self.language}
                ))
        
        return entities
    
    def _extract_imports(self, import_node: tree_sitter.Node, source_code: bytes) -> List[str]:
        """Extract Go import statements."""
        imports = []
        
        import_spec_list = self._find_child_by_type(import_node, 'import_spec_list')
        if import_spec_list:
            for child in import_spec_list.children:
                if child.type == 'import_spec':
                    path_node = self._find_child_by_type(child, 'interpreted_string_literal')
                    if path_node:
                        import_path = self._get_node_text(path_node, source_code).strip('"')
                        imports.append(import_path)
        
        return imports


class RustExtractor(BaseLanguageExtractor):
    """Entity extractor for Rust files."""
    
    def __init__(self):
        super().__init__('rust')
        self._supported_entity_types.extend([EntityType.INTERFACE, EntityType.ENUM])
    
    def extract_entities(self, tree: tree_sitter.Tree, file_path: str, source_code: bytes) -> List[CodeEntity]:
        """Extract entities from Rust AST."""
        entities = []
        
        if not tree.root_node:
            return entities
        
        # Extract crate-level entity
        crate_entity = self._extract_crate_entity(tree.root_node, file_path, source_code)
        entities.append(crate_entity)
        
        # Extract other entities
        self._extract_from_node(tree.root_node, file_path, source_code, entities)
        
        return entities
    
    def _extract_crate_entity(self, root_node: tree_sitter.Node, file_path: str, source_code: bytes) -> CodeEntity:
        """Extract Rust crate/module information."""
        imports = []
        
        # Find use declarations
        for child in root_node.children:
            if child.type == 'use_declaration':
                imports.extend(self._extract_uses(child, source_code))
        
        crate_name = file_path.split('/')[-1].replace('.rs', '')
        
        return CodeEntity(
            type=EntityType.MODULE,
            name=crate_name,
            file_path=file_path,
            location=SourceLocation(line_start=1, line_end=root_node.end_point[0] + 1),
            metadata={
                'imports': imports,
                'language': self.language
            }
        )
    
    def _extract_from_node(self, node: tree_sitter.Node, file_path: str, source_code: bytes, entities: List[CodeEntity]):
        """Recursively extract entities from Rust AST nodes."""
        
        # Function definitions
        if node.type == 'function_item':
            entity = self._extract_function(node, file_path, source_code)
            if entity:
                entities.append(entity)
        
        # Struct definitions
        elif node.type == 'struct_item':
            entity = self._extract_struct(node, file_path, source_code)
            if entity:
                entities.append(entity)
        
        # Enum definitions
        elif node.type == 'enum_item':
            entity = self._extract_enum(node, file_path, source_code)
            if entity:
                entities.append(entity)
        
        # Trait definitions
        elif node.type == 'trait_item':
            entity = self._extract_trait(node, file_path, source_code)
            if entity:
                entities.append(entity)
        
        # Recursively process children
        for child in node.children:
            self._extract_from_node(child, file_path, source_code, entities)
    
    def _extract_function(self, node: tree_sitter.Node, file_path: str, source_code: bytes) -> Optional[CodeEntity]:
        """Extract Rust function entity."""
        name_node = self._find_child_by_type(node, 'identifier')
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, source_code)
        
        # Extract parameters
        parameters = []
        param_list = self._find_child_by_type(node, 'parameters')
        if param_list:
            for child in param_list.children:
                if child.type == 'parameter':
                    param_pattern = self._find_child_by_type(child, 'identifier')
                    if param_pattern:
                        parameters.append(self._get_node_text(param_pattern, source_code))
        
        return CodeEntity(
            type=EntityType.FUNCTION,
            name=name,
            file_path=file_path,
            location=self._get_source_location(node),
            metadata={
                'parameters': parameters,
                'complexity': self._calculate_complexity(node),
                'language': self.language
            }
        )
    
    def _extract_struct(self, node: tree_sitter.Node, file_path: str, source_code: bytes) -> Optional[CodeEntity]:
        """Extract Rust struct entity."""
        name_node = self._find_child_by_type(node, 'type_identifier')
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, source_code)
        
        # Extract fields
        fields = []
        field_list = self._find_child_by_type(node, 'field_declaration_list')
        if field_list:
            for child in field_list.children:
                if child.type == 'field_declaration':
                    field_name = self._find_child_by_type(child, 'field_identifier')
                    if field_name:
                        fields.append(self._get_node_text(field_name, source_code))
        
        return CodeEntity(
            type=EntityType.CLASS,  # Treating struct as class-like
            name=name,
            file_path=file_path,
            location=self._get_source_location(node),
            metadata={
                'fields': fields,
                'is_struct': True,
                'language': self.language
            }
        )
    
    def _extract_enum(self, node: tree_sitter.Node, file_path: str, source_code: bytes) -> Optional[CodeEntity]:
        """Extract Rust enum entity."""
        name_node = self._find_child_by_type(node, 'type_identifier')
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, source_code)
        
        # Extract variants
        variants = []
        enum_list = self._find_child_by_type(node, 'enum_variant_list')
        if enum_list:
            for child in enum_list.children:
                if child.type == 'enum_variant':
                    variant_name = self._find_child_by_type(child, 'identifier')
                    if variant_name:
                        variants.append(self._get_node_text(variant_name, source_code))
        
        return CodeEntity(
            type=EntityType.ENUM,
            name=name,
            file_path=file_path,
            location=self._get_source_location(node),
            metadata={
                'variants': variants,
                'language': self.language
            }
        )
    
    def _extract_trait(self, node: tree_sitter.Node, file_path: str, source_code: bytes) -> Optional[CodeEntity]:
        """Extract Rust trait entity."""
        name_node = self._find_child_by_type(node, 'type_identifier')
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, source_code)
        
        # Extract trait methods
        methods = []
        declaration_list = self._find_child_by_type(node, 'declaration_list')
        if declaration_list:
            for child in declaration_list.children:
                if child.type == 'function_signature_item':
                    method_name = self._find_child_by_type(child, 'identifier')
                    if method_name:
                        methods.append(self._get_node_text(method_name, source_code))
        
        return CodeEntity(
            type=EntityType.INTERFACE,  # Treating trait as interface-like
            name=name,
            file_path=file_path,
            location=self._get_source_location(node),
            metadata={
                'methods': methods,
                'is_trait': True,
                'language': self.language
            }
        )
    
    def _extract_uses(self, use_node: tree_sitter.Node, source_code: bytes) -> List[str]:
        """Extract Rust use statements."""
        imports = []
        
        # This is a simplified implementation - full Rust use parsing is quite complex
        use_tree = use_node.children[-1]  # Usually the last child
        if use_tree:
            imports.append(self._get_node_text(use_tree, source_code))
        
        return imports