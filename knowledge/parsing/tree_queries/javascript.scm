; JavaScript tree-sitter queries for semantic analysis
; These patterns match common JavaScript constructs for knowledge extraction

; Function declarations
(function_declaration
  name: (identifier) @function.name
  parameters: (formal_parameters) @function.parameters
  body: (statement_block) @function.body) @function.declaration

; Function expressions
(function_expression
  name: (identifier)? @function.name
  parameters: (formal_parameters) @function.parameters
  body: (statement_block) @function.body) @function.expression

; Arrow functions
(arrow_function
  parameters: (formal_parameters) @function.parameters
  body: (_) @function.body) @function.arrow

; Class declarations
(class_declaration
  name: (identifier) @class.name
  superclass: (class_heritage)? @class.superclass
  body: (class_body) @class.body) @class.declaration

; Method definitions
(method_definition
  name: (property_name) @method.name
  parameters: (formal_parameters) @method.parameters
  body: (statement_block) @method.body) @method.definition

; Variable declarations
(variable_declarator
  name: (identifier) @variable.name
  value: (_)? @variable.value) @variable.declaration

; Import statements
(import_statement
  source: (string) @import.source) @import.statement

; Import specifiers
(import_specifier
  name: (identifier) @import.name
  alias: (identifier)? @import.alias) @import.specifier

; Export statements
(export_statement) @export.statement

; Call expressions
(call_expression
  function: (_) @call.function
  arguments: (arguments) @call.arguments) @call.expression

; Object properties
(property_definition
  key: (property_name) @property.key
  value: (_) @property.value) @property.definition

; JSX elements (if applicable)
(jsx_element
  open_tag: (jsx_opening_element) @jsx.open_tag
  close_tag: (jsx_closing_element)? @jsx.close_tag) @jsx.element

; Comments
(comment) @comment

; String literals
(string) @string.literal

; Template literals
(template_literal) @string.template

; Identifiers
(identifier) @identifier

; This query captures the main structural elements needed for code analysis