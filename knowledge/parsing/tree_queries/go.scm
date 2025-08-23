; Go tree-sitter queries for semantic analysis
; These patterns match common Go constructs for knowledge extraction

; Function declarations
(function_declaration
  name: (identifier) @function.name
  parameters: (parameter_list) @function.parameters
  result: (parameter_list)? @function.result
  body: (block) @function.body) @function.declaration

; Method declarations
(method_declaration
  receiver: (parameter_list) @method.receiver
  name: (identifier) @method.name
  parameters: (parameter_list) @method.parameters
  result: (parameter_list)? @method.result
  body: (block) @method.body) @method.declaration

; Type declarations
(type_declaration
  (type_spec
    name: (type_identifier) @type.name
    type: (_) @type.definition)) @type.declaration

; Struct types
(struct_type
  (field_declaration_list
    (field_declaration
      name: (field_identifier) @field.name
      type: (_) @field.type))) @struct.type

; Interface types
(interface_type
  (method_spec_list
    (method_spec
      name: (field_identifier) @interface.method.name
      parameters: (parameter_list) @interface.method.parameters
      result: (parameter_list)? @interface.method.result))) @interface.type

; Variable declarations
(var_declaration
  (var_spec
    name: (identifier) @variable.name
    type: (_)? @variable.type
    value: (_)? @variable.value)) @variable.declaration

; Short variable declarations
(short_var_declaration
  left: (expression_list
    (identifier) @variable.name)
  right: (expression_list) @variable.value) @variable.short

; Constants
(const_declaration
  (const_spec
    name: (identifier) @constant.name
    type: (_)? @constant.type
    value: (_) @constant.value)) @constant.declaration

; Package declaration
(package_clause
  (package_identifier) @package.name) @package.declaration

; Import declarations
(import_declaration
  (import_spec
    name: (package_identifier)? @import.alias
    path: (interpreted_string_literal) @import.path)) @import.declaration

; Function calls
(call_expression
  function: (_) @call.function
  arguments: (argument_list) @call.arguments) @call.expression

; For statements
(for_statement
  condition: (_)? @for.condition
  body: (block) @for.body) @for.statement

; If statements
(if_statement
  condition: (_) @if.condition
  consequence: (block) @if.body
  alternative: (_)? @if.else) @if.statement

; Switch statements
(expression_switch_statement
  value: (_)? @switch.value
  body: (expression_switch_body) @switch.body) @switch.statement

; Select statements
(select_statement
  body: (communication_clause)* @select.clauses) @select.statement

; Go statements (goroutines)
(go_statement
  (call_expression) @goroutine.call) @goroutine.statement

; Defer statements
(defer_statement
  (call_expression) @defer.call) @defer.statement

; Comments
(comment) @comment

; String literals
(interpreted_string_literal) @string.literal
(raw_string_literal) @string.raw

; Identifiers
(identifier) @identifier
(type_identifier) @type.identifier

; This query captures the main structural elements needed for Go code analysis