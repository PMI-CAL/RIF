; Rust tree-sitter queries for semantic analysis
; These patterns match common Rust constructs for knowledge extraction

; Function definitions
(function_item
  name: (identifier) @function.name
  parameters: (parameters) @function.parameters
  return_type: (type_annotation)? @function.return_type
  body: (block) @function.body) @function.definition

; Function signatures (in traits)
(function_signature_item
  name: (identifier) @function.name
  parameters: (parameters) @function.parameters
  return_type: (type_annotation)? @function.return_type) @function.signature

; Struct definitions
(struct_item
  name: (type_identifier) @struct.name
  body: (field_declaration_list)? @struct.fields) @struct.definition

; Enum definitions
(enum_item
  name: (type_identifier) @enum.name
  body: (enum_variant_list) @enum.variants) @enum.definition

; Trait definitions
(trait_item
  name: (type_identifier) @trait.name
  body: (declaration_list) @trait.body) @trait.definition

; Implementation blocks
(impl_item
  trait: (type_identifier)? @impl.trait
  type: (_) @impl.type
  body: (declaration_list) @impl.body) @impl.block

; Module declarations
(mod_item
  name: (identifier) @module.name
  body: (declaration_list)? @module.body) @module.definition

; Use declarations (imports)
(use_declaration
  argument: (use_clause) @use.clause) @use.declaration

; Let statements (variable bindings)
(let_declaration
  pattern: (_) @variable.pattern
  type: (type_annotation)? @variable.type
  value: (_)? @variable.value) @variable.declaration

; Static variables
(static_item
  name: (identifier) @static.name
  type: (_) @static.type
  value: (_)? @static.value) @static.declaration

; Constants
(const_item
  name: (identifier) @const.name
  type: (_) @const.type
  value: (_) @const.value) @const.declaration

; Function calls
(call_expression
  function: (_) @call.function
  arguments: (arguments) @call.arguments) @call.expression

; Macro calls
(macro_invocation
  macro: (identifier) @macro.name
  (token_tree) @macro.body) @macro.invocation

; Match expressions
(match_expression
  value: (_) @match.value
  body: (match_block) @match.body) @match.expression

; If expressions
(if_expression
  condition: (_) @if.condition
  consequence: (block) @if.body
  alternative: (_)? @if.else) @if.expression

; Loop expressions
(loop_expression
  body: (block) @loop.body) @loop.expression

; While expressions
(while_expression
  condition: (_) @while.condition
  body: (block) @while.body) @while.expression

; For expressions
(for_expression
  pattern: (_) @for.pattern
  value: (_) @for.iterable
  body: (block) @for.body) @for.expression

; Closure expressions
(closure_expression
  parameters: (closure_parameters) @closure.parameters
  body: (_) @closure.body) @closure.expression

; Attribute macros
(attribute_item
  (attribute) @attribute
  item: (_) @attribute.target) @attribute.usage

; Comments
(line_comment) @comment.line
(block_comment) @comment.block

; String literals
(string_literal) @string.literal
(raw_string_literal) @string.raw

; Identifiers
(identifier) @identifier
(type_identifier) @type.identifier

; Field access
(field_expression
  value: (_) @field.object
  field: (field_identifier) @field.name) @field.access

; This query captures the main structural elements needed for Rust code analysis