; Python tree-sitter queries for semantic analysis
; These patterns match common Python constructs for knowledge extraction

; Function definitions
(function_definition
  name: (identifier) @function.name
  parameters: (parameters) @function.parameters
  body: (block) @function.body) @function.definition

; Class definitions
(class_definition
  name: (identifier) @class.name
  superclasses: (argument_list)? @class.superclasses
  body: (block) @class.body) @class.definition

; Method definitions (functions inside classes)
(class_definition
  body: (block
    (function_definition
      name: (identifier) @method.name
      parameters: (parameters) @method.parameters
      body: (block) @method.body) @method.definition))

; Variable assignments
(assignment
  left: (identifier) @variable.name
  right: (_) @variable.value) @variable.assignment

; Import statements
(import_statement
  name: (dotted_name) @import.name) @import.statement

; Import from statements
(import_from_statement
  module_name: (relative_import)? @import.module
  name: (dotted_name)? @import.name) @import.from

; Decorators
(decorated_definition
  (decorator
    (identifier) @decorator.name) @decorator
  definition: (_) @decorator.target) @decorator.usage

; Call expressions
(call
  function: (_) @call.function
  arguments: (argument_list) @call.arguments) @call.expression

; List comprehensions
(list_comprehension
  body: (_) @comprehension.body
  generators: (for_in_clause) @comprehension.generator) @comprehension.list

; Dictionary comprehensions
(dictionary_comprehension
  body: (pair) @comprehension.body
  generators: (for_in_clause) @comprehension.generator) @comprehension.dict

; Try-except blocks
(try_statement
  body: (block) @try.body
  (except_clause
    type: (_)? @except.type
    name: (identifier)? @except.name
    body: (block) @except.body) @except.clause) @try.statement

; With statements
(with_statement
  (with_clause
    context: (_) @with.context
    alias: (identifier)? @with.alias) @with.clause
  body: (block) @with.body) @with.statement

; Lambda expressions
(lambda
  parameters: (lambda_parameters)? @lambda.parameters
  body: (_) @lambda.body) @lambda.expression

; Comments
(comment) @comment

; String literals
(string) @string.literal

; Identifiers
(identifier) @identifier

; Attribute access
(attribute
  object: (_) @attribute.object
  attribute: (identifier) @attribute.name) @attribute.access

; This query captures the main structural elements needed for Python code analysis