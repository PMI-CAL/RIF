"""
Custom exceptions for the tree-sitter parsing infrastructure.

These exceptions provide specific error handling for different
failure modes in the parsing system.
"""


class ParsingError(Exception):
    """Base exception for all parsing-related errors."""
    
    def __init__(self, message, file_path=None, language=None, original_error=None):
        super().__init__(message)
        self.file_path = file_path
        self.language = language
        self.original_error = original_error


class LanguageNotSupportedError(ParsingError):
    """Raised when attempting to parse an unsupported language."""
    
    def __init__(self, language, file_path=None, supported_languages=None):
        message = f"Language '{language}' is not supported"
        if supported_languages:
            message += f". Supported languages: {', '.join(supported_languages)}"
        super().__init__(message, file_path=file_path, language=language)
        self.supported_languages = supported_languages or []


class GrammarNotFoundError(ParsingError):
    """Raised when tree-sitter grammar cannot be found or loaded."""
    
    def __init__(self, language, grammar_path=None):
        message = f"Tree-sitter grammar not found for language '{language}'"
        if grammar_path:
            message += f" at path: {grammar_path}"
        super().__init__(message, language=language)
        self.grammar_path = grammar_path


class CacheError(ParsingError):
    """Raised when AST cache operations fail."""
    
    def __init__(self, message, operation=None, cache_key=None):
        super().__init__(message)
        self.operation = operation
        self.cache_key = cache_key


class IncrementalParsingError(ParsingError):
    """Raised when incremental parsing fails and full re-parse is needed."""
    
    def __init__(self, file_path, original_error=None):
        message = f"Incremental parsing failed for {file_path}, full re-parse required"
        super().__init__(message, file_path=file_path, original_error=original_error)


class MemoryLimitExceededError(ParsingError):
    """Raised when parser memory usage exceeds configured limits."""
    
    def __init__(self, current_usage, limit, file_path=None):
        message = f"Memory usage {current_usage}MB exceeds limit of {limit}MB"
        if file_path:
            message += f" while parsing {file_path}"
        super().__init__(message, file_path=file_path)
        self.current_usage = current_usage
        self.limit = limit


class ThreadSafetyError(ParsingError):
    """Raised when thread safety violations are detected."""
    
    def __init__(self, message, thread_id=None, resource=None):
        super().__init__(message)
        self.thread_id = thread_id
        self.resource = resource