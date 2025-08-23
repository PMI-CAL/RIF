"""
Text processing for code entities to prepare them for embedding generation.
"""

import re
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..extraction.entity_types import CodeEntity, EntityType


@dataclass
class ProcessedText:
    """Processed text ready for embedding generation."""
    text: str
    content_hash: str
    metadata: Dict[str, Any]
    
    @classmethod
    def create(cls, text: str, metadata: Dict[str, Any] = None) -> 'ProcessedText':
        """Create ProcessedText with hash generation."""
        clean_text = cls._normalize_text(text)
        content_hash = hashlib.sha256(clean_text.encode()).hexdigest()[:16]
        
        return cls(
            text=clean_text,
            content_hash=content_hash,
            metadata=metadata or {}
        )
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for consistent hashing."""
        # Remove extra whitespace and normalize
        normalized = re.sub(r'\s+', ' ', text.strip())
        return normalized.lower()


class TextProcessor:
    """Base class for processing text into embedding-ready format."""
    
    def __init__(self):
        self.stop_words = self._load_stop_words()
    
    def _load_stop_words(self) -> set:
        """Load common stop words to exclude from embeddings."""
        # Basic stop words for code analysis
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'if', 'then', 'else', 'when', 'where',
            'why', 'how', 'what', 'which', 'who', 'whom', 'whose', 'can', 'may',
            'might', 'must', 'shall', 'return', 'def', 'class', 'import', 'from',
            'as', 'try', 'except', 'finally', 'with', 'pass', 'break', 'continue'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for embedding."""
        if not text:
            return ""
        
        # Remove common code artifacts
        text = re.sub(r'[^\w\s_]', ' ', text)  # Remove punctuation except underscore
        text = re.sub(r'\b\d+\b', ' NUM ', text)  # Replace numbers with NUM token
        text = re.sub(r'_+', '_', text)  # Normalize underscores
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Convert to lowercase and strip
        text = text.lower().strip()
        
        # Remove stop words
        words = text.split()
        words = [word for word in words if word not in self.stop_words and len(word) > 1]
        
        return ' '.join(words)
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract important keywords from text."""
        clean_text = self.clean_text(text)
        words = clean_text.split()
        
        # Simple frequency-based keyword extraction
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in keywords[:max_keywords] if freq > 1]


class EntityTextExtractor(TextProcessor):
    """Specialized text extractor for different code entity types."""
    
    def extract_text(self, entity: CodeEntity) -> ProcessedText:
        """Extract meaningful text from a code entity for embedding."""
        
        if entity.type == EntityType.FUNCTION:
            return self._extract_function_text(entity)
        elif entity.type == EntityType.CLASS:
            return self._extract_class_text(entity)
        elif entity.type == EntityType.MODULE:
            return self._extract_module_text(entity)
        elif entity.type in [EntityType.VARIABLE, EntityType.CONSTANT]:
            return self._extract_variable_text(entity)
        elif entity.type == EntityType.INTERFACE:
            return self._extract_interface_text(entity)
        elif entity.type == EntityType.ENUM:
            return self._extract_enum_text(entity)
        else:
            # Fallback to basic extraction
            return self._extract_basic_text(entity)
    
    def _extract_function_text(self, entity: CodeEntity) -> ProcessedText:
        """Extract text from function entity."""
        parts = []
        
        # Function name (most important)
        parts.append(f"function {entity.name}")
        
        # Parameters
        if entity.metadata.get('parameters'):
            params = entity.metadata['parameters']
            parts.append(f"parameters {' '.join(params)}")
        
        # Documentation
        if entity.metadata.get('docstring'):
            docstring = self.clean_text(entity.metadata['docstring'])
            parts.append(f"purpose {docstring}")
        
        # Return type
        if entity.metadata.get('return_type'):
            parts.append(f"returns {entity.metadata['return_type']}")
        
        # Decorators (Python)
        if entity.metadata.get('decorators'):
            decorators = entity.metadata['decorators']
            parts.append(f"decorators {' '.join(decorators)}")
        
        # Language context
        language = entity.metadata.get('language', 'unknown')
        parts.append(f"language {language}")
        
        text = ' '.join(parts)
        
        return ProcessedText.create(text, {
            'entity_type': 'function',
            'entity_id': str(entity.id),
            'name': entity.name,
            'complexity': entity.metadata.get('complexity', 1)
        })
    
    def _extract_class_text(self, entity: CodeEntity) -> ProcessedText:
        """Extract text from class entity."""
        parts = []
        
        # Class name
        parts.append(f"class {entity.name}")
        
        # Methods
        if entity.metadata.get('methods'):
            methods = entity.metadata['methods']
            parts.append(f"methods {' '.join(methods)}")
        
        # Inheritance
        inheritance = entity.metadata.get('extends') or entity.metadata.get('inheritance', [])
        if inheritance:
            if isinstance(inheritance, str):
                parts.append(f"extends {inheritance}")
            else:
                parts.append(f"inherits {' '.join(inheritance)}")
        
        # Documentation
        if entity.metadata.get('docstring'):
            docstring = self.clean_text(entity.metadata['docstring'])
            parts.append(f"purpose {docstring}")
        
        # Fields/attributes
        if entity.metadata.get('fields'):
            fields = entity.metadata['fields']
            parts.append(f"fields {' '.join(fields)}")
        
        # Language context
        language = entity.metadata.get('language', 'unknown')
        parts.append(f"language {language}")
        
        text = ' '.join(parts)
        
        return ProcessedText.create(text, {
            'entity_type': 'class',
            'entity_id': str(entity.id),
            'name': entity.name,
            'method_count': len(entity.metadata.get('methods', []))
        })
    
    def _extract_module_text(self, entity: CodeEntity) -> ProcessedText:
        """Extract text from module entity."""
        parts = []
        
        # Module name
        parts.append(f"module {entity.name}")
        
        # Imports
        if entity.metadata.get('imports'):
            imports = entity.metadata['imports']
            # Take first few imports to avoid very long text
            import_sample = imports[:10] if len(imports) > 10 else imports
            parts.append(f"imports {' '.join(import_sample)}")
        
        # Exports
        if entity.metadata.get('exports'):
            exports = entity.metadata['exports']
            parts.append(f"exports {' '.join(exports)}")
        
        # Package context
        if entity.metadata.get('package'):
            parts.append(f"package {entity.metadata['package']}")
        
        # Language context
        language = entity.metadata.get('language', 'unknown')
        parts.append(f"language {language}")
        
        text = ' '.join(parts)
        
        return ProcessedText.create(text, {
            'entity_type': 'module',
            'entity_id': str(entity.id),
            'name': entity.name,
            'import_count': len(entity.metadata.get('imports', []))
        })
    
    def _extract_variable_text(self, entity: CodeEntity) -> ProcessedText:
        """Extract text from variable/constant entity."""
        parts = []
        
        # Variable/constant name
        entity_type_name = "constant" if entity.type == EntityType.CONSTANT else "variable"
        parts.append(f"{entity_type_name} {entity.name}")
        
        # Declaration type
        if entity.metadata.get('declaration_kind'):
            parts.append(f"declared {entity.metadata['declaration_kind']}")
        
        # Type information
        if entity.metadata.get('type_hint'):
            parts.append(f"type {entity.metadata['type_hint']}")
        
        # Language context
        language = entity.metadata.get('language', 'unknown')
        parts.append(f"language {language}")
        
        text = ' '.join(parts)
        
        return ProcessedText.create(text, {
            'entity_type': entity_type_name,
            'entity_id': str(entity.id),
            'name': entity.name
        })
    
    def _extract_interface_text(self, entity: CodeEntity) -> ProcessedText:
        """Extract text from interface entity."""
        parts = []
        
        # Interface name
        parts.append(f"interface {entity.name}")
        
        # Methods
        if entity.metadata.get('methods'):
            methods = entity.metadata['methods']
            parts.append(f"methods {' '.join(methods)}")
        
        # Members (for TypeScript interfaces)
        if entity.metadata.get('members'):
            members = entity.metadata['members']
            parts.append(f"members {' '.join(members)}")
        
        # Language context
        language = entity.metadata.get('language', 'unknown')
        parts.append(f"language {language}")
        
        text = ' '.join(parts)
        
        return ProcessedText.create(text, {
            'entity_type': 'interface',
            'entity_id': str(entity.id),
            'name': entity.name
        })
    
    def _extract_enum_text(self, entity: CodeEntity) -> ProcessedText:
        """Extract text from enum entity."""
        parts = []
        
        # Enum name
        parts.append(f"enum {entity.name}")
        
        # Values/variants
        values = entity.metadata.get('values') or entity.metadata.get('variants', [])
        if values:
            parts.append(f"values {' '.join(values)}")
        
        # Language context
        language = entity.metadata.get('language', 'unknown')
        parts.append(f"language {language}")
        
        text = ' '.join(parts)
        
        return ProcessedText.create(text, {
            'entity_type': 'enum',
            'entity_id': str(entity.id),
            'name': entity.name,
            'value_count': len(values)
        })
    
    def _extract_basic_text(self, entity: CodeEntity) -> ProcessedText:
        """Basic text extraction for unsupported entity types."""
        parts = []
        
        # Basic name and type
        parts.append(f"{entity.type.value} {entity.name}")
        
        # Language context
        language = entity.metadata.get('language', 'unknown')
        parts.append(f"language {language}")
        
        text = ' '.join(parts)
        
        return ProcessedText.create(text, {
            'entity_type': entity.type.value,
            'entity_id': str(entity.id),
            'name': entity.name
        })
    
    def enhance_with_context(self, processed_text: ProcessedText, 
                           context_entities: List[CodeEntity]) -> ProcessedText:
        """Enhance processed text with related entity context."""
        if not context_entities:
            return processed_text
        
        # Add context from related entities
        context_parts = []
        
        # Add related entity names as context
        related_names = [entity.name for entity in context_entities[:5]]  # Limit to 5
        if related_names:
            context_parts.append(f"related {' '.join(related_names)}")
        
        # Add context by type
        context_types = {}
        for entity in context_entities:
            entity_type = entity.type.value
            context_types[entity_type] = context_types.get(entity_type, 0) + 1
        
        for entity_type, count in context_types.items():
            if count > 1:
                context_parts.append(f"uses {count} {entity_type}")
        
        if context_parts:
            enhanced_text = f"{processed_text.text} context {' '.join(context_parts)}"
            
            # Update metadata
            enhanced_metadata = dict(processed_text.metadata)
            enhanced_metadata['context_enhanced'] = True
            enhanced_metadata['context_entity_count'] = len(context_entities)
            
            return ProcessedText.create(enhanced_text, enhanced_metadata)
        
        return processed_text