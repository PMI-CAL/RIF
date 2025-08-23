"""
Data types and enums for code relationship detection system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from uuid import UUID, uuid4
import hashlib
import json


class RelationshipType(Enum):
    """Types of code relationships that can be detected."""
    IMPORTS = "imports"        # Module A imports module B
    CALLS = "calls"           # Function A calls function B
    EXTENDS = "extends"       # Class A extends class B
    USES = "uses"             # Entity A uses entity B (general dependency)
    IMPLEMENTS = "implements" # Class A implements interface B
    REFERENCES = "references" # Entity A references entity B (variable, type, etc.)
    CONTAINS = "contains"     # Entity A contains entity B (module contains class, etc.)


@dataclass
class RelationshipContext:
    """Additional context information for a relationship."""
    line_number: Optional[int] = None
    column_start: Optional[int] = None  
    column_end: Optional[int] = None
    source_code: Optional[str] = None  # The actual code that establishes the relationship
    call_arguments: Optional[List[str]] = None  # For function calls
    import_alias: Optional[str] = None  # For import statements
    is_conditional: bool = False  # If relationship is conditional (inside if/try/etc.)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'line_number': self.line_number,
            'column_start': self.column_start,
            'column_end': self.column_end,
            'source_code': self.source_code,
            'call_arguments': self.call_arguments,
            'import_alias': self.import_alias,
            'is_conditional': self.is_conditional
        }


@dataclass
class CodeRelationship:
    """
    Represents a relationship between two code entities.
    
    This matches the relationships table schema in DuckDB and provides
    the core data structure for relationship detection and storage.
    """
    id: UUID = field(default_factory=uuid4)
    source_id: UUID = field(default_factory=uuid4)  # Source entity UUID
    target_id: UUID = field(default_factory=uuid4)  # Target entity UUID  
    relationship_type: RelationshipType = RelationshipType.USES
    confidence: float = 1.0  # Confidence score (0.0 to 1.0)
    context: Optional[RelationshipContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate relationship data after initialization."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        if self.source_id == self.target_id:
            raise ValueError("Source and target cannot be the same entity")
    
    def generate_hash(self) -> str:
        """Generate a unique hash for this relationship."""
        content = (
            f"{self.source_id}:{self.target_id}:{self.relationship_type.value}"
        )
        if self.context and self.context.line_number:
            content += f":{self.context.line_number}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_db_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for DuckDB storage."""
        metadata = dict(self.metadata)  # Copy existing metadata
        
        # Add context information to metadata
        if self.context:
            metadata['context'] = self.context.to_dict()
        
        return {
            'id': str(self.id),
            'source_id': str(self.source_id),
            'target_id': str(self.target_id),
            'relationship_type': self.relationship_type.value,
            'confidence': self.confidence,
            'metadata': metadata
        }
    
    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> 'CodeRelationship':
        """Create CodeRelationship from DuckDB row data."""
        
        # Handle UUID conversion
        def to_uuid(value: Union[str, UUID]) -> UUID:
            if isinstance(value, UUID):
                return value
            return UUID(str(value))
        
        # Extract context from metadata if present
        metadata = data.get('metadata', {})
        context = None
        if 'context' in metadata:
            context_data = metadata.pop('context')
            context = RelationshipContext(
                line_number=context_data.get('line_number'),
                column_start=context_data.get('column_start'),
                column_end=context_data.get('column_end'),
                source_code=context_data.get('source_code'),
                call_arguments=context_data.get('call_arguments'),
                import_alias=context_data.get('import_alias'),
                is_conditional=context_data.get('is_conditional', False)
            )
        
        return cls(
            id=to_uuid(data['id']),
            source_id=to_uuid(data['source_id']),
            target_id=to_uuid(data['target_id']),
            relationship_type=RelationshipType(data['relationship_type']),
            confidence=float(data.get('confidence', 1.0)),
            context=context,
            metadata=metadata
        )
    
    def is_cross_file_relationship(self, source_file: str, target_file: str) -> bool:
        """Check if this relationship crosses file boundaries."""
        return source_file != target_file
    
    def get_relationship_strength(self) -> float:
        """Calculate relationship strength based on confidence and context."""
        strength = self.confidence
        
        # Boost confidence for explicit relationships
        if self.relationship_type in [RelationshipType.IMPORTS, RelationshipType.EXTENDS]:
            strength *= 1.1
        
        # Reduce confidence for conditional relationships
        if self.context and self.context.is_conditional:
            strength *= 0.8
        
        return min(strength, 1.0)


@dataclass
class RelationshipDetectionResult:
    """Result of relationship detection from a file or AST node."""
    file_path: str
    language: str
    relationships: List[CodeRelationship]
    detection_time: float
    success: bool = True
    error_message: Optional[str] = None
    
    def get_relationship_counts(self) -> Dict[str, int]:
        """Get count of relationships by type."""
        counts = {}
        for relationship in self.relationships:
            rel_type = relationship.relationship_type.value
            counts[rel_type] = counts.get(rel_type, 0) + 1
        return counts
    
    def get_cross_file_relationships(self, current_file: str) -> List[CodeRelationship]:
        """Get relationships that reference entities in other files."""
        # Note: This would need entity file information to work properly
        # For now, return all relationships as potentially cross-file
        return self.relationships
    
    def get_high_confidence_relationships(self, threshold: float = 0.8) -> List[CodeRelationship]:
        """Get relationships above a confidence threshold."""
        return [r for r in self.relationships if r.confidence >= threshold]


@dataclass 
class RelationshipPattern:
    """Represents a detected pattern in code relationships."""
    pattern_type: str
    frequency: int
    confidence: float
    example_relationships: List[CodeRelationship] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_type': self.pattern_type,
            'frequency': self.frequency,
            'confidence': self.confidence,
            'description': self.description,
            'example_count': len(self.example_relationships)
        }


class RelationshipQuery:
    """Builder class for constructing relationship queries."""
    
    def __init__(self):
        self.source_entity_id: Optional[UUID] = None
        self.target_entity_id: Optional[UUID] = None
        self.relationship_types: List[RelationshipType] = []
        self.min_confidence: float = 0.0
        self.file_path_filter: Optional[str] = None
        self.cross_file_only: bool = False
    
    def from_source(self, entity_id: UUID) -> 'RelationshipQuery':
        """Filter relationships originating from a specific entity."""
        self.source_entity_id = entity_id
        return self
    
    def to_target(self, entity_id: UUID) -> 'RelationshipQuery':
        """Filter relationships targeting a specific entity.""" 
        self.target_entity_id = entity_id
        return self
    
    def of_type(self, *types: RelationshipType) -> 'RelationshipQuery':
        """Filter by relationship types."""
        self.relationship_types.extend(types)
        return self
    
    def with_min_confidence(self, confidence: float) -> 'RelationshipQuery':
        """Filter by minimum confidence score."""
        self.min_confidence = confidence
        return self
    
    def in_file(self, file_path: str) -> 'RelationshipQuery':
        """Filter relationships involving entities in a specific file."""
        self.file_path_filter = file_path
        return self
    
    def cross_file_only(self) -> 'RelationshipQuery':
        """Filter to only cross-file relationships."""
        self.cross_file_only = True
        return self
    
    def to_sql_conditions(self) -> tuple[str, List[Any]]:
        """Convert query to SQL WHERE conditions and parameters."""
        conditions = []
        params = []
        
        if self.source_entity_id:
            conditions.append("source_id = ?")
            params.append(str(self.source_entity_id))
        
        if self.target_entity_id:
            conditions.append("target_id = ?") 
            params.append(str(self.target_entity_id))
        
        if self.relationship_types:
            placeholders = ','.join(['?' for _ in self.relationship_types])
            conditions.append(f"relationship_type IN ({placeholders})")
            params.extend([rt.value for rt in self.relationship_types])
        
        if self.min_confidence > 0.0:
            conditions.append("confidence >= ?")
            params.append(self.min_confidence)
        
        where_clause = ' AND '.join(conditions) if conditions else '1=1'
        return where_clause, params