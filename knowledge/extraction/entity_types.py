"""
Data types and enums for entity extraction system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
import hashlib


class EntityType(Enum):
    """Types of code entities that can be extracted."""
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    VARIABLE = "variable"
    CONSTANT = "constant"
    INTERFACE = "interface"
    ENUM = "enum"
    METHOD = "method"


@dataclass
class SourceLocation:
    """Represents a location in source code."""
    line_start: int
    line_end: int
    column_start: Optional[int] = None
    column_end: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'line_start': self.line_start,
            'line_end': self.line_end,
            'column_start': self.column_start,
            'column_end': self.column_end
        }


@dataclass
class CodeEntity:
    """
    Represents a single code entity extracted from AST.
    
    This is the core data structure that will be stored in DuckDB
    and used by relationship detection and embedding generation.
    """
    id: UUID = field(default_factory=uuid4)
    type: EntityType = EntityType.FUNCTION
    name: str = ""
    file_path: str = ""
    location: Optional[SourceLocation] = None
    ast_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate AST hash if not provided."""
        if not self.ast_hash and self.name and self.file_path:
            self.ast_hash = self.generate_hash()
    
    def generate_hash(self) -> str:
        """Generate a hash for this entity based on its key properties."""
        content = f"{self.type.value}:{self.name}:{self.file_path}"
        if self.location:
            content += f":{self.location.line_start}:{self.location.line_end}"
        # Include all metadata in hash for proper change detection
        if self.metadata:
            # Sort metadata keys for consistent hashing
            metadata_str = str(sorted(self.metadata.items()))
            content += f":{metadata_str}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_db_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for DuckDB storage."""
        return {
            'id': str(self.id),
            'type': self.type.value,
            'name': self.name,
            'file_path': self.file_path,
            'line_start': self.location.line_start if self.location else None,
            'line_end': self.location.line_end if self.location else None,
            'ast_hash': self.ast_hash,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> 'CodeEntity':
        """Create CodeEntity from DuckDB row data."""
        location = None
        if data.get('line_start') and data.get('line_end'):
            location = SourceLocation(
                line_start=data['line_start'],
                line_end=data['line_end']
            )
        
        # Handle both UUID objects and strings from database
        entity_id = data['id']
        if isinstance(entity_id, str):
            entity_id = UUID(entity_id)
        elif not isinstance(entity_id, UUID):
            entity_id = UUID(str(entity_id))
        
        return cls(
            id=entity_id,
            type=EntityType(data['type']),
            name=data['name'],
            file_path=data['file_path'],
            location=location,
            ast_hash=data.get('ast_hash', ''),
            metadata=data.get('metadata', {})
        )


@dataclass
class ExtractionResult:
    """Result of entity extraction from a file."""
    file_path: str
    language: str
    entities: List[CodeEntity]
    extraction_time: float
    success: bool = True
    error_message: Optional[str] = None
    
    def get_entity_counts(self) -> Dict[str, int]:
        """Get count of entities by type."""
        counts = {}
        for entity in self.entities:
            entity_type = entity.type.value
            counts[entity_type] = counts.get(entity_type, 0) + 1
        return counts