"""
Pattern Export/Import System - Issue #80

This module provides functionality to export and import patterns between RIF installations,
enabling pattern sharing, backup/restore, and cross-project pattern migration.

Features:
- Export patterns to JSON format with metadata
- Import patterns with version compatibility checking
- Conflict resolution strategies for existing patterns
- Validation of imported patterns
- Success rate and metadata preservation
"""

import json
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from knowledge.database.database_interface import RIFDatabase
from knowledge.pattern_application.core import Pattern, TechStack, load_pattern_from_json


class MergeStrategy(Enum):
    """Strategies for handling conflicts during pattern import."""
    CONSERVATIVE = "conservative"  # Skip conflicting patterns
    OVERWRITE = "overwrite"      # Replace existing patterns
    MERGE = "merge"              # Merge compatible fields
    VERSIONED = "versioned"      # Create new version with suffix


class ConflictResolution(Enum):
    """Results of conflict resolution."""
    SKIPPED = "skipped"
    MERGED = "merged"
    OVERWRITTEN = "overwritten"
    VERSIONED = "versioned"
    ERROR = "error"


@dataclass
class ConflictInfo:
    """Information about a pattern conflict during import."""
    pattern_id: str
    pattern_name: str
    conflict_type: str
    resolution: ConflictResolution
    details: str
    merged_fields: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_id': self.pattern_id,
            'pattern_name': self.pattern_name,
            'conflict_type': self.conflict_type,
            'resolution': self.resolution.value,
            'details': self.details,
            'merged_fields': self.merged_fields
        }


@dataclass
class ImportResult:
    """Result of pattern import operation."""
    imported_count: int
    skipped_count: int
    error_count: int
    imported_patterns: List[str] = field(default_factory=list)
    conflicts: List[ConflictInfo] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    import_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'imported_count': self.imported_count,
            'skipped_count': self.skipped_count,
            'error_count': self.error_count,
            'imported_patterns': self.imported_patterns,
            'conflicts': [c.to_dict() for c in self.conflicts],
            'errors': self.errors,
            'import_duration': self.import_duration
        }


class PatternPortability:
    """
    Main class for pattern export/import functionality.
    
    Handles:
    - Pattern serialization and deserialization
    - Version compatibility checking
    - Conflict resolution strategies
    - Metadata preservation and validation
    """
    
    EXPORT_VERSION = "1.0.0"
    COMPATIBLE_VERSIONS = ["1.0.0"]
    
    def __init__(self, patterns_dir: Optional[str] = None, 
                 project_id: Optional[str] = None,
                 database: Optional[RIFDatabase] = None):
        """
        Initialize the pattern portability system.
        
        Args:
            patterns_dir: Directory containing pattern files (default: knowledge/patterns)
            project_id: Identifier for the current project
            database: Optional RIF database connection
        """
        self.logger = logging.getLogger(__name__)
        self.project_id = project_id or "unknown-project"
        self.database = database
        
        # Set up patterns directory
        if patterns_dir:
            self.patterns_dir = Path(patterns_dir)
        else:
            # Default to knowledge/patterns relative to current directory
            self.patterns_dir = Path("knowledge/patterns")
        
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Pattern portability initialized: {self.patterns_dir}")
    
    def get_all_patterns(self) -> List[Pattern]:
        """Load all patterns from the patterns directory."""
        patterns = []
        
        try:
            # Load from JSON files in patterns directory
            for pattern_file in self.patterns_dir.glob("*.json"):
                try:
                    if pattern_file.name.startswith('.') or 'projects/' in str(pattern_file):
                        continue  # Skip hidden files and project-specific patterns
                    
                    pattern = load_pattern_from_json(str(pattern_file))
                    patterns.append(pattern)
                    self.logger.debug(f"Loaded pattern: {pattern.name}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load pattern from {pattern_file}: {e}")
            
            self.logger.info(f"Loaded {len(patterns)} patterns from {self.patterns_dir}")
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to load patterns: {e}")
            return []
    
    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Get a specific pattern by ID."""
        patterns = self.get_all_patterns()
        for pattern in patterns:
            if pattern.pattern_id == pattern_id:
                return pattern
        return None
    
    def serialize_pattern(self, pattern: Pattern) -> Dict[str, Any]:
        """Serialize a pattern to dictionary format for export."""
        try:
            pattern_data = pattern.to_dict()
            
            # Add export-specific metadata
            pattern_data['export_metadata'] = {
                'exported_at': datetime.now(timezone.utc).isoformat(),
                'export_version': self.EXPORT_VERSION,
                'source_project': self.project_id
            }
            
            return pattern_data
            
        except Exception as e:
            self.logger.error(f"Failed to serialize pattern {pattern.pattern_id}: {e}")
            raise
    
    def deserialize_pattern(self, pattern_data: Dict[str, Any]) -> Pattern:
        """Deserialize a pattern from dictionary format."""
        try:
            # Extract tech stack if present
            tech_stack = None
            if 'tech_stack' in pattern_data and pattern_data['tech_stack']:
                ts_data = pattern_data['tech_stack']
                tech_stack = TechStack(
                    primary_language=ts_data.get('primary_language', ''),
                    frameworks=ts_data.get('frameworks', []),
                    databases=ts_data.get('databases', []),
                    tools=ts_data.get('tools', []),
                    architecture_pattern=ts_data.get('architecture_pattern'),
                    deployment_target=ts_data.get('deployment_target')
                )
            
            # Create pattern object
            pattern = Pattern(
                pattern_id=pattern_data.get('pattern_id', ''),
                name=pattern_data.get('name', pattern_data.get('pattern_name', 'Unknown Pattern')),
                description=pattern_data.get('description', ''),
                complexity=pattern_data.get('complexity', 'medium'),
                tech_stack=tech_stack,
                domain=pattern_data.get('domain', 'general'),
                tags=pattern_data.get('tags', []),
                confidence=pattern_data.get('confidence', 0.0),
                success_rate=pattern_data.get('success_rate', 0.0),
                usage_count=pattern_data.get('usage_count', 0),
                implementation_steps=pattern_data.get('implementation_steps', []),
                code_examples=pattern_data.get('code_examples', []),
                validation_criteria=pattern_data.get('validation_criteria', [])
            )
            
            return pattern
            
        except Exception as e:
            self.logger.error(f"Failed to deserialize pattern: {e}")
            raise
    
    def calculate_avg_success_rate(self, patterns: List[Pattern]) -> float:
        """Calculate average success rate for a list of patterns."""
        if not patterns:
            return 0.0
        
        success_rates = [p.success_rate for p in patterns if p.success_rate > 0]
        if not success_rates:
            return 0.0
            
        return sum(success_rates) / len(success_rates)
    
    def export_patterns(self, pattern_ids: Optional[List[str]] = None, 
                       output_file: Optional[str] = None) -> str:
        """
        Export patterns to JSON format.
        
        Args:
            pattern_ids: List of pattern IDs to export (None = export all)
            output_file: Output file path (optional)
            
        Returns:
            JSON string containing exported patterns
        """
        try:
            start_time = datetime.now()
            
            # Get patterns to export
            if pattern_ids:
                patterns = []
                for pattern_id in pattern_ids:
                    pattern = self.get_pattern(pattern_id)
                    if pattern:
                        patterns.append(pattern)
                    else:
                        self.logger.warning(f"Pattern not found: {pattern_id}")
            else:
                patterns = self.get_all_patterns()
            
            if not patterns:
                self.logger.warning("No patterns found to export")
                patterns = []
            
            # Create export data
            export_data = {
                'version': self.EXPORT_VERSION,
                'exported_at': datetime.now(timezone.utc).isoformat(),
                'patterns': [self.serialize_pattern(p) for p in patterns],
                'metadata': {
                    'source_project': self.project_id,
                    'pattern_count': len(patterns),
                    'success_rate_avg': self.calculate_avg_success_rate(patterns),
                    'export_duration': (datetime.now() - start_time).total_seconds(),
                    'complexity_breakdown': self._get_complexity_breakdown(patterns),
                    'domain_breakdown': self._get_domain_breakdown(patterns)
                }
            }
            
            # Convert to JSON
            json_data = json.dumps(export_data, indent=2)
            
            # Save to file if requested
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(json_data)
                self.logger.info(f"Exported {len(patterns)} patterns to {output_file}")
            
            self.logger.info(f"Successfully exported {len(patterns)} patterns")
            return json_data
            
        except Exception as e:
            self.logger.error(f"Pattern export failed: {e}")
            raise
    
    def _get_complexity_breakdown(self, patterns: List[Pattern]) -> Dict[str, int]:
        """Get breakdown of patterns by complexity."""
        breakdown = {}
        for pattern in patterns:
            complexity = pattern.complexity
            breakdown[complexity] = breakdown.get(complexity, 0) + 1
        return breakdown
    
    def _get_domain_breakdown(self, patterns: List[Pattern]) -> Dict[str, int]:
        """Get breakdown of patterns by domain."""
        breakdown = {}
        for pattern in patterns:
            domain = pattern.domain
            breakdown[domain] = breakdown.get(domain, 0) + 1
        return breakdown
    
    def validate_version(self, version: str) -> bool:
        """Validate that the import version is compatible."""
        return version in self.COMPATIBLE_VERSIONS
    
    def pattern_exists(self, pattern_id: str) -> bool:
        """Check if a pattern with the given ID already exists."""
        return self.get_pattern(pattern_id) is not None
    
    def resolve_conflict(self, pattern_data: Dict[str, Any], 
                        merge_strategy: MergeStrategy) -> ConflictInfo:
        """
        Resolve a conflict when importing a pattern that already exists.
        
        Args:
            pattern_data: Data for the pattern being imported
            merge_strategy: Strategy to use for conflict resolution
            
        Returns:
            ConflictInfo describing the resolution
        """
        pattern_id = pattern_data.get('pattern_id', 'unknown')
        pattern_name = pattern_data.get('name', pattern_data.get('pattern_name', 'Unknown'))
        
        try:
            existing_pattern = self.get_pattern(pattern_id)
            if not existing_pattern:
                return ConflictInfo(
                    pattern_id=pattern_id,
                    pattern_name=pattern_name,
                    conflict_type="no_conflict",
                    resolution=ConflictResolution.SKIPPED,
                    details="Pattern does not exist, no conflict"
                )
            
            if merge_strategy == MergeStrategy.CONSERVATIVE:
                return ConflictInfo(
                    pattern_id=pattern_id,
                    pattern_name=pattern_name,
                    conflict_type="existing_pattern",
                    resolution=ConflictResolution.SKIPPED,
                    details="Skipped due to conservative merge strategy"
                )
            
            elif merge_strategy == MergeStrategy.OVERWRITE:
                self._save_pattern(pattern_data, overwrite=True)
                return ConflictInfo(
                    pattern_id=pattern_id,
                    pattern_name=pattern_name,
                    conflict_type="existing_pattern",
                    resolution=ConflictResolution.OVERWRITTEN,
                    details="Existing pattern overwritten"
                )
            
            elif merge_strategy == MergeStrategy.MERGE:
                merged_pattern, merged_fields = self._merge_patterns(existing_pattern, pattern_data)
                self._save_pattern(merged_pattern, overwrite=True)
                return ConflictInfo(
                    pattern_id=pattern_id,
                    pattern_name=pattern_name,
                    conflict_type="existing_pattern",
                    resolution=ConflictResolution.MERGED,
                    details=f"Merged with existing pattern: {', '.join(merged_fields)}",
                    merged_fields=merged_fields
                )
            
            elif merge_strategy == MergeStrategy.VERSIONED:
                versioned_id = self._create_versioned_id(pattern_id)
                pattern_data['pattern_id'] = versioned_id
                self._save_pattern(pattern_data)
                return ConflictInfo(
                    pattern_id=versioned_id,
                    pattern_name=pattern_name,
                    conflict_type="existing_pattern",
                    resolution=ConflictResolution.VERSIONED,
                    details=f"Created versioned pattern: {versioned_id}"
                )
            
        except Exception as e:
            self.logger.error(f"Failed to resolve conflict for pattern {pattern_id}: {e}")
            return ConflictInfo(
                pattern_id=pattern_id,
                pattern_name=pattern_name,
                conflict_type="resolution_error",
                resolution=ConflictResolution.ERROR,
                details=f"Error during conflict resolution: {e}"
            )
    
    def _merge_patterns(self, existing: Pattern, new_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Merge existing pattern with new pattern data.
        
        Returns:
            Tuple of (merged_pattern_data, list_of_merged_fields)
        """
        merged_fields = []
        existing_data = existing.to_dict()
        
        # Fields that can be merged
        mergeable_fields = {
            'tags': 'union',
            'implementation_steps': 'append',
            'code_examples': 'append',
            'validation_criteria': 'union',
            'usage_count': 'sum'
        }
        
        # Fields that should use the higher value
        max_fields = ['confidence', 'success_rate']
        
        # Start with existing data
        merged_data = existing_data.copy()
        
        # Merge specific fields
        for field, merge_type in mergeable_fields.items():
            if field in new_data and new_data[field]:
                if field not in existing_data or not existing_data[field]:
                    merged_data[field] = new_data[field]
                    merged_fields.append(field)
                else:
                    if merge_type == 'union' and isinstance(new_data[field], list):
                        # Union of lists
                        merged_data[field] = list(set(existing_data[field] + new_data[field]))
                        merged_fields.append(field)
                    elif merge_type == 'append' and isinstance(new_data[field], list):
                        # Append new items
                        merged_data[field] = existing_data[field] + new_data[field]
                        merged_fields.append(field)
                    elif merge_type == 'sum' and isinstance(new_data[field], (int, float)):
                        # Sum numeric values
                        merged_data[field] = existing_data[field] + new_data[field]
                        merged_fields.append(field)
        
        # Use higher values for certain fields
        for field in max_fields:
            if field in new_data and isinstance(new_data[field], (int, float)):
                if field not in existing_data or new_data[field] > existing_data[field]:
                    merged_data[field] = new_data[field]
                    merged_fields.append(field)
        
        # Update timestamps
        merged_data['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        return merged_data, merged_fields
    
    def _create_versioned_id(self, original_id: str) -> str:
        """Create a versioned ID for a pattern."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{original_id}_v{timestamp}"
    
    def _save_pattern(self, pattern_data: Dict[str, Any], overwrite: bool = False) -> bool:
        """Save pattern data to a JSON file."""
        try:
            pattern_id = pattern_data.get('pattern_id', 'unknown')
            filename = f"{pattern_id}.json"
            filepath = self.patterns_dir / filename
            
            if filepath.exists() and not overwrite:
                self.logger.warning(f"Pattern file already exists: {filepath}")
                return False
            
            with open(filepath, 'w') as f:
                json.dump(pattern_data, f, indent=2)
            
            self.logger.debug(f"Saved pattern: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save pattern: {e}")
            return False
    
    def import_patterns(self, import_data: Union[str, Dict[str, Any]], 
                       merge_strategy: MergeStrategy = MergeStrategy.CONSERVATIVE) -> ImportResult:
        """
        Import patterns from JSON data.
        
        Args:
            import_data: JSON string or dictionary containing pattern data
            merge_strategy: Strategy for handling conflicts
            
        Returns:
            ImportResult with detailed information about the import
        """
        start_time = datetime.now()
        result = ImportResult(
            imported_count=0,
            skipped_count=0,
            error_count=0
        )
        
        try:
            # Parse JSON data if needed
            if isinstance(import_data, str):
                data = json.loads(import_data)
            else:
                data = import_data
            
            # Validate version compatibility
            version = data.get('version', 'unknown')
            if not self.validate_version(version):
                result.errors.append(f"Incompatible version: {version}. Compatible versions: {self.COMPATIBLE_VERSIONS}")
                result.error_count = 1
                return result
            
            # Process each pattern
            patterns = data.get('patterns', [])
            for pattern_data in patterns:
                try:
                    pattern_id = pattern_data.get('pattern_id', 'unknown')
                    
                    if self.pattern_exists(pattern_id):
                        # Handle conflict
                        conflict_info = self.resolve_conflict(pattern_data, merge_strategy)
                        result.conflicts.append(conflict_info)
                        
                        if conflict_info.resolution == ConflictResolution.SKIPPED:
                            result.skipped_count += 1
                        elif conflict_info.resolution == ConflictResolution.ERROR:
                            result.error_count += 1
                            result.errors.append(f"Failed to resolve conflict for pattern {pattern_id}")
                        else:
                            result.imported_count += 1
                            result.imported_patterns.append(
                                conflict_info.pattern_id  # May be different if versioned
                            )
                    else:
                        # Import new pattern
                        if self._save_pattern(pattern_data):
                            result.imported_count += 1
                            result.imported_patterns.append(pattern_id)
                        else:
                            result.error_count += 1
                            result.errors.append(f"Failed to save pattern {pattern_id}")
                
                except Exception as e:
                    result.error_count += 1
                    result.errors.append(f"Error processing pattern: {e}")
                    self.logger.error(f"Error processing pattern: {e}")
            
            # Calculate duration
            result.import_duration = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(
                f"Import completed: {result.imported_count} imported, "
                f"{result.skipped_count} skipped, {result.error_count} errors"
            )
            
            return result
            
        except Exception as e:
            result.error_count += 1
            result.errors.append(f"Import failed: {e}")
            result.import_duration = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Pattern import failed: {e}")
            return result
    
    def validate_patterns(self, pattern_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate a list of pattern data before import.
        
        Returns:
            List of validation results, one per pattern
        """
        results = []
        
        for i, pattern_data in enumerate(pattern_data_list):
            validation = {
                'index': i,
                'pattern_id': pattern_data.get('pattern_id', 'unknown'),
                'valid': True,
                'errors': [],
                'warnings': []
            }
            
            # Required fields
            required_fields = ['pattern_id', 'name', 'description']
            for field in required_fields:
                if field not in pattern_data or not pattern_data[field]:
                    validation['valid'] = False
                    validation['errors'].append(f"Missing required field: {field}")
            
            # Validate data types
            if 'confidence' in pattern_data:
                try:
                    confidence = float(pattern_data['confidence'])
                    if not 0.0 <= confidence <= 1.0:
                        validation['warnings'].append("Confidence should be between 0.0 and 1.0")
                except (ValueError, TypeError):
                    validation['errors'].append("Confidence must be a number")
                    validation['valid'] = False
            
            if 'success_rate' in pattern_data:
                try:
                    success_rate = float(pattern_data['success_rate'])
                    if not 0.0 <= success_rate <= 1.0:
                        validation['warnings'].append("Success rate should be between 0.0 and 1.0")
                except (ValueError, TypeError):
                    validation['errors'].append("Success rate must be a number")
                    validation['valid'] = False
            
            # Validate complexity
            if 'complexity' in pattern_data:
                valid_complexities = ['low', 'medium', 'high', 'very-high']
                if pattern_data['complexity'] not in valid_complexities:
                    validation['warnings'].append(f"Complexity should be one of: {valid_complexities}")
            
            results.append(validation)
        
        return results
    
    def get_export_stats(self) -> Dict[str, Any]:
        """Get statistics about patterns available for export."""
        patterns = self.get_all_patterns()
        
        return {
            'total_patterns': len(patterns),
            'complexity_breakdown': self._get_complexity_breakdown(patterns),
            'domain_breakdown': self._get_domain_breakdown(patterns),
            'avg_success_rate': self.calculate_avg_success_rate(patterns),
            'patterns_with_embeddings': sum(1 for p in patterns if hasattr(p, 'embedding') and p.embedding),
            'patterns_with_examples': sum(1 for p in patterns if p.code_examples),
            'most_successful_domain': max(self._get_domain_breakdown(patterns).items(), 
                                        key=lambda x: x[1], default=('none', 0))[0]
        }