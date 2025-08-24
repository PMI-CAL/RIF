"""
Graph Validation System for RIF Knowledge Graph
Issue #68: Implement graph validation

Provides comprehensive validation for the knowledge graph including:
- Referential integrity checks
- Constraint validation
- Inconsistency detection and reporting
- Fix suggestion generation

Author: RIF-Implementer
Date: 2025-08-23
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import uuid

from .connection_manager import DuckDBConnectionManager
from .database_config import DatabaseConfig


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"  # Data corruption, referential integrity violations
    ERROR = "error"       # Constraint violations, invalid data
    WARNING = "warning"   # Performance issues, best practice violations
    INFO = "info"         # Recommendations, optimizations


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    id: str
    severity: ValidationSeverity
    category: str
    issue_type: str
    message: str
    table_name: Optional[str] = None
    entity_id: Optional[str] = None
    relationship_id: Optional[str] = None
    suggested_fix: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    report_id: str
    timestamp: datetime
    total_issues: int
    issues_by_severity: Dict[str, int]
    issues: List[ValidationIssue]
    database_stats: Dict[str, Any]
    validation_duration: float
    recommendations: List[str]
    
    def __post_init__(self):
        if self.report_id is None:
            self.report_id = str(uuid.uuid4())
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if report contains critical issues."""
        return self.issues_by_severity.get('critical', 0) > 0
    
    @property
    def has_errors(self) -> bool:
        """Check if report contains errors."""
        return self.issues_by_severity.get('error', 0) > 0
    
    def get_issues_by_category(self, category: str) -> List[ValidationIssue]:
        """Get issues filtered by category."""
        return [issue for issue in self.issues if issue.category == category]
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity."""
        return [issue for issue in self.issues if issue.severity == severity]


class GraphValidator:
    """
    Comprehensive graph validation system for RIF knowledge graph.
    
    Validates:
    - Referential integrity between entities and relationships
    - Database constraints and data quality
    - Cross-table consistency
    - Performance and optimization opportunities
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.logger = logging.getLogger(__name__)
        self.connection_manager = DuckDBConnectionManager(self.config)
        
        # Validation categories
        self.validation_categories = [
            'referential_integrity',
            'constraint_validation', 
            'data_consistency',
            'performance_optimization',
            'data_quality'
        ]
        
        self.logger.info("GraphValidator initialized")
    
    def validate_graph(self, 
                      categories: Optional[List[str]] = None,
                      include_stats: bool = True,
                      include_recommendations: bool = True) -> ValidationReport:
        """
        Perform comprehensive graph validation.
        
        Args:
            categories: List of validation categories to run (default: all)
            include_stats: Include database statistics in report
            include_recommendations: Include optimization recommendations
            
        Returns:
            ValidationReport with all findings
        """
        start_time = datetime.now()
        issues = []
        
        # Default to all categories if none specified
        if categories is None:
            categories = self.validation_categories
        
        self.logger.info(f"Starting graph validation for categories: {categories}")
        
        try:
            # Run validation by category
            if 'referential_integrity' in categories:
                issues.extend(self._validate_referential_integrity())
            
            if 'constraint_validation' in categories:
                issues.extend(self._validate_constraints())
            
            if 'data_consistency' in categories:
                issues.extend(self._validate_data_consistency())
            
            if 'performance_optimization' in categories:
                issues.extend(self._validate_performance())
            
            if 'data_quality' in categories:
                issues.extend(self._validate_data_quality())
            
            # Generate statistics
            database_stats = {}
            if include_stats:
                database_stats = self._generate_database_stats()
            
            # Generate recommendations
            recommendations = []
            if include_recommendations:
                recommendations = self._generate_recommendations(issues, database_stats)
            
            # Calculate duration
            validation_duration = (datetime.now() - start_time).total_seconds()
            
            # Count issues by severity
            issues_by_severity = {}
            for severity in ValidationSeverity:
                count = len([issue for issue in issues if issue.severity == severity])
                issues_by_severity[severity.value] = count
            
            # Create report
            report = ValidationReport(
                report_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                total_issues=len(issues),
                issues_by_severity=issues_by_severity,
                issues=issues,
                database_stats=database_stats,
                validation_duration=validation_duration,
                recommendations=recommendations
            )
            
            self.logger.info(f"Graph validation completed: {len(issues)} issues found in {validation_duration:.2f}s")
            return report
            
        except Exception as e:
            self.logger.error(f"Graph validation failed: {e}")
            raise
    
    def _validate_referential_integrity(self) -> List[ValidationIssue]:
        """Validate referential integrity between entities and relationships."""
        issues = []
        
        try:
            with self.connection_manager.get_connection() as conn:
                # Check for relationships with missing source entities
                orphaned_sources = conn.execute("""
                    SELECT r.id, r.source_id, r.relationship_type, r.target_id
                    FROM relationships r
                    LEFT JOIN entities e ON r.source_id = e.id
                    WHERE e.id IS NULL
                """).fetchall()
                
                for row in orphaned_sources:
                    issues.append(ValidationIssue(
                        id=str(uuid.uuid4()),
                        severity=ValidationSeverity.CRITICAL,
                        category='referential_integrity',
                        issue_type='missing_source_entity',
                        message=f"Relationship {row[0]} references non-existent source entity {row[1]}",
                        relationship_id=str(row[0]),
                        suggested_fix=f"Remove relationship or restore entity {row[1]}",
                        metadata={
                            'relationship_type': row[2],
                            'target_id': str(row[3])
                        }
                    ))
                
                # Check for relationships with missing target entities
                orphaned_targets = conn.execute("""
                    SELECT r.id, r.target_id, r.relationship_type, r.source_id
                    FROM relationships r
                    LEFT JOIN entities e ON r.target_id = e.id
                    WHERE e.id IS NULL
                """).fetchall()
                
                for row in orphaned_targets:
                    issues.append(ValidationIssue(
                        id=str(uuid.uuid4()),
                        severity=ValidationSeverity.CRITICAL,
                        category='referential_integrity',
                        issue_type='missing_target_entity',
                        message=f"Relationship {row[0]} references non-existent target entity {row[1]}",
                        relationship_id=str(row[0]),
                        suggested_fix=f"Remove relationship or restore entity {row[1]}",
                        metadata={
                            'relationship_type': row[2],
                            'source_id': str(row[3])
                        }
                    ))
                
                # Check for orphaned entities (no relationships)
                orphaned_entities = conn.execute("""
                    SELECT e.id, e.name, e.type, e.file_path
                    FROM entities e
                    LEFT JOIN relationships r1 ON e.id = r1.source_id
                    LEFT JOIN relationships r2 ON e.id = r2.target_id
                    WHERE r1.id IS NULL AND r2.id IS NULL
                """).fetchall()
                
                for row in orphaned_entities:
                    issues.append(ValidationIssue(
                        id=str(uuid.uuid4()),
                        severity=ValidationSeverity.WARNING,
                        category='referential_integrity',
                        issue_type='orphaned_entity',
                        message=f"Entity '{row[1]}' ({row[2]}) has no relationships",
                        entity_id=str(row[0]),
                        suggested_fix="Review if entity should have relationships or be removed",
                        metadata={
                            'entity_name': row[1],
                            'entity_type': row[2],
                            'file_path': row[3]
                        }
                    ))
        
        except Exception as e:
            self.logger.error(f"Referential integrity validation failed: {e}")
            issues.append(ValidationIssue(
                id=str(uuid.uuid4()),
                severity=ValidationSeverity.ERROR,
                category='referential_integrity',
                issue_type='validation_error',
                message=f"Failed to validate referential integrity: {e}",
                suggested_fix="Check database connection and table integrity"
            ))
        
        return issues
    
    def _validate_constraints(self) -> List[ValidationIssue]:
        """Validate database constraints and data types."""
        issues = []
        
        try:
            with self.connection_manager.get_connection() as conn:
                # Check entity type constraints
                invalid_entity_types = conn.execute("""
                    SELECT id, name, type, file_path
                    FROM entities
                    WHERE type NOT IN ('function', 'class', 'module', 'variable', 'constant', 'interface', 'enum')
                """).fetchall()
                
                for row in invalid_entity_types:
                    issues.append(ValidationIssue(
                        id=str(uuid.uuid4()),
                        severity=ValidationSeverity.ERROR,
                        category='constraint_validation',
                        issue_type='invalid_entity_type',
                        message=f"Entity '{row[1]}' has invalid type '{row[2]}'",
                        entity_id=str(row[0]),
                        suggested_fix="Update entity type to valid value",
                        metadata={
                            'entity_name': row[1],
                            'invalid_type': row[2],
                            'file_path': row[3]
                        }
                    ))
                
                # Check relationship type constraints
                invalid_relationship_types = conn.execute("""
                    SELECT id, source_id, target_id, relationship_type
                    FROM relationships
                    WHERE relationship_type NOT IN ('imports', 'calls', 'extends', 'uses', 'implements', 'references', 'contains')
                """).fetchall()
                
                for row in invalid_relationship_types:
                    issues.append(ValidationIssue(
                        id=str(uuid.uuid4()),
                        severity=ValidationSeverity.ERROR,
                        category='constraint_validation',
                        issue_type='invalid_relationship_type',
                        message=f"Relationship {row[0]} has invalid type '{row[3]}'",
                        relationship_id=str(row[0]),
                        suggested_fix="Update relationship type to valid value",
                        metadata={
                            'invalid_type': row[3],
                            'source_id': str(row[1]),
                            'target_id': str(row[2])
                        }
                    ))
                
                # Check confidence score constraints
                invalid_confidence = conn.execute("""
                    SELECT id, source_id, target_id, confidence
                    FROM relationships
                    WHERE confidence < 0.0 OR confidence > 1.0
                """).fetchall()
                
                for row in invalid_confidence:
                    issues.append(ValidationIssue(
                        id=str(uuid.uuid4()),
                        severity=ValidationSeverity.ERROR,
                        category='constraint_validation',
                        issue_type='invalid_confidence_score',
                        message=f"Relationship {row[0]} has invalid confidence score {row[3]}",
                        relationship_id=str(row[0]),
                        suggested_fix="Set confidence score between 0.0 and 1.0",
                        metadata={
                            'invalid_confidence': row[3],
                            'source_id': str(row[1]),
                            'target_id': str(row[2])
                        }
                    ))
                
                # Check for self-referencing relationships (should be prevented by constraint)
                self_references = conn.execute("""
                    SELECT id, source_id, target_id, relationship_type
                    FROM relationships
                    WHERE source_id = target_id
                """).fetchall()
                
                for row in self_references:
                    issues.append(ValidationIssue(
                        id=str(uuid.uuid4()),
                        severity=ValidationSeverity.ERROR,
                        category='constraint_validation',
                        issue_type='self_reference',
                        message=f"Relationship {row[0]} references itself (entity {row[1]})",
                        relationship_id=str(row[0]),
                        suggested_fix="Remove self-referencing relationship",
                        metadata={
                            'entity_id': str(row[1]),
                            'relationship_type': row[3]
                        }
                    ))
                
        except Exception as e:
            self.logger.error(f"Constraint validation failed: {e}")
            issues.append(ValidationIssue(
                id=str(uuid.uuid4()),
                severity=ValidationSeverity.ERROR,
                category='constraint_validation',
                issue_type='validation_error',
                message=f"Failed to validate constraints: {e}",
                suggested_fix="Check database schema and constraints"
            ))
        
        return issues
    
    def _validate_data_consistency(self) -> List[ValidationIssue]:
        """Validate cross-table data consistency."""
        issues = []
        
        try:
            with self.connection_manager.get_connection() as conn:
                # Check for duplicate entities (same name, type, file_path)
                duplicate_entities = conn.execute("""
                    SELECT name, type, file_path, COUNT(*) as count
                    FROM entities
                    GROUP BY name, type, file_path
                    HAVING COUNT(*) > 1
                """).fetchall()
                
                for row in duplicate_entities:
                    issues.append(ValidationIssue(
                        id=str(uuid.uuid4()),
                        severity=ValidationSeverity.WARNING,
                        category='data_consistency',
                        issue_type='duplicate_entities',
                        message=f"Found {row[3]} duplicate entities: '{row[0]}' ({row[1]}) in {row[2]}",
                        suggested_fix="Merge or remove duplicate entities",
                        metadata={
                            'entity_name': row[0],
                            'entity_type': row[1],
                            'file_path': row[2],
                            'count': row[3]
                        }
                    ))
                
                # Check for duplicate relationships
                duplicate_relationships = conn.execute("""
                    SELECT source_id, target_id, relationship_type, COUNT(*) as count
                    FROM relationships
                    GROUP BY source_id, target_id, relationship_type
                    HAVING COUNT(*) > 1
                """).fetchall()
                
                for row in duplicate_relationships:
                    issues.append(ValidationIssue(
                        id=str(uuid.uuid4()),
                        severity=ValidationSeverity.WARNING,
                        category='data_consistency',
                        issue_type='duplicate_relationships',
                        message=f"Found {row[3]} duplicate relationships: {row[0]} -> {row[1]} ({row[2]})",
                        suggested_fix="Remove or consolidate duplicate relationships",
                        metadata={
                            'source_id': str(row[0]),
                            'target_id': str(row[1]),
                            'relationship_type': row[2],
                            'count': row[3]
                        }
                    ))
        
        except Exception as e:
            self.logger.error(f"Data consistency validation failed: {e}")
            issues.append(ValidationIssue(
                id=str(uuid.uuid4()),
                severity=ValidationSeverity.ERROR,
                category='data_consistency',
                issue_type='validation_error',
                message=f"Failed to validate data consistency: {e}",
                suggested_fix="Check database connection and data integrity"
            ))
        
        return issues
    
    def _validate_performance(self) -> List[ValidationIssue]:
        """Validate performance-related aspects."""
        issues = []
        
        try:
            with self.connection_manager.get_connection() as conn:
                # Check for entities without embeddings (impacts vector search performance)
                entities_without_embeddings = conn.execute("""
                    SELECT COUNT(*) as count
                    FROM entities
                    WHERE embedding IS NULL
                """).fetchone()
                
                if entities_without_embeddings[0] > 0:
                    issues.append(ValidationIssue(
                        id=str(uuid.uuid4()),
                        severity=ValidationSeverity.INFO,
                        category='performance_optimization',
                        issue_type='missing_embeddings',
                        message=f"{entities_without_embeddings[0]} entities lack embeddings for vector search",
                        suggested_fix="Generate embeddings for entities to enable vector similarity search",
                        metadata={'missing_embeddings_count': entities_without_embeddings[0]}
                    ))
                
                # Check for agent memories without embeddings
                memories_without_embeddings = conn.execute("""
                    SELECT COUNT(*) as count
                    FROM agent_memory
                    WHERE embedding IS NULL
                """).fetchone()
                
                if memories_without_embeddings[0] > 0:
                    issues.append(ValidationIssue(
                        id=str(uuid.uuid4()),
                        severity=ValidationSeverity.INFO,
                        category='performance_optimization',
                        issue_type='missing_memory_embeddings',
                        message=f"{memories_without_embeddings[0]} agent memories lack embeddings",
                        suggested_fix="Generate embeddings for agent memories to improve search",
                        metadata={'missing_embeddings_count': memories_without_embeddings[0]}
                    ))
        
        except Exception as e:
            self.logger.error(f"Performance validation failed: {e}")
            issues.append(ValidationIssue(
                id=str(uuid.uuid4()),
                severity=ValidationSeverity.ERROR,
                category='performance_optimization',
                issue_type='validation_error',
                message=f"Failed to validate performance aspects: {e}",
                suggested_fix="Check database performance and optimization settings"
            ))
        
        return issues
    
    def _validate_data_quality(self) -> List[ValidationIssue]:
        """Validate data quality and completeness."""
        issues = []
        
        try:
            with self.connection_manager.get_connection() as conn:
                # Check for entities with empty names
                empty_names = conn.execute("""
                    SELECT COUNT(*) as count
                    FROM entities
                    WHERE name = '' OR name IS NULL
                """).fetchone()
                
                if empty_names[0] > 0:
                    issues.append(ValidationIssue(
                        id=str(uuid.uuid4()),
                        severity=ValidationSeverity.ERROR,
                        category='data_quality',
                        issue_type='empty_entity_names',
                        message=f"{empty_names[0]} entities have empty or null names",
                        suggested_fix="Ensure all entities have meaningful names",
                        metadata={'empty_names_count': empty_names[0]}
                    ))
                
                # Check for entities with empty file paths
                empty_paths = conn.execute("""
                    SELECT COUNT(*) as count
                    FROM entities
                    WHERE file_path = '' OR file_path IS NULL
                """).fetchone()
                
                if empty_paths[0] > 0:
                    issues.append(ValidationIssue(
                        id=str(uuid.uuid4()),
                        severity=ValidationSeverity.ERROR,
                        category='data_quality',
                        issue_type='empty_file_paths',
                        message=f"{empty_paths[0]} entities have empty or null file paths",
                        suggested_fix="Ensure all entities have valid file paths",
                        metadata={'empty_paths_count': empty_paths[0]}
                    ))
                
                # Check for invalid line numbers
                invalid_lines = conn.execute("""
                    SELECT COUNT(*) as count
                    FROM entities
                    WHERE (line_start IS NOT NULL AND line_start < 1)
                       OR (line_end IS NOT NULL AND line_start IS NOT NULL AND line_end < line_start)
                """).fetchone()
                
                if invalid_lines[0] > 0:
                    issues.append(ValidationIssue(
                        id=str(uuid.uuid4()),
                        severity=ValidationSeverity.WARNING,
                        category='data_quality',
                        issue_type='invalid_line_numbers',
                        message=f"{invalid_lines[0]} entities have invalid line numbers",
                        suggested_fix="Ensure line_start >= 1 and line_end >= line_start",
                        metadata={'invalid_lines_count': invalid_lines[0]}
                    ))
        
        except Exception as e:
            self.logger.error(f"Data quality validation failed: {e}")
            issues.append(ValidationIssue(
                id=str(uuid.uuid4()),
                severity=ValidationSeverity.ERROR,
                category='data_quality',
                issue_type='validation_error',
                message=f"Failed to validate data quality: {e}",
                suggested_fix="Check data quality constraints and validation rules"
            ))
        
        return issues
    
    def _generate_database_stats(self) -> Dict[str, Any]:
        """Generate comprehensive database statistics."""
        stats = {}
        
        try:
            with self.connection_manager.get_connection() as conn:
                # Entity statistics
                entity_stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(embedding) as with_embeddings,
                        COUNT(DISTINCT type) as unique_types,
                        COUNT(DISTINCT file_path) as unique_files
                    FROM entities
                """).fetchone()
                
                stats['entities'] = {
                    'total': entity_stats[0],
                    'with_embeddings': entity_stats[1],
                    'unique_types': entity_stats[2],
                    'unique_files': entity_stats[3],
                    'embedding_coverage': (entity_stats[1] / entity_stats[0] * 100) if entity_stats[0] > 0 else 0
                }
                
                # Relationship statistics
                rel_stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(DISTINCT relationship_type) as unique_types,
                        AVG(confidence) as avg_confidence,
                        MIN(confidence) as min_confidence,
                        MAX(confidence) as max_confidence
                    FROM relationships
                """).fetchone()
                
                stats['relationships'] = {
                    'total': rel_stats[0],
                    'unique_types': rel_stats[1],
                    'avg_confidence': round(rel_stats[2], 3) if rel_stats[2] else 0,
                    'min_confidence': rel_stats[3] if rel_stats[3] else 0,
                    'max_confidence': rel_stats[4] if rel_stats[4] else 0
                }
                
                # Agent memory statistics
                memory_stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(DISTINCT agent_type) as unique_agents,
                        COUNT(embedding) as with_embeddings,
                        COUNT(DISTINCT issue_number) as unique_issues
                    FROM agent_memory
                """).fetchone()
                
                stats['agent_memory'] = {
                    'total': memory_stats[0],
                    'unique_agents': memory_stats[1],
                    'with_embeddings': memory_stats[2],
                    'unique_issues': memory_stats[3],
                    'embedding_coverage': (memory_stats[2] / memory_stats[0] * 100) if memory_stats[0] > 0 else 0
                }
                
        except Exception as e:
            self.logger.error(f"Failed to generate database stats: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def _generate_recommendations(self, 
                                issues: List[ValidationIssue], 
                                database_stats: Dict[str, Any]) -> List[str]:
        """Generate optimization and fix recommendations."""
        recommendations = []
        
        # Count issues by severity
        critical_count = len([i for i in issues if i.severity == ValidationSeverity.CRITICAL])
        error_count = len([i for i in issues if i.severity == ValidationSeverity.ERROR])
        warning_count = len([i for i in issues if i.severity == ValidationSeverity.WARNING])
        
        # Critical issues recommendations
        if critical_count > 0:
            recommendations.append(f"🚨 URGENT: Fix {critical_count} critical referential integrity issues immediately")
        
        # Error recommendations
        if error_count > 0:
            recommendations.append(f"⚠️ Fix {error_count} constraint and data quality errors")
        
        # Warning recommendations
        if warning_count > 0:
            recommendations.append(f"💡 Review {warning_count} warnings to optimize data consistency")
        
        # Database-specific recommendations
        if 'entities' in database_stats:
            entity_stats = database_stats['entities']
            if entity_stats.get('embedding_coverage', 0) < 80:
                recommendations.append("📊 Generate embeddings for more entities to improve vector search performance")
        
        if 'agent_memory' in database_stats:
            memory_stats = database_stats['agent_memory']
            if memory_stats.get('embedding_coverage', 0) < 80:
                recommendations.append("🧠 Generate embeddings for agent memories to improve knowledge retrieval")
        
        # Performance recommendations
        performance_issues = [i for i in issues if i.category == 'performance_optimization']
        if performance_issues:
            recommendations.append("⚡ Review performance optimization suggestions to improve query speed")
        
        # General recommendations
        if len(issues) == 0:
            recommendations.append("✅ Graph validation passed - no issues found!")
        else:
            recommendations.append(f"📋 Total {len(issues)} issues found - prioritize by severity")
        
        return recommendations
    
    def close(self):
        """Close the validator and cleanup resources."""
        self.connection_manager.shutdown()
        self.logger.info("GraphValidator closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()