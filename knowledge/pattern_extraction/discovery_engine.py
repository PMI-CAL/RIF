"""
Pattern Discovery Engine - Core pattern identification and extraction.

This module provides the foundation for pattern discovery in the RIF system,
implementing multi-method extraction algorithms, signature generation,
and deduplication logic.
"""

import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re
from pathlib import Path

from ..interface import get_knowledge_system, KnowledgeInterface
from .cache import PatternExtractionCache, get_global_cache, cached_pattern_extraction


@dataclass
class PatternSignature:
    """Unique signature for pattern identification and deduplication."""
    content_hash: str
    structure_hash: str
    metadata_hash: str
    combined_hash: str
    
    @classmethod
    def from_pattern(cls, pattern_data: Dict[str, Any]) -> 'PatternSignature':
        """Generate signature from pattern data."""
        # Content hash - core functional content
        content_str = f"{pattern_data.get('title', '')}{pattern_data.get('description', '')}"
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]
        
        # Structure hash - architectural structure
        structure_elements = []
        if 'architecture' in pattern_data:
            structure_elements.append(json.dumps(pattern_data['architecture'], sort_keys=True))
        if 'components' in pattern_data:
            structure_elements.append(json.dumps(pattern_data['components'], sort_keys=True))
        structure_str = '|'.join(structure_elements)
        structure_hash = hashlib.sha256(structure_str.encode()).hexdigest()[:16]
        
        # Metadata hash - context and classification
        metadata_elements = [
            pattern_data.get('complexity', ''),
            pattern_data.get('domain', ''),
            '|'.join(sorted(pattern_data.get('tags', [])))
        ]
        metadata_str = '|'.join(metadata_elements)
        metadata_hash = hashlib.sha256(metadata_str.encode()).hexdigest()[:16]
        
        # Combined hash for global uniqueness
        combined_str = f"{content_hash}|{structure_hash}|{metadata_hash}"
        combined_hash = hashlib.sha256(combined_str.encode()).hexdigest()[:16]
        
        return cls(
            content_hash=content_hash,
            structure_hash=structure_hash,
            metadata_hash=metadata_hash,
            combined_hash=combined_hash
        )


@dataclass
class ExtractedPattern:
    """Represents a pattern extracted from completed work."""
    title: str
    description: str
    pattern_type: str
    source: str
    content: Dict[str, Any]
    context: Dict[str, Any]
    signature: PatternSignature
    extraction_method: str
    confidence: float
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['signature'] = asdict(self.signature)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractedPattern':
        """Create from dictionary."""
        signature_data = data.pop('signature')
        data['signature'] = PatternSignature(**signature_data)
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class PatternDiscoveryEngine:
    """
    Core pattern discovery and extraction engine.
    
    This class orchestrates the pattern discovery process across multiple
    data sources and extraction methods, providing deduplication,
    signature generation, and unified pattern structure.
    """
    
    def __init__(self, knowledge_system: Optional[KnowledgeInterface] = None, enable_cache: bool = True):
        self.knowledge = knowledge_system or get_knowledge_system()
        self.logger = logging.getLogger(__name__)
        
        # Pattern type extractors will be registered here
        self.extractors = {}
        
        # Deduplication cache
        self.pattern_signatures: Set[str] = set()
        
        # Performance cache
        self._cache = get_global_cache() if enable_cache else None
        
        # Statistics
        self.extraction_stats = {
            'patterns_discovered': 0,
            'duplicates_filtered': 0,
            'errors_encountered': 0,
            'extraction_methods_used': set(),
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def register_extractor(self, pattern_type: str, extractor):
        """Register a pattern extractor for a specific type."""
        self.extractors[pattern_type] = extractor
        self.logger.info(f"Registered extractor for pattern type: {pattern_type}")
    
    def discover_patterns(self, completed_issue: Dict[str, Any]) -> List[ExtractedPattern]:
        """
        Main entry point for pattern discovery from a completed issue.
        
        Args:
            completed_issue: Dictionary containing completed issue data with:
                - issue_number: GitHub issue number
                - title: Issue title
                - body: Issue description
                - code_changes: Git diff data
                - history: Issue state history
                - decisions: Architectural decisions made
                - outcome: Final result and metrics
        
        Returns:
            List of discovered patterns
        """
        patterns = []
        issue_id = completed_issue.get('issue_number', 'unknown')
        
        self.logger.info(f"Starting pattern discovery for issue #{issue_id}")
        
        try:
            # Validate required fields
            if not self._validate_issue_data(completed_issue):
                raise ValueError("Invalid issue data: missing required fields")
                
            # Extract patterns using all registered extractors
            for pattern_type, extractor in self.extractors.items():
                try:
                    extracted = extractor.extract_patterns(completed_issue)
                    patterns.extend(extracted)
                    self.extraction_stats['extraction_methods_used'].add(pattern_type)
                except Exception as e:
                    self.logger.error(f"Error in {pattern_type} extraction: {e}")
                    self.extraction_stats['errors_encountered'] += 1
            
            # Deduplicate patterns
            unique_patterns = self._deduplicate_patterns(patterns)
            
            # Store patterns in knowledge base
            for pattern in unique_patterns:
                self._store_pattern(pattern)
            
            self.extraction_stats['patterns_discovered'] += len(unique_patterns)
            
            self.logger.info(
                f"Pattern discovery complete for issue #{issue_id}: "
                f"{len(unique_patterns)} unique patterns discovered"
            )
            
            return unique_patterns
            
        except Exception as e:
            self.logger.error(f"Pattern discovery failed for issue #{issue_id}: {e}")
            self.extraction_stats['errors_encountered'] += 1
            return []
    
    def _deduplicate_patterns(self, patterns: List[ExtractedPattern]) -> List[ExtractedPattern]:
        """
        Remove duplicate patterns using signature-based deduplication.
        
        Args:
            patterns: List of extracted patterns
            
        Returns:
            List of unique patterns
        """
        unique_patterns = []
        seen_signatures = set()
        
        for pattern in patterns:
            combined_hash = pattern.signature.combined_hash
            
            if combined_hash in seen_signatures or combined_hash in self.pattern_signatures:
                self.extraction_stats['duplicates_filtered'] += 1
                self.logger.debug(f"Filtered duplicate pattern: {pattern.title}")
                continue
            
            seen_signatures.add(combined_hash)
            self.pattern_signatures.add(combined_hash)
            unique_patterns.append(pattern)
        
        return unique_patterns
    
    def _store_pattern(self, pattern: ExtractedPattern) -> bool:
        """
        Store pattern in knowledge base.
        
        Args:
            pattern: Pattern to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pattern_data = pattern.to_dict()
            
            metadata = {
                'type': 'extracted_pattern',
                'pattern_type': pattern.pattern_type,
                'source': pattern.source,
                'extraction_method': pattern.extraction_method,
                'confidence': pattern.confidence,
                'signature': pattern.signature.combined_hash
            }
            
            doc_id = self.knowledge.store_knowledge(
                collection='patterns',
                content=pattern_data,
                metadata=metadata,
                doc_id=f"pattern_{pattern.signature.combined_hash}"
            )
            
            if doc_id:
                self.logger.debug(f"Stored pattern: {pattern.title} ({doc_id})")
                return True
            else:
                self.logger.warning(f"Failed to store pattern: {pattern.title}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error storing pattern {pattern.title}: {e}")
            return False
    
    def find_similar_patterns(self, 
                            query_pattern: ExtractedPattern, 
                            similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Find patterns similar to the query pattern.
        
        Args:
            query_pattern: Pattern to find similarities for
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of similar patterns with similarity scores
        """
        try:
            # Search for patterns with similar content
            search_query = f"{query_pattern.title} {query_pattern.description}"
            
            similar_patterns = self.knowledge.retrieve_knowledge(
                query=search_query,
                collection='patterns',
                n_results=10,
                filters={'pattern_type': query_pattern.pattern_type}
            )
            
            # Calculate signature-based similarity
            results = []
            for pattern in similar_patterns:
                similarity = self._calculate_pattern_similarity(query_pattern, pattern)
                if similarity >= similarity_threshold:
                    pattern['similarity_score'] = similarity
                    results.append(pattern)
            
            # Sort by similarity score
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error finding similar patterns: {e}")
            return []
    
    def _calculate_pattern_similarity(self, 
                                    pattern1: ExtractedPattern, 
                                    pattern2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two patterns.
        
        Args:
            pattern1: First pattern (ExtractedPattern)
            pattern2: Second pattern (from knowledge base)
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            # Get signature from stored pattern
            if 'content' in pattern2 and 'signature' in pattern2['content']:
                sig2_data = pattern2['content']['signature']
                sig2 = PatternSignature(**sig2_data)
            else:
                # Calculate signature if not available
                sig2 = PatternSignature.from_pattern(pattern2.get('content', {}))
            
            sig1 = pattern1.signature
            
            # Compare signatures
            content_match = sig1.content_hash == sig2.content_hash
            structure_match = sig1.structure_hash == sig2.structure_hash
            metadata_match = sig1.metadata_hash == sig2.metadata_hash
            
            # Weight different aspects
            weights = {'content': 0.5, 'structure': 0.3, 'metadata': 0.2}
            
            similarity = (
                weights['content'] * (1.0 if content_match else 0.0) +
                weights['structure'] * (1.0 if structure_match else 0.0) +
                weights['metadata'] * (1.0 if metadata_match else 0.0)
            )
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern similarity: {e}")
            return 0.0
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """
        Get extraction statistics.
        
        Returns:
            Dictionary with extraction statistics
        """
        stats = self.extraction_stats.copy()
        stats['extraction_methods_used'] = list(stats['extraction_methods_used'])
        stats['unique_patterns_cached'] = len(self.pattern_signatures)
        return stats
    
    def clear_cache(self):
        """Clear the deduplication cache."""
        self.pattern_signatures.clear()
        self.logger.info("Pattern signature cache cleared")
    
    def _validate_issue_data(self, issue_data: Dict[str, Any]) -> bool:
        """
        Validate that issue data has minimum required structure.
        
        Args:
            issue_data: Issue data to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(issue_data, dict):
            return False
        
        # Check for at least one required field that makes sense for pattern extraction
        required_field_groups = [
            ['code_changes'],
            ['decisions'],
            ['history'],  
            ['agent_interactions'],
            ['title', 'body']
        ]
        
        # At least one group of fields should be present
        for field_group in required_field_groups:
            if all(field in issue_data and issue_data[field] for field in field_group):
                return True
        
        # If issue_number is None, it's definitely malformed
        if issue_data.get('issue_number') is None:
            return False
            
        return False
    
    def validate_pattern_structure(self, pattern_data: Dict[str, Any]) -> bool:
        """
        Validate that pattern data has required structure.
        
        Args:
            pattern_data: Pattern data to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['title', 'description', 'source']
        
        for field in required_fields:
            if field not in pattern_data or not pattern_data[field]:
                self.logger.warning(f"Pattern missing required field: {field}")
                return False
        
        return True
    
    def export_patterns(self, output_path: str, pattern_type: Optional[str] = None) -> bool:
        """
        Export discovered patterns to file.
        
        Args:
            output_path: Path to export file
            pattern_type: Optional filter by pattern type
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filters = {'type': 'extracted_pattern'}
            if pattern_type:
                filters['pattern_type'] = pattern_type
            
            patterns = self.knowledge.retrieve_knowledge(
                query="*",
                collection='patterns',
                n_results=1000,
                filters=filters
            )
            
            # Handle case where patterns might not be a list (e.g., Mock object in tests)
            if hasattr(patterns, '__len__'):
                pattern_count = len(patterns)
            else:
                pattern_count = 0
                patterns = []
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'pattern_count': pattern_count,
                'pattern_type_filter': pattern_type,
                'patterns': patterns
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported {len(patterns)} patterns to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting patterns: {e}")
            return False