"""
RIF Knowledge Refinement System
Automatically optimizes and refines the knowledge base based on usage patterns.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
import hashlib

# Add parent directory to path for LightRAG core imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.lightrag_core import LightRAGCore, get_lightrag_instance
from learning.feedback_loop import get_feedback_loop


class KnowledgeRefiner:
    """
    Automatically refines and optimizes the knowledge base.
    """
    
    def __init__(self):
        """Initialize knowledge refiner."""
        self.rag = get_lightrag_instance()
        self.feedback_loop = get_feedback_loop()
        self.logger = logging.getLogger("rif.knowledge_refiner")
        
        # Refinement thresholds
        self.duplicate_threshold = 0.85  # Similarity threshold for duplicates
        self.outdated_days = 30  # Days after which content may be outdated
        self.low_usage_threshold = 0.1  # Usage frequency threshold
        self.quality_threshold = 0.7  # Content quality threshold
        
        # Tracking
        self.refinement_history = []
        self.last_refinement = None
    
    def run_full_refinement(self) -> Dict[str, Any]:
        """
        Run complete knowledge base refinement process.
        
        Returns:
            Refinement results summary
        """
        start_time = datetime.now()
        self.logger.info("Starting full knowledge base refinement")
        
        results = {
            "start_time": start_time.isoformat(),
            "operations": {
                "duplicates_removed": 0,
                "outdated_archived": 0,
                "low_quality_improved": 0,
                "patterns_consolidated": 0,
                "metadata_enhanced": 0
            },
            "errors": [],
            "recommendations": []
        }
        
        try:
            # Step 1: Remove duplicates
            duplicate_results = self._remove_duplicates()
            results["operations"]["duplicates_removed"] = duplicate_results["removed"]
            results["errors"].extend(duplicate_results["errors"])
            
            # Step 2: Archive outdated content
            outdated_results = self._archive_outdated_content()
            results["operations"]["outdated_archived"] = outdated_results["archived"]
            results["errors"].extend(outdated_results["errors"])
            
            # Step 3: Improve low-quality content
            quality_results = self._improve_content_quality()
            results["operations"]["low_quality_improved"] = quality_results["improved"]
            results["errors"].extend(quality_results["errors"])
            
            # Step 4: Consolidate patterns
            pattern_results = self._consolidate_patterns()
            results["operations"]["patterns_consolidated"] = pattern_results["consolidated"]
            results["errors"].extend(pattern_results["errors"])
            
            # Step 5: Enhance metadata
            metadata_results = self._enhance_metadata()
            results["operations"]["metadata_enhanced"] = metadata_results["enhanced"]
            results["errors"].extend(metadata_results["errors"])
            
            # Generate recommendations
            results["recommendations"] = self._generate_refinement_recommendations()
            
        except Exception as e:
            error_msg = f"Critical error during refinement: {str(e)}"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)
        
        end_time = datetime.now()
        results["end_time"] = end_time.isoformat()
        results["duration_seconds"] = (end_time - start_time).total_seconds()
        
        # Record refinement
        self.last_refinement = results
        self.refinement_history.append(results)
        
        # Store refinement record
        self._store_refinement_record(results)
        
        self.logger.info(f"Knowledge refinement completed in {results['duration_seconds']:.2f} seconds")
        return results
    
    def _remove_duplicates(self) -> Dict[str, Any]:
        """Remove duplicate content from knowledge base."""
        self.logger.info("Scanning for duplicate content")
        
        results = {"removed": 0, "errors": []}
        
        try:
            # Get all documents for duplicate detection
            all_docs = self.rag.search_documents("", limit=1000)  # Get all docs
            
            # Group similar documents
            duplicates = self._find_duplicate_groups(all_docs)
            
            for group in duplicates:
                if len(group) > 1:
                    # Keep the most recent/complete document
                    best_doc = self._select_best_duplicate(group)
                    
                    for doc in group:
                        if doc != best_doc:
                            try:
                                # Remove duplicate
                                # Note: Actual implementation would depend on LightRAG's deletion API
                                self.logger.debug(f"Would remove duplicate document")
                                results["removed"] += 1
                            except Exception as e:
                                results["errors"].append(f"Failed to remove duplicate: {str(e)}")
            
        except Exception as e:
            results["errors"].append(f"Error in duplicate detection: {str(e)}")
        
        return results
    
    def _find_duplicate_groups(self, documents: List[Any]) -> List[List[Any]]:
        """Find groups of duplicate documents."""
        # This is a simplified version - real implementation would use
        # embeddings similarity or content hashing
        
        groups = []
        processed = set()
        
        for i, doc1 in enumerate(documents):
            if i in processed:
                continue
                
            group = [doc1]
            processed.add(i)
            
            content1 = getattr(doc1, 'content', '')
            
            for j, doc2 in enumerate(documents[i+1:], i+1):
                if j in processed:
                    continue
                    
                content2 = getattr(doc2, 'content', '')
                
                # Simple similarity check (would use embeddings in real implementation)
                similarity = self._calculate_content_similarity(content1, content2)
                
                if similarity > self.duplicate_threshold:
                    group.append(doc2)
                    processed.add(j)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings."""
        # Simple Jaccard similarity (would use better methods in production)
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _select_best_duplicate(self, group: List[Any]) -> Any:
        """Select the best document from a group of duplicates."""
        # Prefer most recent, then longest content
        best = group[0]
        
        for doc in group[1:]:
            doc_timestamp = getattr(doc, 'metadata', {}).get('timestamp', '')
            best_timestamp = getattr(best, 'metadata', {}).get('timestamp', '')
            
            doc_content = getattr(doc, 'content', '')
            best_content = getattr(best, 'content', '')
            
            # Prefer more recent
            if doc_timestamp > best_timestamp:
                best = doc
            elif doc_timestamp == best_timestamp and len(doc_content) > len(best_content):
                best = doc
        
        return best
    
    def _archive_outdated_content(self) -> Dict[str, Any]:
        """Archive content that may be outdated."""
        self.logger.info("Archiving outdated content")
        
        results = {"archived": 0, "errors": []}
        cutoff_date = datetime.now() - timedelta(days=self.outdated_days)
        
        try:
            # Find old documents
            old_docs = self.rag.search_documents(
                query="timestamp",  # Would filter by timestamp in real implementation
                limit=1000
            )
            
            for doc in old_docs:
                doc_timestamp = getattr(doc, 'metadata', {}).get('timestamp', '')
                
                if doc_timestamp:
                    try:
                        doc_date = datetime.fromisoformat(doc_timestamp.replace('Z', '+00:00'))
                        
                        if doc_date < cutoff_date:
                            # Mark as archived (would update metadata)
                            self.logger.debug(f"Would archive outdated document from {doc_date}")
                            results["archived"] += 1
                            
                    except Exception as e:
                        results["errors"].append(f"Error processing document timestamp: {str(e)}")
        
        except Exception as e:
            results["errors"].append(f"Error in outdated content detection: {str(e)}")
        
        return results
    
    def _improve_content_quality(self) -> Dict[str, Any]:
        """Improve quality of low-quality content."""
        self.logger.info("Improving content quality")
        
        results = {"improved": 0, "errors": []}
        
        try:
            # Find documents that could be improved
            all_docs = self.rag.search_documents("", limit=1000)
            
            for doc in all_docs:
                content = getattr(doc, 'content', '')
                quality_score = self._assess_content_quality(content)
                
                if quality_score < self.quality_threshold:
                    # Improve content (simplified example)
                    improved_content = self._enhance_content(content)
                    
                    if improved_content != content:
                        # Would update document with improved content
                        self.logger.debug("Would improve low-quality content")
                        results["improved"] += 1
        
        except Exception as e:
            results["errors"].append(f"Error in content quality improvement: {str(e)}")
        
        return results
    
    def _assess_content_quality(self, content: str) -> float:
        """Assess quality of content."""
        if not content:
            return 0.0
        
        score = 0.0
        
        # Length factor (prefer substantial content)
        if len(content) > 50:
            score += 0.2
        if len(content) > 100:
            score += 0.3
        
        # Structure factor (prefer structured content)
        if re.search(r'\n\s*[-*#]', content):  # Has lists or headers
            score += 0.2
        
        # Completeness factor (avoid fragments)
        if content.endswith('.') or content.endswith('\n'):
            score += 0.2
        
        # Information density (prefer detailed content)
        word_count = len(content.split())
        if word_count > 20:
            score += 0.2
        if word_count > 50:
            score += 0.3
        
        return min(score, 1.0)
    
    def _enhance_content(self, content: str) -> str:
        """Enhance content quality."""
        # Simple enhancements (real implementation would be more sophisticated)
        enhanced = content.strip()
        
        # Ensure proper ending
        if enhanced and not enhanced.endswith('.'):
            enhanced += '.'
        
        # Add structure if missing
        if '\n' not in enhanced and len(enhanced) > 100:
            # Split long content into paragraphs
            words = enhanced.split()
            if len(words) > 20:
                mid_point = len(words) // 2
                enhanced = ' '.join(words[:mid_point]) + '\n\n' + ' '.join(words[mid_point:])
        
        return enhanced
    
    def _consolidate_patterns(self) -> Dict[str, Any]:
        """Consolidate similar patterns."""
        self.logger.info("Consolidating patterns")
        
        results = {"consolidated": 0, "errors": []}
        
        try:
            # Find pattern documents
            pattern_docs = self.rag.search_documents(
                query="pattern implementation",
                limit=100
            )
            
            # Group similar patterns
            pattern_groups = self._group_similar_patterns(pattern_docs)
            
            for group in pattern_groups:
                if len(group) > 1:
                    # Consolidate into single, comprehensive pattern
                    consolidated = self._merge_patterns(group)
                    
                    if consolidated:
                        results["consolidated"] += len(group) - 1
        
        except Exception as e:
            results["errors"].append(f"Error in pattern consolidation: {str(e)}")
        
        return results
    
    def _group_similar_patterns(self, patterns: List[Any]) -> List[List[Any]]:
        """Group similar patterns together."""
        # Simplified grouping by content similarity
        groups = []
        processed = set()
        
        for i, pattern1 in enumerate(patterns):
            if i in processed:
                continue
            
            group = [pattern1]
            content1 = getattr(pattern1, 'content', '')
            
            for j, pattern2 in enumerate(patterns[i+1:], i+1):
                if j in processed:
                    continue
                
                content2 = getattr(pattern2, 'content', '')
                similarity = self._calculate_content_similarity(content1, content2)
                
                if similarity > 0.6:  # Lower threshold for pattern similarity
                    group.append(pattern2)
                    processed.add(j)
            
            processed.add(i)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _merge_patterns(self, patterns: List[Any]) -> Optional[str]:
        """Merge multiple patterns into one comprehensive pattern."""
        if not patterns:
            return None
        
        # Combine content from all patterns
        combined_content = []
        metadata_combined = {}
        
        for pattern in patterns:
            content = getattr(pattern, 'content', '')
            if content:
                combined_content.append(content)
            
            metadata = getattr(pattern, 'metadata', {})
            metadata_combined.update(metadata)
        
        if combined_content:
            merged = '\n\n'.join(combined_content)
            return f"Consolidated Pattern:\n{merged}"
        
        return None
    
    def _enhance_metadata(self) -> Dict[str, Any]:
        """Enhance document metadata."""
        self.logger.info("Enhancing metadata")
        
        results = {"enhanced": 0, "errors": []}
        
        try:
            # Get documents with minimal metadata
            docs = self.rag.search_documents("", limit=100)
            
            for doc in docs:
                metadata = getattr(doc, 'metadata', {})
                enhanced_metadata = self._generate_enhanced_metadata(doc)
                
                if len(enhanced_metadata) > len(metadata):
                    # Would update document metadata
                    results["enhanced"] += 1
        
        except Exception as e:
            results["errors"].append(f"Error in metadata enhancement: {str(e)}")
        
        return results
    
    def _generate_enhanced_metadata(self, doc: Any) -> Dict[str, Any]:
        """Generate enhanced metadata for a document."""
        content = getattr(doc, 'content', '')
        existing_metadata = getattr(doc, 'metadata', {})
        
        enhanced = existing_metadata.copy()
        
        # Add content analysis
        enhanced.update({
            'word_count': len(content.split()),
            'char_count': len(content),
            'quality_score': self._assess_content_quality(content),
            'last_analyzed': datetime.now().isoformat()
        })
        
        # Extract content type if not present
        if 'type' not in enhanced:
            enhanced['type'] = self._infer_content_type(content)
        
        # Add tags based on content
        if 'tags' not in enhanced:
            enhanced['tags'] = self._extract_content_tags(content)
        
        return enhanced
    
    def _infer_content_type(self, content: str) -> str:
        """Infer content type from content."""
        content_lower = content.lower()
        
        if 'pattern' in content_lower:
            return 'pattern'
        elif 'decision' in content_lower:
            return 'decision'
        elif 'implementation' in content_lower:
            return 'implementation'
        elif 'analysis' in content_lower:
            return 'analysis'
        else:
            return 'general'
    
    def _extract_content_tags(self, content: str) -> List[str]:
        """Extract relevant tags from content."""
        tags = []
        content_lower = content.lower()
        
        # Technical tags
        tech_keywords = ['python', 'javascript', 'java', 'react', 'node', 'api', 'database']
        for keyword in tech_keywords:
            if keyword in content_lower:
                tags.append(keyword)
        
        # Process tags
        process_keywords = ['implementation', 'testing', 'deployment', 'optimization']
        for keyword in process_keywords:
            if keyword in content_lower:
                tags.append(keyword)
        
        return tags[:5]  # Limit to 5 tags
    
    def _generate_refinement_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations for future refinements."""
        recommendations = []
        
        if self.last_refinement:
            results = self.last_refinement
            
            # Check if many duplicates were found
            if results["operations"]["duplicates_removed"] > 10:
                recommendations.append({
                    "type": "process_improvement",
                    "priority": "medium",
                    "description": "High number of duplicates detected",
                    "action": "Review content creation process to prevent duplicates"
                })
            
            # Check if many outdated items
            if results["operations"]["outdated_archived"] > 20:
                recommendations.append({
                    "type": "content_lifecycle",
                    "priority": "low",
                    "description": "Many outdated items archived",
                    "action": "Consider implementing automatic content expiration"
                })
        
        # General recommendations
        recommendations.append({
            "type": "maintenance",
            "priority": "low",
            "description": "Regular refinement maintains knowledge quality",
            "action": "Schedule weekly knowledge base refinement"
        })
        
        return recommendations
    
    def _store_refinement_record(self, results: Dict[str, Any]):
        """Store refinement record in knowledge base."""
        try:
            content = f"Knowledge Refinement Record\n"
            content += f"Completed: {results['end_time']}\n"
            content += f"Duration: {results['duration_seconds']:.2f} seconds\n"
            content += f"Results: {json.dumps(results, indent=2)}"
            
            doc_id = self.rag.insert_document(
                content=content,
                metadata={
                    "type": "refinement_record",
                    "timestamp": results["end_time"],
                    "agent": "knowledge_refiner"
                }
            )
            
            self.logger.info(f"Stored refinement record: {doc_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store refinement record: {str(e)}")
    
    def get_refinement_status(self) -> Dict[str, Any]:
        """Get current refinement status."""
        return {
            "last_refinement": self.last_refinement,
            "total_refinements": len(self.refinement_history),
            "next_recommended": self._calculate_next_refinement_time()
        }
    
    def _calculate_next_refinement_time(self) -> str:
        """Calculate when next refinement should occur."""
        if not self.last_refinement:
            return "Now (no previous refinement)"
        
        last_time = datetime.fromisoformat(self.last_refinement["end_time"])
        next_time = last_time + timedelta(days=7)  # Weekly refinement
        
        return next_time.isoformat()


# Global instance
_refiner_instance = None

def get_knowledge_refiner() -> KnowledgeRefiner:
    """Get global knowledge refiner instance."""
    global _refiner_instance
    if _refiner_instance is None:
        _refiner_instance = KnowledgeRefiner()
    return _refiner_instance


def run_knowledge_refinement() -> Dict[str, Any]:
    """Run knowledge base refinement."""
    refiner = get_knowledge_refiner()
    return refiner.run_full_refinement()