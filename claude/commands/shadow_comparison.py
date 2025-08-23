#!/usr/bin/env python3
"""
Shadow Mode Comparison Framework
Advanced comparison and analysis tools for evaluating differences between
knowledge systems in shadow mode testing.
"""

import json
import os
import sys
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import difflib
import statistics


@dataclass
class ContentDifference:
    """Represents a specific content difference."""
    type: str  # 'content', 'metadata', 'structure', 'performance'
    field: str
    primary_value: Any
    shadow_value: Any
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    impact_score: float  # 0.0 to 1.0


@dataclass
class ComparisonReport:
    """Detailed comparison report."""
    operation: str
    timestamp: str
    primary_system: str
    shadow_system: str
    overall_similarity: float
    differences: List[ContentDifference]
    performance_analysis: Dict[str, Any]
    recommendations: List[str]
    metadata: Dict[str, Any]


class AdvancedComparator:
    """Advanced comparison engine for knowledge system results."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.AdvancedComparator")
        
        # Similarity thresholds
        self.thresholds = {
            'content_exact_match': 0.95,
            'content_high_similarity': 0.80,
            'content_medium_similarity': 0.60,
            'metadata_similarity': 0.70,
            'performance_variance_high': 2.0,
            'performance_variance_medium': 1.5
        }
        
        # Update thresholds from config
        comparison_config = self.config.get('comparison', {})
        if comparison_config:
            self.thresholds.update({
                'content_exact_match': comparison_config.get('content_similarity_threshold', 0.95),
                'performance_variance_high': comparison_config.get('timing_variance_threshold', 2.0)
            })
    
    def compare_store_results(self, primary_result, shadow_result, operation_context: Dict[str, Any] = None) -> ComparisonReport:
        """Compare store operation results in detail."""
        differences = []
        
        # Compare success status
        if primary_result.success != shadow_result.success:
            differences.append(ContentDifference(
                type='status',
                field='success',
                primary_value=primary_result.success,
                shadow_value=shadow_result.success,
                severity='critical',
                description=f'Operation success status differs between systems',
                impact_score=1.0
            ))
        
        # Compare returned document IDs
        if primary_result.success and shadow_result.success:
            if primary_result.data != shadow_result.data:
                # This is expected for different systems, but note it
                differences.append(ContentDifference(
                    type='identifier',
                    field='document_id',
                    primary_value=primary_result.data,
                    shadow_value=shadow_result.data,
                    severity='low',
                    description='Document IDs differ (expected for different systems)',
                    impact_score=0.1
                ))
        
        # Compare errors if present
        if primary_result.error or shadow_result.error:
            if primary_result.error != shadow_result.error:
                differences.append(ContentDifference(
                    type='error',
                    field='error_message',
                    primary_value=primary_result.error,
                    shadow_value=shadow_result.error,
                    severity='high',
                    description='Different error messages between systems',
                    impact_score=0.8
                ))
        
        # Performance comparison
        performance_analysis = self._analyze_performance(primary_result, shadow_result)
        if performance_analysis['significant_difference']:
            differences.append(ContentDifference(
                type='performance',
                field='duration',
                primary_value=primary_result.duration_ms,
                shadow_value=shadow_result.duration_ms,
                severity='medium',
                description=f'Performance difference: {performance_analysis["ratio"]:.2f}x',
                impact_score=0.3
            ))
        
        # Calculate overall similarity
        overall_similarity = self._calculate_overall_similarity(differences)
        
        # Generate recommendations
        recommendations = self._generate_store_recommendations(differences, performance_analysis)
        
        return ComparisonReport(
            operation='store_knowledge',
            timestamp=datetime.utcnow().isoformat(),
            primary_system='primary',
            shadow_system='shadow',
            overall_similarity=overall_similarity,
            differences=differences,
            performance_analysis=performance_analysis,
            recommendations=recommendations,
            metadata=operation_context or {}
        )
    
    def compare_retrieve_results(self, primary_result, shadow_result, operation_context: Dict[str, Any] = None) -> ComparisonReport:
        """Compare retrieve operation results in detail."""
        differences = []
        
        # Compare success status
        if primary_result.success != shadow_result.success:
            differences.append(ContentDifference(
                type='status',
                field='success',
                primary_value=primary_result.success,
                shadow_value=shadow_result.success,
                severity='critical',
                description=f'Operation success status differs between systems',
                impact_score=1.0
            ))
        
        # Compare results if both successful
        if primary_result.success and shadow_result.success and primary_result.data and shadow_result.data:
            primary_results = primary_result.data
            shadow_results = shadow_result.data
            
            # Compare result counts
            if len(primary_results) != len(shadow_results):
                differences.append(ContentDifference(
                    type='structure',
                    field='result_count',
                    primary_value=len(primary_results),
                    shadow_value=len(shadow_results),
                    severity='medium',
                    description=f'Different number of results returned',
                    impact_score=0.5
                ))
            
            # Detailed content comparison
            content_analysis = self._analyze_retrieve_content(primary_results, shadow_results)
            differences.extend(content_analysis['differences'])
            
            # Relevance score comparison
            relevance_analysis = self._analyze_relevance_scores(primary_results, shadow_results)
            if relevance_analysis['significant_difference']:
                differences.append(ContentDifference(
                    type='relevance',
                    field='relevance_scores',
                    primary_value=relevance_analysis['primary_avg'],
                    shadow_value=relevance_analysis['shadow_avg'],
                    severity='medium',
                    description='Significant difference in relevance scoring',
                    impact_score=0.4
                ))
        
        # Performance comparison
        performance_analysis = self._analyze_performance(primary_result, shadow_result)
        if performance_analysis['significant_difference']:
            differences.append(ContentDifference(
                type='performance',
                field='duration',
                primary_value=primary_result.duration_ms,
                shadow_value=shadow_result.duration_ms,
                severity='medium',
                description=f'Performance difference: {performance_analysis["ratio"]:.2f}x',
                impact_score=0.3
            ))
        
        # Calculate overall similarity
        overall_similarity = self._calculate_overall_similarity(differences)
        
        # Generate recommendations
        recommendations = self._generate_retrieve_recommendations(differences, performance_analysis)
        
        return ComparisonReport(
            operation='retrieve_knowledge',
            timestamp=datetime.utcnow().isoformat(),
            primary_system='primary',
            shadow_system='shadow',
            overall_similarity=overall_similarity,
            differences=differences,
            performance_analysis=performance_analysis,
            recommendations=recommendations,
            metadata=operation_context or {}
        )
    
    def _analyze_performance(self, primary_result, shadow_result) -> Dict[str, Any]:
        """Analyze performance differences between results."""
        primary_ms = primary_result.duration_ms
        shadow_ms = shadow_result.duration_ms
        
        if primary_ms == 0:
            ratio = float('inf') if shadow_ms > 0 else 1.0
        else:
            ratio = shadow_ms / primary_ms
        
        significant_difference = (
            ratio > self.thresholds['performance_variance_high'] or 
            ratio < (1.0 / self.thresholds['performance_variance_high'])
        )
        
        return {
            'primary_duration_ms': primary_ms,
            'shadow_duration_ms': shadow_ms,
            'ratio': ratio,
            'absolute_difference_ms': abs(shadow_ms - primary_ms),
            'significant_difference': significant_difference,
            'faster_system': 'primary' if primary_ms < shadow_ms else 'shadow',
            'performance_category': self._categorize_performance_difference(ratio)
        }
    
    def _categorize_performance_difference(self, ratio: float) -> str:
        """Categorize performance difference severity."""
        if ratio > self.thresholds['performance_variance_high'] or ratio < (1.0 / self.thresholds['performance_variance_high']):
            return 'high'
        elif ratio > self.thresholds['performance_variance_medium'] or ratio < (1.0 / self.thresholds['performance_variance_medium']):
            return 'medium'
        else:
            return 'low'
    
    def _analyze_retrieve_content(self, primary_results: List[Dict], shadow_results: List[Dict]) -> Dict[str, Any]:
        """Analyze content differences in retrieve results."""
        differences = []
        
        # Create maps for easier comparison
        primary_by_id = {r.get('id', str(i)): r for i, r in enumerate(primary_results)}
        shadow_by_id = {r.get('id', str(i)): r for i, r in enumerate(shadow_results)}
        
        # Check for content overlap
        primary_ids = set(primary_by_id.keys())
        shadow_ids = set(shadow_by_id.keys())
        
        common_ids = primary_ids.intersection(shadow_ids)
        primary_only = primary_ids - shadow_ids
        shadow_only = shadow_ids - primary_ids
        
        # Analyze common results
        for result_id in common_ids:
            primary_item = primary_by_id[result_id]
            shadow_item = shadow_by_id[result_id]
            
            # Compare content
            content_similarity = self._calculate_text_similarity(
                primary_item.get('content', ''),
                shadow_item.get('content', '')
            )
            
            if content_similarity < self.thresholds['content_high_similarity']:
                differences.append(ContentDifference(
                    type='content',
                    field=f'result_{result_id}_content',
                    primary_value=primary_item.get('content', '')[:100] + '...',
                    shadow_value=shadow_item.get('content', '')[:100] + '...',
                    severity=self._severity_from_similarity(content_similarity),
                    description=f'Content differs for result {result_id} (similarity: {content_similarity:.2f})',
                    impact_score=1.0 - content_similarity
                ))
            
            # Compare metadata
            metadata_similarity = self._calculate_metadata_similarity(
                primary_item.get('metadata', {}),
                shadow_item.get('metadata', {})
            )
            
            if metadata_similarity < self.thresholds['metadata_similarity']:
                differences.append(ContentDifference(
                    type='metadata',
                    field=f'result_{result_id}_metadata',
                    primary_value=primary_item.get('metadata', {}),
                    shadow_value=shadow_item.get('metadata', {}),
                    severity=self._severity_from_similarity(metadata_similarity),
                    description=f'Metadata differs for result {result_id} (similarity: {metadata_similarity:.2f})',
                    impact_score=(1.0 - metadata_similarity) * 0.5  # Lower impact than content
                ))
        
        # Note unique results
        if primary_only:
            differences.append(ContentDifference(
                type='structure',
                field='unique_primary_results',
                primary_value=list(primary_only),
                shadow_value=[],
                severity='medium',
                description=f'{len(primary_only)} results only in primary system',
                impact_score=len(primary_only) / max(len(primary_results), 1) * 0.7
            ))
        
        if shadow_only:
            differences.append(ContentDifference(
                type='structure',
                field='unique_shadow_results',
                primary_value=[],
                shadow_value=list(shadow_only),
                severity='medium',
                description=f'{len(shadow_only)} results only in shadow system',
                impact_score=len(shadow_only) / max(len(shadow_results), 1) * 0.7
            ))
        
        return {
            'differences': differences,
            'common_results': len(common_ids),
            'primary_unique': len(primary_only),
            'shadow_unique': len(shadow_only),
            'content_overlap_ratio': len(common_ids) / max(len(primary_results), len(shadow_results), 1)
        }
    
    def _analyze_relevance_scores(self, primary_results: List[Dict], shadow_results: List[Dict]) -> Dict[str, Any]:
        """Analyze relevance score differences."""
        primary_scores = [r.get('distance', 0) for r in primary_results if 'distance' in r]
        shadow_scores = [r.get('distance', 0) for r in shadow_results if 'distance' in r]
        
        if not primary_scores or not shadow_scores:
            return {'significant_difference': False}
        
        primary_avg = statistics.mean(primary_scores) if primary_scores else 0
        shadow_avg = statistics.mean(shadow_scores) if shadow_scores else 0
        
        # Convert distances to similarity scores (lower distance = higher similarity)
        primary_sim_avg = 1.0 - primary_avg if primary_avg <= 1.0 else 0
        shadow_sim_avg = 1.0 - shadow_avg if shadow_avg <= 1.0 else 0
        
        difference = abs(primary_sim_avg - shadow_sim_avg)
        significant_difference = difference > 0.2  # 20% difference threshold
        
        return {
            'primary_avg': primary_avg,
            'shadow_avg': shadow_avg,
            'primary_sim_avg': primary_sim_avg,
            'shadow_sim_avg': shadow_sim_avg,
            'difference': difference,
            'significant_difference': significant_difference
        }
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 and not text2:
            return 1.0
        
        if not text1 or not text2:
            return 0.0
        
        # Use SequenceMatcher for text similarity
        matcher = difflib.SequenceMatcher(None, text1.lower(), text2.lower())
        return matcher.ratio()
    
    def _calculate_metadata_similarity(self, meta1: Dict, meta2: Dict) -> float:
        """Calculate similarity between two metadata dictionaries."""
        if not meta1 and not meta2:
            return 1.0
        
        if not meta1 or not meta2:
            return 0.0
        
        # Compare keys
        keys1 = set(meta1.keys())
        keys2 = set(meta2.keys())
        
        common_keys = keys1.intersection(keys2)
        all_keys = keys1.union(keys2)
        
        if not all_keys:
            return 1.0
        
        key_similarity = len(common_keys) / len(all_keys)
        
        # Compare values for common keys
        value_similarity = 0.0
        if common_keys:
            for key in common_keys:
                val1 = str(meta1[key])
                val2 = str(meta2[key])
                
                if val1 == val2:
                    value_similarity += 1.0
                else:
                    # Partial similarity for string values
                    value_similarity += self._calculate_text_similarity(val1, val2) * 0.5
            
            value_similarity /= len(common_keys)
        
        # Combine key and value similarities
        return (key_similarity + value_similarity) / 2
    
    def _severity_from_similarity(self, similarity: float) -> str:
        """Determine severity based on similarity score."""
        if similarity >= self.thresholds['content_exact_match']:
            return 'low'
        elif similarity >= self.thresholds['content_high_similarity']:
            return 'low'
        elif similarity >= self.thresholds['content_medium_similarity']:
            return 'medium'
        else:
            return 'high'
    
    def _calculate_overall_similarity(self, differences: List[ContentDifference]) -> float:
        """Calculate overall similarity score."""
        if not differences:
            return 1.0
        
        # Weight differences by impact score
        total_impact = sum(diff.impact_score for diff in differences)
        
        # Normalize to 0-1 scale (assuming max possible impact per difference is 1.0)
        max_possible_impact = len(differences)
        
        if max_possible_impact == 0:
            return 1.0
        
        normalized_impact = total_impact / max_possible_impact
        
        # Convert impact to similarity (invert)
        return max(0.0, 1.0 - normalized_impact)
    
    def _generate_store_recommendations(self, differences: List[ContentDifference], performance_analysis: Dict) -> List[str]:
        """Generate recommendations for store operation differences."""
        recommendations = []
        
        # Check for critical issues
        critical_diffs = [d for d in differences if d.severity == 'critical']
        if critical_diffs:
            recommendations.append("CRITICAL: Address operation success status differences before production deployment")
        
        # Performance recommendations
        if performance_analysis['significant_difference']:
            faster_system = performance_analysis['faster_system']
            ratio = performance_analysis['ratio']
            
            if faster_system == 'shadow' and ratio > 2.0:
                recommendations.append(f"Shadow system is {ratio:.2f}x faster - investigate optimization opportunities for primary system")
            elif faster_system == 'primary' and ratio < 0.5:
                recommendations.append(f"Primary system is {1/ratio:.2f}x faster - investigate performance issues in shadow system")
        
        # Error handling recommendations
        error_diffs = [d for d in differences if d.type == 'error']
        if error_diffs:
            recommendations.append("Review error handling consistency between systems")
        
        return recommendations
    
    def _generate_retrieve_recommendations(self, differences: List[ContentDifference], performance_analysis: Dict) -> List[str]:
        """Generate recommendations for retrieve operation differences."""
        recommendations = []
        
        # Check for critical issues
        critical_diffs = [d for d in differences if d.severity == 'critical']
        if critical_diffs:
            recommendations.append("CRITICAL: Address operation success status differences before production deployment")
        
        # Content quality recommendations
        content_diffs = [d for d in differences if d.type == 'content']
        if len(content_diffs) > 3:  # Arbitrary threshold
            recommendations.append("Multiple content differences detected - review semantic search calibration")
        
        # Structure recommendations
        structure_diffs = [d for d in differences if d.type == 'structure']
        if structure_diffs:
            recommendations.append("Result structure differences may indicate index inconsistency")
        
        # Performance recommendations
        if performance_analysis['significant_difference']:
            faster_system = performance_analysis['faster_system']
            ratio = performance_analysis['ratio']
            
            if faster_system == 'shadow' and ratio > 2.0:
                recommendations.append(f"Shadow system is {ratio:.2f}x faster - consider migrating to new system")
            elif faster_system == 'primary' and ratio < 0.5:
                recommendations.append(f"Primary system is {1/ratio:.2f}x faster - optimize shadow system before migration")
        
        return recommendations


class ComparisonLogger:
    """Handles logging and persistence of comparison results."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.ComparisonLogger")
        
        # Setup log directory
        self.log_dir = Path(self.config.get('logging', {}).get('reports_dir', 'knowledge/shadow-mode-reports'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def log_comparison(self, report: ComparisonReport):
        """Log a comparison report."""
        
        # Log to standard logger
        self._log_to_standard_logger(report)
        
        # Save detailed report to file
        self._save_detailed_report(report)
        
        # Update metrics database
        self._update_metrics_database(report)
    
    def _log_to_standard_logger(self, report: ComparisonReport):
        """Log summary to standard logger."""
        high_severity_count = len([d for d in report.differences if d.severity in ['high', 'critical']])
        
        log_msg = (
            f"Shadow comparison - {report.operation}: "
            f"similarity={report.overall_similarity:.2f}, "
            f"differences={len(report.differences)}, "
            f"high_severity={high_severity_count}"
        )
        
        if high_severity_count > 0:
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)
    
    def _save_detailed_report(self, report: ComparisonReport):
        """Save detailed report to JSON file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{report.operation}_{timestamp}.json"
        filepath = self.log_dir / filename
        
        # Convert to serializable format
        report_dict = asdict(report)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            self.logger.debug(f"Saved detailed comparison report: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save comparison report: {e}")
    
    def _update_metrics_database(self, report: ComparisonReport):
        """Update metrics database with report data."""
        metrics_file = self.log_dir / "metrics.jsonl"
        
        metric_entry = {
            "timestamp": report.timestamp,
            "operation": report.operation,
            "overall_similarity": report.overall_similarity,
            "differences_count": len(report.differences),
            "high_severity_count": len([d for d in report.differences if d.severity in ['high', 'critical']]),
            "performance_ratio": report.performance_analysis.get('ratio', 1.0) if report.performance_analysis else 1.0,
            "recommendations_count": len(report.recommendations)
        }
        
        try:
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metric_entry) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to update metrics database: {e}")


def create_comparison_framework(config: Dict[str, Any] = None) -> Tuple[AdvancedComparator, ComparisonLogger]:
    """Factory function to create comparison framework components."""
    comparator = AdvancedComparator(config)
    logger = ComparisonLogger(config)
    
    return comparator, logger


if __name__ == "__main__":
    # Test the comparison framework
    from shadow_mode import OperationResult
    
    comparator = AdvancedComparator()
    logger = ComparisonLogger()
    
    # Test store comparison
    primary = OperationResult(success=True, data="doc123", duration_ms=50.0)
    shadow = OperationResult(success=True, data="doc456", duration_ms=75.0)
    
    store_report = comparator.compare_store_results(primary, shadow)
    print("Store Comparison Report:")
    print(f"Similarity: {store_report.overall_similarity:.2f}")
    print(f"Differences: {len(store_report.differences)}")
    print(f"Recommendations: {len(store_report.recommendations)}")
    
    logger.log_comparison(store_report)
    
    print("\nComparison framework test completed.")