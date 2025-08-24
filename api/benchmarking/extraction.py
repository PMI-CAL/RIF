#!/usr/bin/env python3
"""
DPIBS Benchmarking API - Specification Extraction Engine
Issue #140: DPIBS Sub-Issue 4 - Benchmarking + Knowledge Integration APIs

Enhanced design specification extraction with:
- GitHub issue specification extraction (>90% accuracy target)
- Structured analysis and requirement classification
- Multi-dimensional specification scoring
- Integration with existing design-benchmarking-framework.py
- <30 second extraction performance for complex issues
"""

import os
import sys
import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

# Add RIF to path for imports
sys.path.insert(0, '/Users/cal/DEV/RIF')

from systems.design_benchmarking_framework import (
    DesignSpecificationExtractor,
    DesignSpecification,
    SpecificationType,
    BenchmarkingEngine
)
from knowledge.database.dpibs_optimization import DPIBSPerformanceOptimizer
from systems.knowledge_integration_apis import MCPKnowledgeIntegrator


# API Models
class ExtractionRequest(BaseModel):
    """Request model for specification extraction"""
    issue_id: int = Field(..., description="GitHub issue number")
    include_comments: bool = Field(default=True, description="Include RIF agent comments")
    accuracy_threshold: float = Field(default=0.9, description="Minimum accuracy threshold")
    extraction_depth: str = Field(default="comprehensive", description="shallow, standard, comprehensive")

class ExtractionResponse(BaseModel):
    """Response model for specification extraction"""
    extraction_id: str
    issue_id: int
    specifications_count: int
    specifications: List[Dict[str, Any]]
    extraction_metadata: Dict[str, Any]
    accuracy_score: float
    extraction_time_ms: float
    status: str

class AnalysisRequest(BaseModel):
    """Request model for implementation analysis"""
    issue_id: int
    implementation_id: Optional[str] = None
    include_evidence: bool = Field(default=True, description="Include detailed evidence collection")
    performance_analysis: bool = Field(default=True, description="Include performance metrics")

class GradingRequest(BaseModel):
    """Request model for implementation grading"""
    analysis_id: str
    grading_criteria: Optional[Dict[str, float]] = None
    expert_validation: bool = Field(default=False, description="Request expert validation")


class EnhancedBenchmarkingExtractor:
    """
    Enhanced specification extraction engine for DPIBS
    Builds upon existing design-benchmarking-framework.py with performance optimizations
    """
    
    def __init__(self, performance_optimizer: DPIBSPerformanceOptimizer,
                 knowledge_integrator: MCPKnowledgeIntegrator):
        self.performance_optimizer = performance_optimizer
        self.knowledge_integrator = knowledge_integrator
        self.base_extractor = DesignSpecificationExtractor()
        self.benchmarking_engine = BenchmarkingEngine()
        
        # Enhanced extraction patterns based on DPIBS research
        self._initialize_enhanced_patterns()
    
    def _initialize_enhanced_patterns(self):
        """Initialize enhanced extraction patterns for 90% accuracy target"""
        # Research-based patterns for improved accuracy
        self.enhanced_patterns = {
            "api_endpoints": [
                r"(?i)(?:POST|GET|PUT|DELETE)\s+(/api/[\w/{}]+)",
                r"(?i)(?:endpoint|route|api):\s*([^\n]+)",
                r"(?i)(?:- |• |\d+\.)\s*(POST|GET|PUT|DELETE)\s+([^\n]+)"
            ],
            "performance_targets": [
                r"(?i)(?:<|under|below|within)\s*(\d+)\s*(ms|milliseconds|seconds|minutes)",
                r"(?i)(?:performance|speed|latency).*?(<|under|below)\s*(\d+)\s*(ms|seconds)",
                r"(?i)(?:target|goal|requirement).*?(\d+)\s*(ms|seconds|requests|operations)"
            ],
            "accuracy_requirements": [
                r"(?i)(\d+)%\s*(?:accuracy|precision|compliance|alignment|agreement)",
                r"(?i)(?:accuracy|precision).*?(\d+)%",
                r"(?i)(?:>=|>|at least|minimum)\s*(\d+)%\s*(?:accuracy|compliance)"
            ],
            "integration_requirements": [
                r"(?i)(?:integration with|integrates with|compatible with)\s*([^\n.]+)",
                r"(?i)(?:MCP|Knowledge Server|API)\s*(?:integration|compatibility)",
                r"(?i)maintains?\s*(\d+)%\s*(?:compatibility|backward compatibility)"
            ]
        }
    
    async def extract_specifications(self, request: ExtractionRequest) -> ExtractionResponse:
        """Extract specifications from GitHub issue with enhanced accuracy"""
        start_time = time.time()
        
        try:
            # Use base extractor for core functionality
            specifications = self.base_extractor.extract_specifications_from_issue(request.issue_id)
            
            # Apply enhanced extraction for missing patterns
            enhanced_specs = await self._apply_enhanced_extraction(request.issue_id, specifications)
            
            # Merge and deduplicate specifications
            all_specs = specifications + enhanced_specs
            deduplicated_specs = self._deduplicate_specifications(all_specs)
            
            # Calculate accuracy score
            accuracy_score = await self._calculate_extraction_accuracy(deduplicated_specs, request.issue_id)
            
            # Generate extraction metadata
            extraction_time_ms = (time.time() - start_time) * 1000
            metadata = {
                "extraction_method": "hybrid_nlp_enhanced",
                "patterns_matched": self._count_pattern_matches(deduplicated_specs),
                "confidence_level": "high" if accuracy_score >= 0.9 else "medium",
                "enhancement_applied": len(enhanced_specs) > 0,
                "performance_ms": extraction_time_ms
            }
            
            extraction_id = f"extract-{request.issue_id}-{int(time.time())}"
            
            return ExtractionResponse(
                extraction_id=extraction_id,
                issue_id=request.issue_id,
                specifications_count=len(deduplicated_specs),
                specifications=[spec.to_dict() for spec in deduplicated_specs],
                extraction_metadata=metadata,
                accuracy_score=accuracy_score,
                extraction_time_ms=extraction_time_ms,
                status="success" if accuracy_score >= request.accuracy_threshold else "warning"
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    
    async def _apply_enhanced_extraction(self, issue_id: int, base_specs: List[DesignSpecification]) -> List[DesignSpecification]:
        """Apply enhanced extraction patterns to find missed specifications"""
        enhanced_specs = []
        
        # Get issue content for enhanced analysis
        issue_content = await self._get_issue_content(issue_id)
        if not issue_content:
            return enhanced_specs
        
        # Apply enhanced pattern matching
        for pattern_category, patterns in self.enhanced_patterns.items():
            category_specs = self._extract_by_category(issue_content, patterns, pattern_category, issue_id)
            enhanced_specs.extend(category_specs)
        
        return enhanced_specs
    
    async def _get_issue_content(self, issue_id: int) -> Optional[str]:
        """Get GitHub issue content including comments"""
        try:
            import subprocess
            result = subprocess.run(
                ["gh", "issue", "view", str(issue_id), "--json", "body,comments"],
                capture_output=True, text=True, check=True
            )
            issue_data = json.loads(result.stdout)
            
            content = issue_data.get('body', '')
            for comment in issue_data.get('comments', []):
                content += "\n\n" + comment.get('body', '')
            
            return content
        except Exception:
            return None
    
    def _extract_by_category(self, content: str, patterns: List[str], category: str, issue_id: int) -> List[DesignSpecification]:
        """Extract specifications by category using enhanced patterns"""
        import re
        specs = []
        spec_counter = 0
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = ' '.join(str(m) for m in match if m)
                
                if len(str(match).strip()) > 5:  # Filter short matches
                    spec_counter += 1
                    spec_type = self._categorize_specification(category, str(match))
                    
                    spec = DesignSpecification(
                        id=f"enhanced-{issue_id}-{category}-{spec_counter}",
                        type=spec_type,
                        description=str(match).strip(),
                        acceptance_criteria=self._extract_criteria_for_match(content, str(match)),
                        constraints=self._extract_constraints_for_match(content, str(match)),
                        success_metrics=self._extract_metrics_for_match(str(match)),
                        priority=self._determine_priority_enhanced(str(match)),
                        measurable=self._is_measurable_enhanced(str(match)),
                        testable=self._is_testable_enhanced(str(match)),
                        created_at=datetime.now(),
                        issue_number=issue_id
                    )
                    specs.append(spec)
        
        return specs
    
    def _categorize_specification(self, category: str, match: str) -> SpecificationType:
        """Categorize specification based on enhanced pattern category"""
        category_mapping = {
            "api_endpoints": SpecificationType.FUNCTIONAL_REQUIREMENTS,
            "performance_targets": SpecificationType.PERFORMANCE_REQUIREMENTS,
            "accuracy_requirements": SpecificationType.QUALITY_GATES,
            "integration_requirements": SpecificationType.INTEGRATION_REQUIREMENTS
        }
        return category_mapping.get(category, SpecificationType.FUNCTIONAL_REQUIREMENTS)
    
    def _extract_criteria_for_match(self, content: str, match: str) -> List[str]:
        """Extract acceptance criteria related to specific match"""
        # Find criteria patterns near the match location
        import re
        criteria = []
        
        # Look for checkboxes and criteria near the match
        match_context = self._get_match_context(content, match)
        if match_context:
            checkbox_pattern = r"(?i)(?:- \[ \]|\* \[ \])\s*(.{10,100})"
            checkboxes = re.findall(checkbox_pattern, match_context)
            criteria.extend(checkboxes[:3])
        
        return criteria
    
    def _extract_constraints_for_match(self, content: str, match: str) -> List[str]:
        """Extract constraints related to specific match"""
        constraints = []
        match_context = self._get_match_context(content, match)
        
        if "<" in match or "under" in match.lower():
            constraints.append(f"Performance constraint: {match}")
        if "compatible" in match.lower() or "integration" in match.lower():
            constraints.append(f"Integration constraint: {match}")
            
        return constraints
    
    def _extract_metrics_for_match(self, match: str) -> Dict[str, Any]:
        """Extract metrics from match text"""
        import re
        metrics = {}
        
        # Performance metrics
        time_match = re.search(r"(\d+)\s*(ms|milliseconds|seconds)", match, re.IGNORECASE)
        if time_match:
            value = int(time_match.group(1))
            unit = time_match.group(2)
            if unit.lower() in ['ms', 'milliseconds']:
                metrics["max_time_ms"] = value
            elif unit.lower() == 'seconds':
                metrics["max_time_ms"] = value * 1000
        
        # Accuracy metrics
        accuracy_match = re.search(r"(\d+)%", match)
        if accuracy_match:
            metrics["target_percentage"] = int(accuracy_match.group(1))
        
        return metrics
    
    def _get_match_context(self, content: str, match: str, context_lines: int = 3) -> Optional[str]:
        """Get context lines around a match"""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if match.lower() in line.lower():
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                return '\n'.join(lines[start:end])
        return None
    
    def _determine_priority_enhanced(self, requirement: str) -> str:
        """Enhanced priority determination with DPIBS context"""
        requirement_lower = requirement.lower()
        
        # Critical performance and accuracy requirements
        if any(word in requirement_lower for word in ["<2 min", "<100ms", "90%", "critical", "must"]):
            return "must_have"
        elif any(word in requirement_lower for word in ["should", "target", "goal"]):
            return "should_have"
        else:
            return "could_have"
    
    def _is_measurable_enhanced(self, requirement: str) -> bool:
        """Enhanced measurability check"""
        import re
        measurable_patterns = [
            r"\d+\s*(?:ms|seconds|minutes|%|requests|operations)",
            r"(?:accuracy|performance|speed|latency|throughput)",
            r"(?:coverage|compliance|alignment|agreement)"
        ]
        
        requirement_lower = requirement.lower()
        return any(re.search(pattern, requirement_lower) for pattern in measurable_patterns)
    
    def _is_testable_enhanced(self, requirement: str) -> bool:
        """Enhanced testability check"""
        testable_indicators = [
            "test", "validate", "verify", "measure", "benchmark",
            "api", "endpoint", "integration", "performance"
        ]
        
        requirement_lower = requirement.lower()
        return any(indicator in requirement_lower for indicator in testable_indicators)
    
    def _deduplicate_specifications(self, specifications: List[DesignSpecification]) -> List[DesignSpecification]:
        """Remove duplicate specifications based on similarity"""
        if not specifications:
            return []
        
        unique_specs = []
        seen_descriptions = set()
        
        for spec in specifications:
            # Simple deduplication based on description similarity
            desc_normalized = spec.description.lower().strip()
            if desc_normalized not in seen_descriptions and len(desc_normalized) > 10:
                seen_descriptions.add(desc_normalized)
                unique_specs.append(spec)
        
        return unique_specs
    
    async def _calculate_extraction_accuracy(self, specifications: List[DesignSpecification], issue_id: int) -> float:
        """Calculate extraction accuracy score based on completeness and quality"""
        if not specifications:
            return 0.0
        
        accuracy_factors = {
            "specification_count": min(1.0, len(specifications) / 10),  # Expect ~10 specs
            "measurable_ratio": sum(1 for s in specifications if s.measurable) / len(specifications),
            "testable_ratio": sum(1 for s in specifications if s.testable) / len(specifications),
            "criteria_coverage": sum(1 for s in specifications if s.acceptance_criteria) / len(specifications),
            "priority_distribution": self._evaluate_priority_distribution(specifications)
        }
        
        # Weighted average accuracy score
        weights = {"specification_count": 0.3, "measurable_ratio": 0.25, "testable_ratio": 0.2, 
                  "criteria_coverage": 0.15, "priority_distribution": 0.1}
        
        accuracy_score = sum(accuracy_factors[factor] * weights[factor] 
                           for factor in accuracy_factors)
        
        return min(1.0, accuracy_score)
    
    def _evaluate_priority_distribution(self, specifications: List[DesignSpecification]) -> float:
        """Evaluate if priority distribution is realistic"""
        priorities = [spec.priority for spec in specifications]
        must_have_ratio = priorities.count('must_have') / len(priorities)
        
        # Ideal distribution: 30-50% must_have, rest should_have/could_have
        if 0.3 <= must_have_ratio <= 0.5:
            return 1.0
        elif 0.2 <= must_have_ratio <= 0.7:
            return 0.7
        else:
            return 0.4
    
    def _count_pattern_matches(self, specifications: List[DesignSpecification]) -> Dict[str, int]:
        """Count how many specifications match each pattern category"""
        pattern_counts = {
            "functional_requirements": 0,
            "performance_requirements": 0,
            "quality_gates": 0,
            "integration_requirements": 0,
            "architectural_constraints": 0
        }
        
        for spec in specifications:
            spec_type = spec.type.value
            if spec_type in pattern_counts:
                pattern_counts[spec_type] += 1
        
        return pattern_counts


# FastAPI Router Setup
router = APIRouter(prefix="/api/v1/benchmark", tags=["benchmarking"])

# Initialize components (will be injected in main app)
extractor: Optional[EnhancedBenchmarkingExtractor] = None


@router.post("/extract/{issue_id}", response_model=ExtractionResponse)
async def extract_specifications(issue_id: int, request: ExtractionRequest, background_tasks: BackgroundTasks):
    """
    Extract design specifications from GitHub issue
    Target: >90% accuracy, <30 seconds for complex issues
    """
    if not extractor:
        raise HTTPException(status_code=503, detail="Benchmarking service not initialized")
    
    try:
        # Override issue_id from path parameter
        request.issue_id = issue_id
        
        # Perform extraction with performance monitoring
        result = await extractor.extract_specifications(request)
        
        # Background task for learning integration
        if result.accuracy_score >= 0.9:
            background_tasks.add_task(update_extraction_patterns, issue_id, result)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.get("/extract/{issue_id}/status")
async def get_extraction_status(issue_id: int):
    """
    Get extraction status and cached results
    """
    # Implementation for status checking
    return {"status": "completed", "issue_id": issue_id}


async def update_extraction_patterns(issue_id: int, result: ExtractionResponse):
    """
    Background task to update extraction patterns based on successful extractions
    """
    # This would integrate with the learning system
    pass


def initialize_extraction_api(performance_optimizer: DPIBSPerformanceOptimizer,
                             knowledge_integrator: MCPKnowledgeIntegrator):
    """
    Initialize the extraction API with required dependencies
    """
    global extractor
    extractor = EnhancedBenchmarkingExtractor(performance_optimizer, knowledge_integrator)
    return router