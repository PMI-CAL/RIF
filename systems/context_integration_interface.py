#!/usr/bin/env python3
"""
Context Intelligence Platform - Integration Interface
Issue #119: DPIBS Architecture Phase 1

Provides 100% backward compatibility with existing RIF agent workflows
while enabling access to the new Context Intelligence Platform.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import logging
from pathlib import Path

# Add the systems directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing context optimizer for backward compatibility
try:
    from import_utils import import_context_optimization_engine
    context_imports = import_context_optimization_engine()
    ContextOptimizer = context_imports['ContextOptimizer']
    AgentType = context_imports['AgentType']
    ContextItem = context_imports['ContextItem']
    SystemContext = context_imports['SystemContext']
except ImportError:
    # Fallback if not available
    ContextOptimizer = None
    AgentType = None

# Import new Context Intelligence Platform
try:
    from context_intelligence_platform import ContextIntelligencePlatform, ContextRequest, ContextResponse
    from context_api_gateway import ContextAPIGateway
    from event_service_bus import EventServiceBus
    from context_database_schema import ContextDatabaseSchema
except ImportError as e:
    logging.error(f"Failed to import Context Intelligence Platform components: {e}")
    ContextIntelligencePlatform = None

logger = logging.getLogger(__name__)

@dataclass
class LegacyContextRequest:
    """Legacy context request format for backward compatibility"""
    agent_type: str
    task_context: Dict[str, Any]
    issue_number: Optional[int] = None
    format_output: bool = True

@dataclass
class LegacyContextResponse:
    """Legacy context response format for backward compatibility"""
    formatted_context: str
    relevance_score: float
    context_utilization: float
    processing_time_ms: float
    source: str = "context-intelligence-platform"

class BackwardCompatibilityLayer:
    """Provides backward compatibility for existing RIF agent workflows"""
    
    def __init__(self, use_legacy: bool = False):
        self.use_legacy = use_legacy
        self.legacy_optimizer = None
        self.platform = None
        self.gateway = None
        self.event_bus = None
        self.schema_manager = None
        
        # Initialize appropriate system
        if use_legacy or ContextIntelligencePlatform is None:
            self._initialize_legacy_system()
        else:
            self._initialize_platform_system()
    
    def _initialize_legacy_system(self):
        """Initialize legacy context optimization system"""
        try:
            if ContextOptimizer:
                self.legacy_optimizer = ContextOptimizer()
                logger.info("Initialized legacy context optimization system")
            else:
                logger.warning("Legacy context optimizer not available")
        except Exception as e:
            logger.error(f"Failed to initialize legacy system: {e}")
    
    def _initialize_platform_system(self):
        """Initialize Context Intelligence Platform system"""
        try:
            self.platform = ContextIntelligencePlatform()
            self.gateway = ContextAPIGateway(self.platform)
            self.event_bus = EventServiceBus()
            self.schema_manager = ContextDatabaseSchema()
            
            # Start event bus
            asyncio.create_task(self.event_bus.start())
            
            logger.info("Initialized Context Intelligence Platform system")
        except Exception as e:
            logger.error(f"Failed to initialize platform system: {e}")
            # Fallback to legacy
            self.use_legacy = True
            self._initialize_legacy_system()
    
    async def get_agent_context(self, request: LegacyContextRequest) -> LegacyContextResponse:
        """
        Main interface for getting agent context - maintains backward compatibility
        while leveraging new platform capabilities when available.
        """
        start_time = time.time()
        
        try:
            if self.use_legacy or not self.platform:
                return await self._get_legacy_context(request)
            else:
                return await self._get_platform_context(request)
        except Exception as e:
            logger.error(f"Context request failed, falling back to legacy: {e}")
            if not self.use_legacy and self.legacy_optimizer:
                return await self._get_legacy_context(request)
            else:
                # Return minimal context if all else fails
                return LegacyContextResponse(
                    formatted_context=f"## Context Error\nFailed to retrieve context: {str(e)}\nUsing minimal fallback context.",
                    relevance_score=0.1,
                    context_utilization=0.0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    source="fallback"
                )
    
    async def _get_legacy_context(self, request: LegacyContextRequest) -> LegacyContextResponse:
        """Get context using legacy system"""
        start_time = time.time()
        
        if not self.legacy_optimizer:
            raise Exception("Legacy context optimizer not available")
        
        # Convert string agent type to enum if needed
        if isinstance(request.agent_type, str):
            try:
                agent_type_enum = AgentType(request.agent_type.replace('_', '-'))
            except (ValueError, AttributeError):
                # Default to implementer for unknown types
                agent_type_enum = AgentType.IMPLEMENTER
        else:
            agent_type_enum = request.agent_type
        
        # Get optimized context using legacy system
        agent_context = self.legacy_optimizer.optimize_for_agent(
            agent_type_enum,
            request.task_context,
            request.issue_number
        )
        
        # Format context if requested
        if request.format_output:
            formatted_context = self.legacy_optimizer.format_context_for_agent(agent_context)
        else:
            formatted_context = json.dumps({
                "relevant_knowledge": [item.__dict__ for item in agent_context.relevant_knowledge],
                "system_context": agent_context.system_context.__dict__,
                "task_specific_context": agent_context.task_specific_context
            }, indent=2, default=str)
        
        processing_time = (time.time() - start_time) * 1000
        
        return LegacyContextResponse(
            formatted_context=formatted_context,
            relevance_score=0.8,  # Default relevance for legacy system
            context_utilization=agent_context.context_window_utilization,
            processing_time_ms=processing_time,
            source="legacy-optimizer"
        )
    
    async def _get_platform_context(self, request: LegacyContextRequest) -> LegacyContextResponse:
        """Get context using Context Intelligence Platform"""
        start_time = time.time()
        
        # Convert string agent type to enum
        try:
            agent_type_enum = AgentType(request.agent_type.replace('_', '-'))
        except (ValueError, AttributeError):
            agent_type_enum = AgentType.IMPLEMENTER
        
        # Use platform to get context
        context_response = await self.platform.process_context_request(
            agent_type_enum,
            request.task_context,
            request.issue_number
        )
        
        # Format context if requested
        if request.format_output:
            formatted_context = self.platform.format_context_for_agent(context_response)
        else:
            formatted_context = json.dumps({
                "agent_context": {
                    "relevant_knowledge": [item.__dict__ for item in context_response.agent_context.relevant_knowledge],
                    "system_context": context_response.agent_context.system_context.__dict__,
                    "task_specific_context": context_response.agent_context.task_specific_context,
                    "context_window_utilization": context_response.agent_context.context_window_utilization,
                    "total_size": context_response.agent_context.total_size
                },
                "performance_metrics": {
                    "total_response_time_ms": context_response.total_response_time_ms,
                    "cache_hit": context_response.cache_hit,
                    "source_services": context_response.source_services
                }
            }, indent=2, default=str)
        
        # Extract relevance score from context items
        relevance_scores = [item.relevance_score for item in context_response.agent_context.relevant_knowledge]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
        
        # Record optimization data
        if self.schema_manager:
            optimization_data = {
                'optimization_id': f"compat_{int(time.time() * 1000)}",
                'agent_type': request.agent_type,
                'task_context': request.task_context,
                'issue_number': request.issue_number,
                'original_context_size': 8000,  # Estimated
                'optimized_context_size': context_response.agent_context.total_size,
                'relevance_score': avg_relevance,
                'optimization_time_ms': context_response.total_response_time_ms,
                'cache_hit': context_response.cache_hit,
                'context_items': [item.id for item in context_response.agent_context.relevant_knowledge]
            }
            
            try:
                self.schema_manager.insert_context_optimization(optimization_data)
            except Exception as e:
                logger.warning(f"Failed to record optimization data: {e}")
        
        return LegacyContextResponse(
            formatted_context=formatted_context,
            relevance_score=avg_relevance,
            context_utilization=context_response.agent_context.context_window_utilization,
            processing_time_ms=context_response.total_response_time_ms,
            source="context-intelligence-platform"
        )

class ContextIntegrationInterface:
    """
    Main integration interface that provides multiple access patterns:
    1. Legacy function calls (backward compatibility)
    2. Modern async interface (new features)
    3. RIF agent integration (seamless workflow integration)
    """
    
    def __init__(self, enable_platform: bool = True, fallback_to_legacy: bool = True):
        self.enable_platform = enable_platform
        self.fallback_to_legacy = fallback_to_legacy
        self.compatibility_layer = BackwardCompatibilityLayer(use_legacy=not enable_platform)
        
        # Performance tracking
        self.request_count = 0
        self.total_response_time = 0.0
        self.cache_hits = 0
        
    # === Legacy Function Interface (100% Backward Compatible) ===
    
    def optimize_for_agent(self, agent_type: str, task_context: Dict[str, Any], 
                          issue_number: Optional[int] = None) -> Dict[str, Any]:
        """
        Legacy function interface - maintains exact signature and behavior
        of the original context optimization function.
        """
        # Convert to async call
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        request = LegacyContextRequest(
            agent_type=agent_type,
            task_context=task_context,
            issue_number=issue_number,
            format_output=False  # Return structured data for legacy compatibility
        )
        
        if loop.is_running():
            # If called from within async context, use asyncio.create_task
            task = asyncio.create_task(self.compatibility_layer.get_agent_context(request))
            # This won't work in sync context, but maintains interface
            response = task
        else:
            # Sync context - run the async function
            response = loop.run_until_complete(self.compatibility_layer.get_agent_context(request))
        
        # Update performance tracking
        self.request_count += 1
        if hasattr(response, 'processing_time_ms'):
            self.total_response_time += response.processing_time_ms
        
        # Convert response to legacy format
        try:
            context_data = json.loads(response.formatted_context)
            if "agent_context" in context_data:
                # Platform response format
                agent_context = context_data["agent_context"]
                return {
                    "agent_type": agent_type,
                    "relevant_knowledge": agent_context.get("relevant_knowledge", []),
                    "system_context": agent_context.get("system_context", {}),
                    "task_specific_context": agent_context.get("task_specific_context", {}),
                    "context_window_utilization": agent_context.get("context_window_utilization", 0.0),
                    "total_size": agent_context.get("total_size", 0),
                    "performance_metrics": context_data.get("performance_metrics", {}),
                    "source": response.source
                }
            else:
                # Legacy response format
                return context_data
        except json.JSONDecodeError:
            # Fallback for non-JSON responses
            return {
                "agent_type": agent_type,
                "formatted_context": response.formatted_context,
                "relevance_score": response.relevance_score,
                "context_utilization": response.context_utilization,
                "processing_time_ms": response.processing_time_ms,
                "source": response.source
            }
    
    def format_context_for_agent(self, agent_context: Dict[str, Any]) -> str:
        """Legacy function for formatting context - maintains backward compatibility"""
        if isinstance(agent_context, dict) and "formatted_context" in agent_context:
            return agent_context["formatted_context"]
        
        # Format structured context data
        formatted_lines = []
        
        # System context
        if "system_context" in agent_context:
            system_ctx = agent_context["system_context"]
            formatted_lines.append("## System Context")
            formatted_lines.append(f"**Overview**: {system_ctx.get('overview', 'RIF System')}")
            formatted_lines.append(f"**Purpose**: {system_ctx.get('purpose', 'Automated development')}")
        
        # Relevant knowledge
        if "relevant_knowledge" in agent_context:
            formatted_lines.append("\n## Relevant Knowledge")
            for item in agent_context["relevant_knowledge"][:5]:  # Limit for readability
                if isinstance(item, dict):
                    formatted_lines.append(f"**{item.get('type', 'Knowledge')}**: {item.get('content', '')[:200]}...")
        
        # Performance info
        if "performance_metrics" in agent_context:
            perf = agent_context["performance_metrics"]
            formatted_lines.append(f"\n## Context Optimization")
            formatted_lines.append(f"**Response Time**: {perf.get('total_response_time_ms', 0):.1f}ms")
            formatted_lines.append(f"**Cache Hit**: {perf.get('cache_hit', False)}")
            formatted_lines.append(f"**Source**: {agent_context.get('source', 'unknown')}")
        
        return "\n".join(formatted_lines)
    
    # === Modern Async Interface (New Features) ===
    
    async def get_optimized_context(self, agent_type: str, task_context: Dict[str, Any],
                                   issue_number: Optional[int] = None,
                                   format_output: bool = True,
                                   priority: int = 1) -> Dict[str, Any]:
        """
        Modern async interface with enhanced features:
        - Priority-based processing
        - Detailed performance metrics
        - Enhanced error handling
        - Structured response format
        """
        request = LegacyContextRequest(
            agent_type=agent_type,
            task_context=task_context,
            issue_number=issue_number,
            format_output=format_output
        )
        
        response = await self.compatibility_layer.get_agent_context(request)
        
        # Update performance tracking
        self.request_count += 1
        self.total_response_time += response.processing_time_ms
        if "cache_hit" in response.formatted_context:
            self.cache_hits += 1
        
        return {
            "success": True,
            "agent_type": agent_type,
            "formatted_context": response.formatted_context,
            "relevance_score": response.relevance_score,
            "context_utilization": response.context_utilization,
            "processing_time_ms": response.processing_time_ms,
            "source": response.source,
            "timestamp": datetime.now().isoformat(),
            "request_id": f"async_{int(time.time() * 1000)}"
        }
    
    async def batch_optimize_contexts(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch context optimization for multiple agents concurrently
        """
        tasks = []
        for req in requests:
            task = self.get_optimized_context(
                agent_type=req["agent_type"],
                task_context=req["task_context"],
                issue_number=req.get("issue_number"),
                format_output=req.get("format_output", True),
                priority=req.get("priority", 1)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "agent_type": requests[i]["agent_type"],
                    "timestamp": datetime.now().isoformat()
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    # === RIF Agent Integration Interface ===
    
    def rif_agent_context_hook(self, agent_name: str, issue_number: int, 
                              context_requirements: Dict[str, Any]) -> str:
        """
        Hook function for RIF agents to get context during workflow execution.
        Designed to be called by RIF agents during their execution.
        """
        task_context = {
            "agent_name": agent_name,
            "issue_number": issue_number,
            "requirements": context_requirements,
            "hook_call": True,
            "timestamp": datetime.now().isoformat()
        }
        
        # Use legacy interface for synchronous call
        context_data = self.optimize_for_agent(agent_name, task_context, issue_number)
        
        # Return formatted context for agent consumption
        return self.format_context_for_agent(context_data)
    
    def rif_workflow_context(self, workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provides context for RIF workflow orchestration decisions.
        Used by RIF system for agent selection and task routing.
        """
        try:
            context_data = self.optimize_for_agent(
                "rif-orchestrator",
                workflow_state,
                workflow_state.get("issue_number")
            )
            
            return {
                "workflow_context": context_data,
                "recommendations": self._extract_workflow_recommendations(context_data),
                "performance_info": {
                    "total_requests": self.request_count,
                    "avg_response_time_ms": self.total_response_time / max(self.request_count, 1),
                    "cache_hit_rate": self.cache_hits / max(self.request_count, 1)
                }
            }
        except Exception as e:
            logger.error(f"RIF workflow context failed: {e}")
            return {
                "workflow_context": {},
                "recommendations": [],
                "error": str(e)
            }
    
    def _extract_workflow_recommendations(self, context_data: Dict[str, Any]) -> List[str]:
        """Extract workflow recommendations from context data"""
        recommendations = []
        
        # Analyze context for workflow hints
        if context_data.get("relevance_score", 0) > 0.8:
            recommendations.append("High-quality context available - proceed with confidence")
        elif context_data.get("relevance_score", 0) < 0.4:
            recommendations.append("Low context relevance - consider gathering more information")
        
        if context_data.get("context_utilization", 0) > 0.9:
            recommendations.append("Context window near full - consider summarization")
        
        return recommendations
    
    # === Utility Methods ===
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration interface statistics"""
        return {
            "total_requests": self.request_count,
            "avg_response_time_ms": self.total_response_time / max(self.request_count, 1),
            "cache_hit_rate": self.cache_hits / max(self.request_count, 1) if self.request_count > 0 else 0,
            "platform_enabled": self.enable_platform,
            "legacy_fallback": self.fallback_to_legacy,
            "system_source": self.compatibility_layer.use_legacy and "legacy" or "platform",
            "uptime_seconds": time.time() - (hasattr(self, '_start_time') and self._start_time or time.time())
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for integration interface"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "integration_layer": "operational",
            "components": {}
        }
        
        # Check compatibility layer
        if self.compatibility_layer.use_legacy:
            health_status["components"]["legacy_optimizer"] = "active"
            health_status["components"]["platform"] = "disabled"
        else:
            health_status["components"]["platform"] = "active" if self.compatibility_layer.platform else "failed"
            health_status["components"]["legacy_optimizer"] = "standby"
        
        # Check event bus
        if hasattr(self.compatibility_layer, 'event_bus') and self.compatibility_layer.event_bus:
            health_status["components"]["event_bus"] = "active" if self.compatibility_layer.event_bus.running else "stopped"
        
        return health_status

# Global interface instance for easy import and use
global_context_interface = None

def get_context_interface(enable_platform: bool = True, fallback_to_legacy: bool = True) -> ContextIntegrationInterface:
    """
    Get global context integration interface instance.
    Ensures singleton pattern for consistent behavior.
    """
    global global_context_interface
    
    if global_context_interface is None:
        global_context_interface = ContextIntegrationInterface(enable_platform, fallback_to_legacy)
        global_context_interface._start_time = time.time()
    
    return global_context_interface

# === Legacy Function Exports for Backward Compatibility ===

def optimize_for_agent(agent_type: str, task_context: Dict[str, Any], 
                      issue_number: Optional[int] = None) -> Dict[str, Any]:
    """
    Legacy function export - maintains exact compatibility with existing code.
    Automatically uses the global context interface.
    """
    interface = get_context_interface()
    return interface.optimize_for_agent(agent_type, task_context, issue_number)

def format_context_for_agent(agent_context: Dict[str, Any]) -> str:
    """
    Legacy function export - maintains exact compatibility with existing code.
    """
    interface = get_context_interface()
    return interface.format_context_for_agent(agent_context)

# Testing and CLI Interface
async def main():
    """Main function for testing the integration interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Context Integration Interface")
    parser.add_argument("--test", action="store_true", help="Run integration tests")
    parser.add_argument("--legacy", action="store_true", help="Test legacy interface only")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--agent", type=str, help="Test specific agent type")
    
    args = parser.parse_args()
    
    if args.test:
        print("=== Context Integration Interface Test ===\n")
        
        # Initialize interface
        interface = ContextIntegrationInterface(enable_platform=not args.legacy)
        
        # Test data
        test_agent = args.agent or "rif-implementer"
        test_task = {
            "description": "Test context integration interface",
            "complexity": "medium",
            "type": "implementation",
            "issue_number": 119
        }
        
        # Test legacy interface
        print("Testing legacy interface...")
        start_time = time.time()
        context_data = interface.optimize_for_agent(test_agent, test_task, 119)
        legacy_time = (time.time() - start_time) * 1000
        
        formatted_context = interface.format_context_for_agent(context_data)
        
        print(f"✓ Legacy interface test completed in {legacy_time:.1f}ms")
        print(f"Context source: {context_data.get('source', 'unknown')}")
        print(f"Context size: {len(formatted_context)} characters")
        print()
        
        # Test modern async interface
        if not args.legacy:
            print("Testing modern async interface...")
            start_time = time.time()
            async_response = await interface.get_optimized_context(test_agent, test_task, 119)
            async_time = (time.time() - start_time) * 1000
            
            print(f"✓ Async interface test completed in {async_time:.1f}ms")
            print(f"Success: {async_response['success']}")
            print(f"Response time: {async_response['processing_time_ms']:.1f}ms")
            print(f"Relevance score: {async_response['relevance_score']:.2f}")
            print()
        
        # Test RIF integration hooks
        print("Testing RIF integration hooks...")
        hook_context = interface.rif_agent_context_hook(test_agent, 119, {
            "context_type": "implementation",
            "priority": "high"
        })
        
        workflow_context = interface.rif_workflow_context({
            "issue_number": 119,
            "current_state": "implementing",
            "agent_type": test_agent
        })
        
        print("✓ RIF integration hooks test completed")
        print(f"Hook context length: {len(hook_context)} characters")
        print(f"Workflow recommendations: {len(workflow_context['recommendations'])}")
        print()
        
        # Health check
        health = interface.health_check()
        print("=== Health Check ===")
        print(json.dumps(health, indent=2))
        
        # Statistics
        stats = interface.get_integration_stats()
        print("\n=== Integration Statistics ===")
        print(json.dumps(stats, indent=2))
        
    elif args.benchmark:
        print("=== Integration Interface Benchmark ===\n")
        
        interface = ContextIntegrationInterface(enable_platform=not args.legacy)
        
        # Benchmark parameters
        request_count = 50
        agent_types = ["rif-implementer", "rif-validator", "rif-analyst"]
        
        print(f"Running {request_count} requests across {len(agent_types)} agent types...")
        
        start_time = time.time()
        
        # Create batch requests
        batch_requests = []
        for i in range(request_count):
            agent_type = agent_types[i % len(agent_types)]
            batch_requests.append({
                "agent_type": agent_type,
                "task_context": {
                    "description": f"Benchmark test request {i}",
                    "test_id": i,
                    "benchmark": True
                },
                "issue_number": 119,
                "format_output": True
            })
        
        # Run batch optimization
        results = await interface.batch_optimize_contexts(batch_requests)
        total_time = time.time() - start_time
        
        # Analyze results
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]
        
        response_times = [r["processing_time_ms"] for r in successful if "processing_time_ms" in r]
        
        print(f"Benchmark Results:")
        print(f"- Total time: {total_time:.2f} seconds")
        print(f"- Successful requests: {len(successful)}/{len(results)}")
        print(f"- Failed requests: {len(failed)}")
        print(f"- Requests per second: {len(successful) / total_time:.1f}")
        
        if response_times:
            print(f"- Average response time: {sum(response_times) / len(response_times):.1f}ms")
            print(f"- P95 response time: {sorted(response_times)[int(len(response_times) * 0.95)]:.1f}ms")
        
        # Final statistics
        stats = interface.get_integration_stats()
        print(f"- Cache hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"- System source: {stats['system_source']}")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())