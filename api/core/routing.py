#!/usr/bin/env python3
"""
Core API Routing Infrastructure
Issue #137: DPIBS Sub-Issue 1 - Core API Framework + Context Optimization Engine

Provides RESTful API framework with standardized routing, middleware, and error handling.
Integrates with existing context optimization engine for <200ms agent context delivery.
"""

import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import asyncio
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import existing context optimization engine
import sys
sys.path.append('/Users/cal/DEV/RIF/systems')

# Import from context optimization engine (with hyphens in filename)
import importlib.util
spec = importlib.util.spec_from_file_location("context_optimization_engine", "/Users/cal/DEV/RIF/systems/context-optimization-engine.py")
context_optimization_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context_optimization_engine)
ContextOptimizer = context_optimization_engine.ContextOptimizer
AgentType = context_optimization_engine.AgentType
AgentContext = context_optimization_engine.AgentContext

# Import live system context engine (with hyphens in filename)
spec2 = importlib.util.spec_from_file_location("live_system_context_engine", "/Users/cal/DEV/RIF/systems/live-system-context-engine.py")
live_system_context_engine = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(live_system_context_engine)
LiveSystemContextEngine = live_system_context_engine.LiveSystemContextEngine

class APIFramework:
    """
    Core API Framework providing RESTful routing and middleware infrastructure.
    Integrates with existing RIF context optimization systems for agent-specific delivery.
    """
    
    def __init__(self, 
                 knowledge_base_path: str = "/Users/cal/DEV/RIF/knowledge",
                 performance_monitoring: bool = True):
        self.app = FastAPI(
            title="DPIBS API Framework",
            description="Core API Framework for Dynamic Performance Intelligence Benchmarking System",
            version="1.0.0",
            docs_url="/api/v1/docs",
            redoc_url="/api/v1/redoc"
        )
        
        self.knowledge_base_path = knowledge_base_path
        self.performance_monitoring = performance_monitoring
        
        # Initialize context optimization engine
        self.context_optimizer = ContextOptimizer(knowledge_base_path)
        self.live_context_engine = LiveSystemContextEngine()
        
        # Performance tracking
        self.request_metrics = []
        self.active_requests = {}
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        self._setup_error_handlers()
        
    def _setup_middleware(self):
        """Setup API middleware stack"""
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Compression middleware
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Performance monitoring middleware
        @self.app.middleware("http")
        async def performance_middleware(request: Request, call_next):
            """Monitor API performance and response times"""
            start_time = time.time()
            request_id = str(uuid.uuid4())
            
            # Track active request
            self.active_requests[request_id] = {
                "start_time": start_time,
                "path": request.url.path,
                "method": request.method
            }
            
            try:
                response = await call_next(request)
                
                # Calculate performance metrics
                duration_ms = (time.time() - start_time) * 1000
                
                if self.performance_monitoring:
                    self._record_performance_metric({
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "duration_ms": duration_ms,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Add performance headers
                response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
                response.headers["X-Request-ID"] = request_id
                
                return response
                
            finally:
                # Cleanup active request tracking
                self.active_requests.pop(request_id, None)
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Health check endpoint
        @self.app.get("/api/v1/health")
        async def health_check():
            """API health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "active_requests": len(self.active_requests)
            }
        
        # Context optimization endpoints
        @self.app.post("/api/v1/context/request")
        async def request_context(request_data: ContextRequest):
            """
            Request optimized context for specific agent type and task.
            Target: <200ms response time
            """
            try:
                start_time = time.time()
                
                # Convert request data to appropriate format
                agent_type = AgentType(request_data.agent_type)
                task_context = request_data.task_context
                issue_number = request_data.issue_number
                
                # Optimize context using existing engine
                agent_context = self.context_optimizer.optimize_for_agent(
                    agent_type, task_context, issue_number
                )
                
                # Format context for API response
                formatted_context = self.context_optimizer.format_context_for_agent(agent_context)
                
                optimization_time = (time.time() - start_time) * 1000
                
                response = {
                    "request_id": str(uuid.uuid4()),
                    "agent_type": agent_type.value,
                    "context": formatted_context,
                    "metadata": {
                        "total_items": len(agent_context.relevant_knowledge),
                        "context_size": agent_context.total_size,
                        "window_utilization": agent_context.context_window_utilization,
                        "optimization_time_ms": optimization_time
                    },
                    "performance": {
                        "response_time_ms": optimization_time,
                        "target_met": optimization_time < 200,
                        "cache_hit": optimization_time < 50  # Indicates likely cache hit
                    }
                }
                
                return response
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid agent type: {str(e)}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Context optimization failed: {str(e)}")
        
        # Agent-specific context endpoint
        @self.app.get("/api/v1/context/agent/{agent_id}")
        async def get_agent_context(agent_id: str, 
                                  issue_number: Optional[int] = None,
                                  context_type: Optional[str] = None):
            """
            Get optimized context for specific agent.
            Supports caching for improved performance.
            """
            try:
                # Map agent_id to AgentType
                agent_type = AgentType(agent_id.replace("-", "_").upper())
                
                task_context = {
                    "description": f"Context request for {agent_id}",
                    "type": context_type or "general"
                }
                
                agent_context = self.context_optimizer.optimize_for_agent(
                    agent_type, task_context, issue_number
                )
                
                return {
                    "agent_id": agent_id,
                    "context": self.context_optimizer.format_context_for_agent(agent_context),
                    "metadata": asdict(agent_context)
                }
                
            except ValueError:
                raise HTTPException(status_code=404, detail=f"Agent type '{agent_id}' not found")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get agent context: {str(e)}")
        
        # Context optimization endpoint
        @self.app.put("/api/v1/context/optimize/{session_id}")
        async def optimize_context(session_id: str, optimization_data: OptimizationRequest):
            """
            Optimize context for specific session with feedback.
            Enables continuous improvement of context delivery.
            """
            try:
                agent_type = AgentType(optimization_data.agent_type)
                
                # Apply feedback to optimization
                # This would integrate with learning system in production
                self.context_optimizer.update_knowledge_utilization_metrics(
                    agent_type=agent_type,
                    context_used=optimization_data.context_used,
                    decisions_made=optimization_data.decisions_made,
                    problems_found=optimization_data.problems_found
                )
                
                return {
                    "session_id": session_id,
                    "optimization_applied": True,
                    "feedback_recorded": True,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Context optimization failed: {str(e)}")
        
        # Context status endpoint
        @self.app.get("/api/v1/context/status/{request_id}")
        async def get_context_status(request_id: str):
            """
            Get status of context request.
            Supports async context generation for large requests.
            """
            # In production, this would check async job status
            return {
                "request_id": request_id,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
        
        # Performance metrics endpoint
        @self.app.get("/api/v1/metrics/performance")
        async def get_performance_metrics():
            """Get API performance metrics"""
            recent_metrics = self.request_metrics[-100:]  # Last 100 requests
            
            if not recent_metrics:
                return {"message": "No metrics available"}
            
            # Calculate statistics
            durations = [m["duration_ms"] for m in recent_metrics]
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)
            
            # Count by status
            status_counts = {}
            for metric in recent_metrics:
                status = metric["status_code"]
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                "total_requests": len(recent_metrics),
                "average_duration_ms": round(avg_duration, 2),
                "max_duration_ms": round(max_duration, 2),
                "min_duration_ms": round(min_duration, 2),
                "target_compliance": {
                    "under_200ms": sum(1 for d in durations if d < 200),
                    "compliance_rate": sum(1 for d in durations if d < 200) / len(durations)
                },
                "status_codes": status_counts,
                "active_requests": len(self.active_requests)
            }
        
        # System Context API endpoints (Issue #139)
        @self.app.get("/api/v1/system/components")
        async def get_system_components(force_rescan: bool = False, max_components: int = 200):
            """
            Get system component discovery with performance optimization.
            Issue #139: System Context + Understanding APIs
            Target: <500ms for complex system understanding queries
            """
            try:
                start_time = time.time()
                
                # Import system context API (lazy load for performance)
                sys.path.append('/Users/cal/DEV/RIF/systems')
                from system_context_apis import create_system_context_api
                from knowledge.database.dpibs_optimization import DPIBSPerformanceOptimizer
                
                optimizer = DPIBSPerformanceOptimizer()
                system_api = create_system_context_api(optimizer)
                
                result = system_api.discover_components(force_rescan, max_components)
                
                query_time = (time.time() - start_time) * 1000
                result["performance"] = {
                    "query_time_ms": query_time,
                    "target_met": query_time < 500,
                    "cache_used": result.get("cache_used", False)
                }
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"System component discovery failed: {str(e)}")
        
        @self.app.get("/api/v1/system/dependencies")
        async def get_system_dependencies():
            """
            Analyze system dependencies with real-time tracking.
            Issue #139: System Context + Understanding APIs
            """
            try:
                start_time = time.time()
                
                sys.path.append('/Users/cal/DEV/RIF/systems')
                from system_context_apis import create_system_context_api
                from knowledge.database.dpibs_optimization import DPIBSPerformanceOptimizer
                
                optimizer = DPIBSPerformanceOptimizer()
                system_api = create_system_context_api(optimizer)
                
                result = system_api.analyze_dependencies()
                
                query_time = (time.time() - start_time) * 1000
                result["performance"] = {
                    "query_time_ms": query_time,
                    "target_met": query_time < 500
                }
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Dependency analysis failed: {str(e)}")
        
        @self.app.get("/api/v1/system/big-picture")
        async def get_system_big_picture(max_components: int = 100, quick_analysis: bool = True):
            """
            Get comprehensive "big picture" system understanding.
            Issue #139: System Context + Understanding APIs
            Target: <500ms for complex system understanding queries
            """
            try:
                start_time = time.time()
                
                sys.path.append('/Users/cal/DEV/RIF/systems')
                from system_context_apis import create_system_context_api
                from knowledge.database.dpibs_optimization import DPIBSPerformanceOptimizer
                
                optimizer = DPIBSPerformanceOptimizer()
                system_api = create_system_context_api(optimizer)
                
                result = system_api.get_big_picture(max_components, quick_analysis)
                
                query_time = (time.time() - start_time) * 1000
                
                # Add performance validation
                result["performance_validation"] = {
                    "query_time_ms": query_time,
                    "target_met": query_time < 500,
                    "optimization_used": result["system_overview"].get("performance_optimized", False),
                    "quick_analysis_used": result["system_overview"].get("quick_analysis_used", False)
                }
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Big picture analysis failed: {str(e)}")
        
        @self.app.get("/api/v1/system/health")
        async def get_system_health():
            """Get system context engine health status"""
            try:
                sys.path.append('/Users/cal/DEV/RIF/systems')
                from system_context_apis import create_system_context_api
                from knowledge.database.dpibs_optimization import DPIBSPerformanceOptimizer
                
                optimizer = DPIBSPerformanceOptimizer()
                system_api = create_system_context_api(optimizer)
                
                return system_api.get_system_health()
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"System health check failed: {str(e)}")
        
        @self.app.post("/api/v1/system/context/store")
        async def store_system_context(context_data: SystemContextRequest):
            """Store system context snapshot for future reference"""
            try:
                sys.path.append('/Users/cal/DEV/RIF/systems')
                from system_context_apis import create_system_context_api
                from knowledge.database.dpibs_optimization import DPIBSPerformanceOptimizer
                
                optimizer = DPIBSPerformanceOptimizer()
                system_api = create_system_context_api(optimizer)
                
                context_id = system_api.store_system_context(
                    context_data.context_name,
                    context_data.context_type, 
                    context_data.system_snapshot,
                    context_data.confidence_level
                )
                
                return {
                    "status": "success",
                    "context_id": context_id,
                    "stored_at": datetime.now().isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to store system context: {str(e)}")
        
        # Integration Architecture API endpoints (Issue #141)
        @self.app.get("/api/v1/integration/compatibility/validate")
        async def validate_integration_compatibility():
            """
            Validate MCP Knowledge Server compatibility before migration.
            Issue #141: Integration Architecture + Migration Plan
            Target: 100% backward compatibility with <5% performance overhead
            """
            try:
                start_time = time.time()
                
                # Import integration API (lazy load for performance)
                sys.path.append('/Users/cal/DEV/RIF/api')
                from integration.architecture import create_integration_api
                from knowledge.database.dpibs_optimization import DPIBSPerformanceOptimizer
                
                optimizer = DPIBSPerformanceOptimizer()
                integration_api = create_integration_api(optimizer)
                
                result = await integration_api.validate_compatibility()
                
                validation_time = (time.time() - start_time) * 1000
                result["performance"] = {
                    "validation_time_ms": validation_time,
                    "target_met": validation_time < 5000  # 5 second validation target
                }
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Compatibility validation failed: {str(e)}")
        
        @self.app.post("/api/v1/integration/migration/execute")
        async def execute_migration_plan(migration_request: MigrationRequest):
            """
            Execute comprehensive migration plan with 5 defined phases.
            Issue #141: Integration Architecture + Migration Plan
            Target: <30 minutes total execution with rollback capability
            """
            try:
                start_time = time.time()
                
                sys.path.append('/Users/cal/DEV/RIF/api')
                from integration.architecture import create_integration_api
                from knowledge.database.dpibs_optimization import DPIBSPerformanceOptimizer
                
                optimizer = DPIBSPerformanceOptimizer()
                integration_api = create_integration_api(optimizer)
                
                result = await integration_api.execute_migration(migration_request.dry_run)
                
                execution_time = (time.time() - start_time) * 1000
                result["api_performance"] = {
                    "api_execution_time_ms": execution_time,
                    "total_migration_time_minutes": result.get("total_duration_minutes", 0),
                    "migration_target_met": result.get("target_met", False)
                }
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Migration execution failed: {str(e)}")
        
        @self.app.get("/api/v1/integration/migration/status")
        async def get_migration_status():
            """Get current migration status and progress"""
            try:
                sys.path.append('/Users/cal/DEV/RIF/api')
                from integration.architecture import create_integration_api
                from knowledge.database.dpibs_optimization import DPIBSPerformanceOptimizer
                
                optimizer = DPIBSPerformanceOptimizer()
                integration_api = create_integration_api(optimizer)
                
                return integration_api.get_migration_status()
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get migration status: {str(e)}")
        
        @self.app.get("/api/v1/integration/health")
        async def get_integration_health():
            """Get integration architecture health status"""
            try:
                sys.path.append('/Users/cal/DEV/RIF/api')
                from integration.architecture import create_integration_api
                from knowledge.database.dpibs_optimization import DPIBSPerformanceOptimizer
                
                optimizer = DPIBSPerformanceOptimizer()
                integration_api = create_integration_api(optimizer)
                
                return {
                    "status": "healthy",
                    "integration_available": True,
                    "migration_status": integration_api.get_migration_status(),
                    "compatibility_validated": integration_api.architect.mcp_integration_state.get("compatibility_validated", False),
                    "performance_baseline": integration_api.architect.migration_state.get("performance_baseline"),
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Integration health check failed: {str(e)}")
    
    def _setup_error_handlers(self):
        """Setup standardized error handling"""
        
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """Handle HTTP exceptions with standardized format"""
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "code": exc.status_code,
                        "message": exc.detail,
                        "timestamp": datetime.now().isoformat(),
                        "path": request.url.path
                    }
                }
            )
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Handle general exceptions"""
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": 500,
                        "message": "Internal server error",
                        "timestamp": datetime.now().isoformat(),
                        "path": request.url.path
                    }
                }
            )
    
    def _record_performance_metric(self, metric: Dict[str, Any]):
        """Record performance metric for monitoring"""
        self.request_metrics.append(metric)
        
        # Keep only last 1000 metrics in memory
        if len(self.request_metrics) > 1000:
            self.request_metrics = self.request_metrics[-1000:]
    
    def start_server(self, host: str = "127.0.0.1", port: int = 8000, **kwargs):
        """Start the API server"""
        uvicorn.run(self.app, host=host, port=port, **kwargs)


# Pydantic models for request/response validation
class ContextRequest(BaseModel):
    """Request model for context optimization"""
    agent_type: str = Field(..., description="Type of RIF agent requesting context")
    task_context: Dict[str, Any] = Field(..., description="Task-specific context information")
    issue_number: Optional[int] = Field(None, description="GitHub issue number for context")
    max_items: Optional[int] = Field(10, description="Maximum context items to return")

class OptimizationRequest(BaseModel):
    """Request model for context optimization feedback"""
    agent_type: str = Field(..., description="Type of RIF agent")
    context_used: bool = Field(..., description="Whether context was used effectively")
    decisions_made: int = Field(..., description="Number of decisions made")
    problems_found: int = Field(0, description="Number of problems identified")

class SystemContextRequest(BaseModel):
    """Request model for storing system context snapshots"""
    context_name: str = Field(..., description="Name/identifier for the system context")
    context_type: str = Field(..., description="Type of system context (e.g., 'component_analysis', 'dependency_graph')")
    system_snapshot: Dict[str, Any] = Field(..., description="Complete system context snapshot")
    confidence_level: float = Field(1.0, description="Confidence level in the context accuracy (0.0 to 1.0)")

class MigrationRequest(BaseModel):
    """Request model for migration execution"""
    dry_run: bool = Field(True, description="Execute migration in dry run mode for testing")
    force_migration: bool = Field(False, description="Force migration even if compatibility issues exist")
    backup_before_migration: bool = Field(True, description="Create system backup before migration")

class RollbackRequest(BaseModel):
    """Request model for rollback execution"""
    migration_id: str = Field(..., description="ID of the migration to rollback")
    force_rollback: bool = Field(False, description="Force rollback even if not recommended")

# Factory function for easy API creation
def create_api_framework(**kwargs) -> APIFramework:
    """Create and configure API framework instance"""
    return APIFramework(**kwargs)

# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DPIBS Core API Framework")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Create API framework
    api = create_api_framework()
    
    print(f"Starting DPIBS Core API Framework on {args.host}:{args.port}")
    print(f"API Documentation: http://{args.host}:{args.port}/api/v1/docs")
    print(f"Health Check: http://{args.host}:{args.port}/api/v1/health")
    
    # Start server
    api.start_server(host=args.host, port=args.port, reload=args.reload)