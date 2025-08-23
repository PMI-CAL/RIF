"""
Test suite for Hybrid Query Planner - Issue #33

This test suite validates all components of the hybrid search system:
- Query parsing and intent recognition
- Strategy planning and optimization  
- Hybrid search execution with vector and graph searches
- Result ranking with multi-signal relevance scoring
- Performance requirements (<100ms P95 latency)
- Caching and optimization features

Run specific test categories:
- python -m pytest test_query_planner.py::TestQueryParser -v
- python -m pytest test_query_planner.py::TestPerformance -v
- python -m pytest test_query_planner.py::TestIntegration -v

Or run all tests:
- python -m pytest test_query_planner.py -v
"""

__version__ = "1.0.0"
__all__ = ["test_query_planner"]