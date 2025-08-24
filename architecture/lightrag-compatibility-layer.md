# LightRAG Compatibility Layer Architecture

## Overview

The LightRAG compatibility layer implements a transparent adapter pattern that enables seamless migration from the current LightRAG system to the new hybrid knowledge system without requiring any changes to existing agent code.

## Design Pattern: Adapter Pattern with Translation Layer

### Core Components

1. **LightRAG Compatibility Adapter** - Main interface maintaining exact API compatibility
2. **Query Translation Engine** - Bidirectional format conversion between systems
3. **Response Translator** - Semantic equivalence preservation in results
4. **Performance Monitor** - Real-time metrics and fallback tracking
5. **Migration Controller** - Gradual rollout management

### System Architecture

```
RIF Agents → LightRAG Compatibility Layer → Query Router
                                         ↓
                              Hybrid Knowledge System
                                         ↓
                              Query/Response Translation
                                         ↓
                              Performance Monitoring
```

## Integration Points

### Current Integration Point
- **File**: `lightrag/agents/agent_integration.py`
- **Classes**: `RIFAgentRAG`, `RIFAnalystRAG`, `RIFArchitectRAG`, etc.
- **Interface**: `retrieve_knowledge()`, `capture_knowledge()`, `query_similar_work()`

### New Implementation Structure
```
lightrag/
├── compatibility/
│   ├── __init__.py
│   ├── lightrag_adapter.py          # Main compatibility layer
│   ├── query_translator.py          # Translation engine
│   ├── metadata_mapper.py           # Metadata conversion
│   ├── performance_monitor.py       # Performance tracking
│   └── migration_controller.py      # Rollout management
```

## Implementation Guidelines

### Core Adapter Class
```python
class LightRAGCompatibilityAdapter:
    """
    Drop-in replacement for current LightRAG integration.
    Maintains exact API compatibility while routing to hybrid system.
    """
    
    def __init__(self, enable_hybrid: bool = False, fallback_enabled: bool = True):
        self.enable_hybrid = enable_hybrid
        self.fallback_enabled = fallback_enabled
        self.legacy_rag = get_lightrag_instance()
        self.hybrid_rag = get_hybrid_system() if enable_hybrid else None
        self.translator = QueryTranslationEngine()
        self.monitor = PerformanceMonitor()
    
    def retrieve_knowledge(self, query: str, collection_name: str = None, 
                         n_results: int = 5, filters: Dict = None) -> List[Dict]:
        """Maintains exact LightRAG API signature"""
        # Implementation handles routing and translation
```

## Quality Requirements

### Performance SLAs
- **Translation Latency**: <100ms for 95% of queries (P95 SLA)
- **Accuracy**: >99.5% semantic equivalence in translations
- **Fallback Time**: <500ms to switch to legacy system on hybrid failure
- **Memory Overhead**: <50MB additional memory usage

### Migration Strategy
1. **Phase 1**: Install compatibility layer with hybrid disabled
2. **Phase 2**: Enable hybrid for 10% of queries (shadow mode)
3. **Phase 3**: Gradual rollout from 10% → 50% → 90% → 100%
4. **Phase 4**: Retire legacy system after 30-day stability period

## Dependencies
- **Blocking**: Issue #34 (Context optimizer) must be complete
- **Enables**: Issue #37 (Parallel testing), #38 (Monitoring), #39 (Migration)

## Risk Mitigation
- Comprehensive test suite with 1000+ historical queries
- Real-time accuracy tracking with alerts at <99% accuracy
- Instant rollback capability via environment variable toggle