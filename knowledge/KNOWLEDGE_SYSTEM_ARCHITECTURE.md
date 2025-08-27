# RIF Knowledge System Architecture - FINAL WORKING VERSION

## System Overview

The RIF Knowledge System is now **FULLY OPERATIONAL** with 12,933+ entities and growing. This document captures the working architecture after resolving multiple critical failures.

## Current Architecture

### Database: DuckDB
- **Primary Database**: `/Users/cal/DEV/RIF/knowledge/knowledge.duckdb`
- **Schema Version**: 2.0 (consolidated)
- **Access Mode**: Read-only for MCP server, read-write for ingestion service
- **Contents**:
  - 12,933+ entities (code elements, patterns, issues, decisions, Claude docs)
  - 54+ relationships (growing with analysis)
  - Full Claude Code documentation (16 tools, 6 capabilities, 4 limitations)

### Components

#### 1. MCP Server (`mcp/rif-knowledge-server/rif_knowledge_server.py`)
- **Purpose**: Provides knowledge access to Claude Code via MCP protocol
- **Database Path**: `knowledge/knowledge.duckdb` (NOT chromadb/entities.duckdb)
- **Access Mode**: READ-ONLY (prevents locking conflicts)
- **Functions**: 
  - `query_knowledge`: Search entities by query
  - `get_patterns`: Retrieve patterns
  - `get_relationships`: Get entity relationships
  - `check_compatibility`: Validate approaches

#### 2. Auto Ingestion Service (`knowledge/auto_ingestion_service.py`)
- **Purpose**: Automatically ingests new knowledge files
- **Features**:
  - Monitors patterns/, issues/, decisions/ directories
  - Handles database locking conflicts
  - Can run continuously (`--watch`) or once
  - Extracts code entities from Python files
- **Usage**: `python3 knowledge/auto_ingestion_service.py --watch`

#### 3. Knowledge Interface (`knowledge/interface.py`)
- **Purpose**: Abstract interface for knowledge operations
- **Pattern**: Repository pattern with dependency injection
- **Methods**: store_knowledge, retrieve_knowledge, store_pattern, store_decision

#### 4. Database Interface (`knowledge/database/database_interface.py`)
- **Purpose**: Manages DuckDB database operations
- **Features**: Connection pooling, query execution, schema management

## Entity Types

### Knowledge Entities
- **patterns** (109): Successful code patterns
- **issues** (10): Issue resolutions 
- **decisions** (44): Architectural decisions

### Code Entities (12,770+)
- **module** (148): Python modules
- **class** (554): Class definitions
- **function** (3,012): Function definitions
- **variable** (8,454): Variables
- **constant** (576): Constants

### Claude Documentation (26)
- **tool** (16): Claude Code tools (Bash, Edit, Read, etc.)
- **capability** (6): Claude capabilities
- **limitation** (4): Claude limitations

## Relationships

Current relationships (54) include:
- Module → Class containment
- Class → Method containment  
- Function → Variable usage
- Pattern → Issue references

## Git Configuration

**CRITICAL**: Binary database files must be in `.gitignore`:
```gitignore
# Database files (binary, should not be tracked)
*.duckdb
*.duckdb.wal
*.sqlite3
knowledge/chromadb/
```

## Known Issues & Solutions

### Issue 1: Database Not Found
**Error**: "Cannot open database entities.duckdb"
**Cause**: MCP server looking in wrong location
**Solution**: Database is at `knowledge/knowledge.duckdb`, not `chromadb/entities.duckdb`

### Issue 2: Git Corruption
**Error**: Database files corrupted after git stash
**Cause**: Binary .duckdb files tracked by git
**Solution**: Added *.duckdb to .gitignore

### Issue 3: Database Locking
**Error**: "Conflicting lock" when ingesting
**Cause**: MCP server opens database read-only but prevents writes
**Solution**: Separate auto_ingestion_service handles writes

### Issue 4: Missing Relationships
**Status**: Only 54 relationships vs hundreds expected
**Solution**: Need enhanced relationship extraction

## Validation Commands

```bash
# Test the complete system
python3 knowledge/test_knowledge_system.py

# Run auto ingestion (once)
python3 knowledge/auto_ingestion_service.py

# Run auto ingestion (continuous)
python3 knowledge/auto_ingestion_service.py --watch

# Check database stats directly
python3 -c "import duckdb; conn = duckdb.connect('knowledge/knowledge.duckdb', read_only=True); print(conn.execute('SELECT type, COUNT(*) FROM entities GROUP BY type').fetchall())"
```

## Success Criteria

✅ **ACHIEVED**:
- 500+ entities requirement: **12,933 entities**
- Claude Code documentation: **100% complete**
- Database integrity: **No orphaned relationships**
- MCP server operational: **Connected and queryable**
- Auto ingestion working: **Monitors and ingests automatically**

## Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│            Claude Code (MCP Client)              │
└────────────────────┬────────────────────────────┘
                     │ MCP Protocol
                     ▼
┌─────────────────────────────────────────────────┐
│         MCP Server (READ-ONLY Access)           │
│     mcp/rif-knowledge-server/rif_knowledge.py   │
└────────────────────┬────────────────────────────┘
                     │ Read
                     ▼
┌─────────────────────────────────────────────────┐
│          DuckDB: knowledge.duckdb               │
│         12,933+ entities, 54+ relationships     │
└────────────────────┬────────────────────────────┘
                     │ Write
                     ▼
┌─────────────────────────────────────────────────┐
│    Auto Ingestion Service (READ-WRITE Access)   │
│       knowledge/auto_ingestion_service.py       │
└─────────────────────────────────────────────────┘
                     ▲
                     │ Monitor & Ingest
┌────────────────────┴────────────────────────────┐
│  Knowledge Files: patterns/, issues/, decisions/ │
└──────────────────────────────────────────────────┘
```

## Recovery Procedures

If the system breaks again:

1. **Check database exists**: 
   ```bash
   ls -la knowledge/knowledge.duckdb
   ```

2. **Verify MCP server path**:
   ```python
   # In mcp/rif-knowledge-server/rif_knowledge_server.py
   self.duckdb_path = self.knowledge_path / "knowledge.duckdb"
   ```

3. **Run validation test**:
   ```bash
   python3 knowledge/test_knowledge_system.py
   ```

4. **Re-ingest if needed**:
   ```bash
   python3 knowledge/auto_ingestion_service.py
   ```

5. **Check MCP logs**:
   ```bash
   tail -100 /Users/cal/Library/Caches/claude-cli-nodejs/-Users-cal-DEV-RIF/mcp-logs-rif-knowledge/*.txt
   ```

## Maintenance Tasks

- **Daily**: Monitor auto_ingestion_service logs
- **Weekly**: Run test_knowledge_system.py validation
- **Monthly**: Extract new code relationships
- **As Needed**: Add new knowledge patterns from resolved issues

---

**Status**: FULLY OPERATIONAL as of 2025-08-27
**Entities**: 12,933+
**Relationships**: 54+
**Claude Docs**: 100% complete