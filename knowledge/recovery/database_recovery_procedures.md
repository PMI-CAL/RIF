# Database Recovery Procedures
Generated: 2025-08-24T02:51:37.802765+00:00
Issue: #102 - Database Authentication Failure Fix

## Quick Health Check
```bash
python3 knowledge/recovery/database_health_check.py
```

## Common Issues and Solutions

### 1. Connection Issues
If you see connection errors:
- Check if database file exists: `knowledge/chromadb/entities.duckdb`
- Verify permissions on database directory
- Run health check script above

### 2. Schema Issues  
If tables are missing:
```python
from knowledge.database.database_interface import RIFDatabase
with RIFDatabase() as db:
    verification = db.verify_setup()
    print(verification)
```

### 3. Performance Issues
If database is slow:
- Check memory settings (should be 500MB)
- Run ANALYZE: `conn.execute("ANALYZE")`  
- Check connection pool stats

### 4. VSS Extension Issues
If vector search fails:
```sql
INSTALL vss;
LOAD vss;
SET hnsw_enable_experimental_persistence=true;
```

## Recovery Commands
```bash
# Restart database connections
python3 -c "
from knowledge.database.database_interface import RIFDatabase
with RIFDatabase() as db:
    print('Database restarted successfully')
"

# Check database statistics
python3 -c "
from knowledge.database.database_interface import RIFDatabase  
with RIFDatabase() as db:
    stats = db.get_database_stats()
    print(f'Entities: {stats["entities"]["total"]}')
    print(f'Relationships: {stats["relationships"]["total"]}')
"
```

## Contact Information
This fix was implemented by RIF-Implementer for Issue #102.
Database authentication failure resolved - no actual authentication issues found.
