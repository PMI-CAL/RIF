# MCP Server Changes Reversal Instructions

## Files to Delete
```bash
rm -f /Users/cal/DEV/RIF/mcp_server_fix.py
```

## Files to Restore

### 1. `/Users/cal/DEV/RIF/.claude/settings.local.json`
Restore line 68 to add back the testproj directory (though it doesn't exist):
```json
"additionalDirectories": [
  "/Users/cal/DEV/testproj-1755548666",
  "/Users/cal/DEV"
]
```

### 2. `/Users/cal/Library/Application Support/Claude/claude_desktop_config.json`
Remove lines 36-43 (the entire "rif-knowledge" section)

### 3. `/Users/cal/DEV/RIF/mcp/claude-code-knowledge/server.py`
This file needs to be restored to its original state. The changes were:
- Lines 1-54: Revert warning suppression and import changes
- Lines 167-190: Revert `_setup_logging` method 
- Lines 901-1024: Revert entire `main()` function

## Git Commands to Revert (if in git)
```bash
git checkout -- /Users/cal/DEV/RIF/mcp/claude-code-knowledge/server.py
git checkout -- /Users/cal/DEV/RIF/.claude/settings.local.json
```

## Manual Revert for claude_desktop_config.json
Edit `/Users/cal/Library/Application Support/Claude/claude_desktop_config.json` and remove the "rif-knowledge" section (lines 36-43).