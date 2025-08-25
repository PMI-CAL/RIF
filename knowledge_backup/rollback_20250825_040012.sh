#!/bin/bash
# Rollback script for knowledge base cleanup
# Generated: Mon Aug 25 04:00:12 PDT 2025

set -e

echo "Rolling back knowledge base from backup..."
echo "Backup file: /Users/cal/dev/rif/knowledge_backup/knowledge_pre_cleanup_20250825_040012.tar.gz"

# Remove current knowledge directory
if [ -d "/Users/cal/dev/rif/knowledge" ]; then
    echo "Removing current knowledge directory..."
    rm -rf "/Users/cal/dev/rif/knowledge"
fi

# Extract backup
echo "Restoring from backup..."
tar -xzf "/Users/cal/dev/rif/knowledge_backup/knowledge_pre_cleanup_20250825_040012.tar.gz" -C "/Users/cal/dev/rif"

echo "Rollback complete!"
echo "Knowledge base restored from /Users/cal/dev/rif/knowledge_backup/knowledge_pre_cleanup_20250825_040012.tar.gz"
