#!/bin/bash
# Rollback script for knowledge base cleanup
# Generated: 2025-08-25T04:00:11.974407

set -e

echo "Rolling back knowledge base from backup..."
echo "Backup file: /Users/cal/dev/rif/knowledge_backup/knowledge_backup_20250825_040010.tar.gz"

# Remove current knowledge directory
if [ -d "/Users/cal/dev/rif/knowledge" ]; then
    echo "Removing current knowledge directory..."
    rm -rf "/Users/cal/dev/rif/knowledge"
fi

# Extract backup
echo "Restoring from backup..."
tar -xzf "/Users/cal/dev/rif/knowledge_backup/knowledge_backup_20250825_040010.tar.gz" -C "/Users/cal/dev/rif"

echo "Rollback complete!"
echo "Knowledge base restored from /Users/cal/dev/rif/knowledge_backup/knowledge_backup_20250825_040010.tar.gz"
