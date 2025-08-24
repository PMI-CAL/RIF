# DPIBS Operations Guide
## Production Operations and Maintenance

**Version**: 1.0  
**Last Updated**: August 2024  
**Target Audience**: System Administrators, DevOps Engineers, Site Reliability Engineers

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Installation and Deployment](#installation-and-deployment)
3. [Service Management](#service-management)
4. [Monitoring and Alerting](#monitoring-and-alerting)
5. [Backup and Recovery](#backup-and-recovery)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting](#troubleshooting)
8. [Security Operations](#security-operations)
9. [Capacity Planning](#capacity-planning)
10. [Maintenance Procedures](#maintenance-procedures)

---

## System Overview

### Architecture Components

```
Production DPIBS Architecture:

┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer                          │
│                   (nginx/haproxy)                          │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────┼───────────────────────────────────────────┐
│                 │              DPIBS Core                  │
│  ┌──────────────▼──────────────┐  ┌─────────────────────┐  │
│  │    Monitoring Dashboard     │  │  Performance        │  │
│  │    Port: 8080               │  │  Optimizer          │  │
│  └──────────┬──────────────────┘  └─────────────────────┘  │
│             │                                               │
│  ┌──────────▼──────────────┐  ┌─────────────────────────┐  │
│  │    Cache Manager        │  │  Alert System           │  │
│  │  • L1 (Memory): 100MB   │  │  • Email Notifications  │  │
│  │  • L2 (Compressed): 500MB│  │  • Slack Integration   │  │
│  │  • L3 (Persistent): 2GB │  │  • Webhook Support      │  │
│  └─────────────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼─────────────────────────────────┐
│                   Data Layer                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────┐ │
│  │   SQLite DB     │  │   Log Files     │  │  Metrics  │ │
│  │   (dpibs.db)    │  │   (/var/log/)   │  │  Storage  │ │
│  └─────────────────┘  └─────────────────┘  └───────────┘ │
└───────────────────────────────────────────────────────────┘
```

### Service Dependencies

| Service | Depends On | Port | Health Check |
|---------|------------|------|--------------|
| dpibs-monitoring | SQLite DB | 8080 | `/health` |
| dpibs-optimizer | Cache Layer | - | Internal |
| dpibs-alerts | SMTP/Slack | - | Config test |

---

## Installation and Deployment

### Production Deployment

#### Automated Deployment

```bash
# Clone repository
git clone <repository-url>
cd RIF

# Run production deployment
./deploy-production.sh

# Follow post-deployment steps
sudo cp /tmp/dpibs-monitoring.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable dpibs-monitoring
sudo systemctl start dpibs-monitoring
```

#### Manual Deployment Steps

1. **Environment Preparation**
   ```bash
   # Create service user
   sudo useradd -r -s /bin/false dpibs
   sudo mkdir -p /opt/dpibs
   sudo chown dpibs:dpibs /opt/dpibs
   
   # Install Python dependencies
   sudo pip3 install pyyaml sqlite3 asyncio
   ```

2. **File Installation**
   ```bash
   # Copy application files
   sudo cp -r systems/ /opt/dpibs/
   sudo cp -r config/ /opt/dpibs/
   sudo cp -r knowledge/ /opt/dpibs/
   sudo chown -R dpibs:dpibs /opt/dpibs/
   ```

3. **Service Configuration**
   ```bash
   # Create systemd service
   sudo tee /etc/systemd/system/dpibs-monitoring.service > /dev/null << 'EOF'
   [Unit]
   Description=DPIBS Monitoring Dashboard
   After=network.target
   
   [Service]
   Type=simple
   User=dpibs
   Group=dpibs
   WorkingDirectory=/opt/dpibs
   Environment=ENVIRONMENT=production
   Environment=DPIBS_LOG_LEVEL=INFO
   ExecStart=/usr/bin/python3 systems/dpibs_monitoring_dashboard.py
   Restart=always
   RestartSec=10
   StandardOutput=journal
   StandardError=journal
   
   [Install]
   WantedBy=multi-user.target
   EOF
   ```

#### Environment-Specific Configurations

**Development**:
```bash
export DPIBS_LOG_LEVEL=DEBUG
export DPIBS_CACHE_SIZE_MB=50
export DPIBS_METRICS_RETENTION_HOURS=24
```

**Staging**:
```bash
export DPIBS_LOG_LEVEL=INFO
export DPIBS_CACHE_SIZE_MB=200
export DPIBS_METRICS_RETENTION_HOURS=72
```

**Production**:
```bash
export DPIBS_LOG_LEVEL=WARNING
export DPIBS_CACHE_SIZE_MB=500
export DPIBS_METRICS_RETENTION_HOURS=168
export DPIBS_BACKUP_ENABLED=true
```

---

## Service Management

### SystemD Operations

#### Basic Service Control
```bash
# Start service
sudo systemctl start dpibs-monitoring

# Stop service
sudo systemctl stop dpibs-monitoring

# Restart service
sudo systemctl restart dpibs-monitoring

# Check service status
sudo systemctl status dpibs-monitoring

# Enable auto-start on boot
sudo systemctl enable dpibs-monitoring

# Disable auto-start
sudo systemctl disable dpibs-monitoring
```

#### Service Health Checks
```bash
# Check if service is running
systemctl is-active dpibs-monitoring

# Check if service is enabled
systemctl is-enabled dpibs-monitoring

# View service logs
journalctl -u dpibs-monitoring -f

# View recent service logs
journalctl -u dpibs-monitoring --since "1 hour ago"

# Check service failures
journalctl -u dpibs-monitoring --since "1 day ago" | grep -i error
```

### Process Management

#### Manual Process Control
```bash
# Find DPIBS processes
ps aux | grep dpibs

# Kill specific process
pkill -f "dpibs_monitoring_dashboard"

# Force kill if needed
pkill -9 -f "dpibs_monitoring_dashboard"

# Start manually for testing
cd /opt/dpibs
sudo -u dpibs python3 systems/dpibs_monitoring_dashboard.py
```

#### Process Monitoring
```bash
# Monitor resource usage
top -p $(pgrep -f dpibs_monitoring_dashboard)

# Detailed process information
ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | grep dpibs

# Monitor file descriptors
lsof -p $(pgrep -f dpibs_monitoring_dashboard)
```

---

## Monitoring and Alerting

### Key Performance Indicators

#### System Health Metrics

| Metric | Target | Critical Threshold | Command |
|--------|--------|-------------------|---------|
| Response Time | <200ms | >500ms | `curl -w "%{time_total}" http://localhost:8080/health` |
| Cache Hit Rate | >70% | <50% | `curl http://localhost:8080/cache/stats` |
| CPU Usage | <70% | >90% | `top -bn1 \| grep dpibs` |
| Memory Usage | <75% | >90% | `ps -eo pid,ppid,cmd,%mem \| grep dpibs` |
| Disk Space | <80% | >95% | `df -h /opt/dpibs` |

#### Application Metrics

```bash
# Get comprehensive system status
curl -s http://localhost:8080/metrics | jq '.'

# Check specific performance metrics
curl -s http://localhost:8080/metrics | jq '.performance'

# Monitor cache effectiveness
curl -s http://localhost:8080/cache/stats | jq '.l1_cache.hit_rate_percent'

# Check for active alerts
curl -s http://localhost:8080/alerts/active | jq '.'
```

### Alerting Configuration

#### Email Notifications
```yaml
# config/monitoring.yaml
alerts:
  enabled: true
  email:
    smtp_host: smtp.company.com
    smtp_port: 587
    username: alerts@company.com
    password: ${SMTP_PASSWORD}
    from_address: dpibs-alerts@company.com
    to_addresses:
      - ops-team@company.com
      - sre-team@company.com
```

#### Slack Integration
```yaml
alerts:
  slack:
    webhook_url: ${SLACK_WEBHOOK_URL}
    channel: "#ops-alerts"
    username: "DPIBS Monitor"
    icon_emoji: ":warning:"
```

#### Custom Webhook Alerts
```bash
# Test webhook functionality
curl -X POST ${WEBHOOK_URL} \
  -H "Content-Type: application/json" \
  -d '{
    "alert_type": "test",
    "severity": "info",
    "message": "Test alert from DPIBS",
    "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"
  }'
```

### External Monitoring Integration

#### Prometheus Metrics
```bash
# Add Prometheus endpoint
curl http://localhost:8080/metrics/prometheus
```

#### Nagios/Icinga Checks
```bash
#!/bin/bash
# /usr/local/lib/nagios/plugins/check_dpibs

HEALTH_CHECK=$(curl -s -f http://localhost:8080/health)
if [ $? -eq 0 ]; then
    echo "OK - DPIBS is healthy"
    exit 0
else
    echo "CRITICAL - DPIBS health check failed"
    exit 2
fi
```

---

## Backup and Recovery

### Automated Backup Strategy

#### Database Backup
```bash
#!/bin/bash
# /opt/dpibs/scripts/backup-database.sh

BACKUP_DIR="/backup/dpibs/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup SQLite database
sqlite3 /opt/dpibs/knowledge/dpibs.db ".backup $BACKUP_DIR/dpibs_$(date +%H%M).db"

# Compress backup
gzip "$BACKUP_DIR/dpibs_$(date +%H%M).db"

# Cleanup old backups (keep 30 days)
find /backup/dpibs -name "*.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_DIR"
```

#### Configuration Backup
```bash
#!/bin/bash
# /opt/dpibs/scripts/backup-config.sh

BACKUP_DIR="/backup/dpibs-config/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup configuration files
tar -czf "$BACKUP_DIR/config_$(date +%H%M).tar.gz" \
    /opt/dpibs/config/ \
    /opt/dpibs/.env.production \
    /etc/systemd/system/dpibs-monitoring.service

echo "Configuration backup completed: $BACKUP_DIR"
```

#### Automated Backup Schedule
```bash
# Add to crontab
sudo crontab -e

# Daily database backup at 2 AM
0 2 * * * /opt/dpibs/scripts/backup-database.sh >> /var/log/dpibs-backup.log 2>&1

# Weekly configuration backup on Sundays at 3 AM
0 3 * * 0 /opt/dpibs/scripts/backup-config.sh >> /var/log/dpibs-backup.log 2>&1
```

### Recovery Procedures

#### Database Recovery
```bash
# Stop DPIBS service
sudo systemctl stop dpibs-monitoring

# Restore from backup
cd /backup/dpibs/$(date +%Y%m%d)
gunzip -c dpibs_HHMM.db.gz > /opt/dpibs/knowledge/dpibs.db

# Fix permissions
sudo chown dpibs:dpibs /opt/dpibs/knowledge/dpibs.db

# Start service
sudo systemctl start dpibs-monitoring

# Verify recovery
curl http://localhost:8080/health
```

#### Configuration Recovery
```bash
# Stop service
sudo systemctl stop dpibs-monitoring

# Restore configuration
cd /backup/dpibs-config/$(date +%Y%m%d)
tar -xzf config_HHMM.tar.gz -C /

# Reload systemd
sudo systemctl daemon-reload

# Start service
sudo systemctl start dpibs-monitoring
```

#### Disaster Recovery Plan

1. **Assessment Phase**
   - Identify scope of failure
   - Determine data loss extent
   - Assess system availability

2. **Recovery Phase**
   ```bash
   # Emergency recovery script
   #!/bin/bash
   echo "Starting DPIBS disaster recovery..."
   
   # Stop all services
   sudo systemctl stop dpibs-monitoring
   
   # Restore from latest backup
   LATEST_BACKUP=$(ls -t /backup/dpibs/ | head -n1)
   echo "Restoring from: $LATEST_BACKUP"
   
   # Restore database
   gunzip -c /backup/dpibs/$LATEST_BACKUP/dpibs_*.db.gz > /opt/dpibs/knowledge/dpibs.db
   
   # Restore configuration
   tar -xzf /backup/dpibs-config/$LATEST_BACKUP/config_*.tar.gz -C /
   
   # Fix permissions
   sudo chown -R dpibs:dpibs /opt/dpibs/
   
   # Start services
   sudo systemctl daemon-reload
   sudo systemctl start dpibs-monitoring
   
   # Verify recovery
   sleep 30
   curl -f http://localhost:8080/health && echo "Recovery successful" || echo "Recovery failed"
   ```

---

## Performance Tuning

### Cache Optimization

#### Cache Size Tuning
```bash
# Monitor current cache usage
curl -s http://localhost:8080/cache/stats | jq '{
  l1_utilization: (.l1_cache.size_mb / 100 * 100),
  l2_utilization: (.l2_cache.size_mb / 500 * 100),
  l3_utilization: (.l3_cache.size_mb / 2048 * 100)
}'

# Adjust cache sizes based on utilization
# Edit config/monitoring.yaml
cache_configuration:
  l1_cache:
    max_size_mb: 200  # Increase if utilization >90%
  l2_cache:
    max_size_mb: 1000  # Increase if utilization >85%
  l3_cache:
    max_size_mb: 4096  # Increase if utilization >80%
```

#### Cache Hit Rate Optimization
```bash
# Analyze cache performance over time
curl -s http://localhost:8080/metrics | jq '.cache_summary'

# Identify cache misses by type
tail -n 1000 /var/log/journal/dpibs-monitoring.log | grep "cache_miss" | \
  awk '{print $5}' | sort | uniq -c | sort -nr
```

### Database Performance

#### SQLite Optimization
```sql
-- Connect to database
sqlite3 /opt/dpibs/knowledge/dpibs.db

-- Analyze database size
.dbinfo

-- Optimize database
VACUUM;
ANALYZE;

-- Check index usage
.schema

-- Add performance indexes if needed
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON dpibs_performance_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_cache_timestamp ON dpibs_cache_stats(timestamp);
```

#### Database Maintenance
```bash
# Automated database optimization script
#!/bin/bash
# /opt/dpibs/scripts/optimize-database.sh

echo "Starting database optimization..."

# Stop service temporarily
sudo systemctl stop dpibs-monitoring

# Backup before optimization
cp /opt/dpibs/knowledge/dpibs.db /opt/dpibs/knowledge/dpibs.db.backup

# Optimize database
sqlite3 /opt/dpibs/knowledge/dpibs.db "VACUUM; ANALYZE;"

# Check database integrity
INTEGRITY_CHECK=$(sqlite3 /opt/dpibs/knowledge/dpibs.db "PRAGMA integrity_check;")
if [ "$INTEGRITY_CHECK" != "ok" ]; then
    echo "Database integrity check failed, restoring backup..."
    mv /opt/dpibs/knowledge/dpibs.db.backup /opt/dpibs/knowledge/dpibs.db
    exit 1
fi

# Remove backup if successful
rm /opt/dpibs/knowledge/dpibs.db.backup

# Start service
sudo systemctl start dpibs-monitoring

echo "Database optimization completed successfully"
```

### System Resource Optimization

#### Memory Optimization
```bash
# Monitor memory usage
ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | head -20

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full python3 systems/dpibs_monitoring_dashboard.py

# Adjust Python memory settings
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=100000
```

#### CPU Optimization
```bash
# Monitor CPU usage patterns
sar -u 5 60  # Monitor for 5 minutes with 5-second intervals

# Adjust process priority if needed
sudo renice -n 5 $(pgrep -f dpibs_monitoring_dashboard)

# Enable CPU frequency scaling for better performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Service Won't Start

**Symptoms**:
- `systemctl start dpibs-monitoring` fails
- Error in systemd logs

**Diagnosis**:
```bash
# Check service status
sudo systemctl status dpibs-monitoring

# View detailed logs
journalctl -u dpibs-monitoring --no-pager

# Check configuration
python3 -m yaml config/monitoring.yaml

# Test manual startup
sudo -u dpibs python3 /opt/dpibs/systems/dpibs_monitoring_dashboard.py
```

**Solutions**:
- Fix configuration syntax errors
- Check file permissions
- Verify database accessibility
- Check port availability

#### Issue: High Memory Usage

**Symptoms**:
- System running out of memory
- OOM killer terminating processes

**Diagnosis**:
```bash
# Monitor memory usage over time
watch -n 5 'ps -eo pid,ppid,cmd,%mem --sort=-%mem | head -10'

# Check for memory leaks
valgrind --tool=massif python3 systems/dpibs_monitoring_dashboard.py

# Analyze cache memory usage
curl -s http://localhost:8080/cache/stats | jq '.memory_usage'
```

**Solutions**:
```bash
# Reduce cache sizes temporarily
curl -X POST http://localhost:8080/cache/clear

# Adjust cache configuration
# Edit config/monitoring.yaml to reduce max_size_mb values

# Restart service to apply changes
sudo systemctl restart dpibs-monitoring
```

#### Issue: Poor Performance

**Symptoms**:
- High response times (>500ms)
- Low cache hit rates (<50%)
- High CPU usage

**Diagnosis**:
```bash
# Generate performance report
python3 -c "
from systems.dpibs_monitoring_dashboard import DPIBSMonitoringDashboard
dashboard = DPIBSMonitoringDashboard()
report = dashboard.export_metrics_report(hours=1)
print('Performance Issues:')
print(f'Avg Response Time: {report.get(\"performance_summary\", {}).get(\"avg_response_time_ms\", \"N/A\")}ms')
print(f'Cache Hit Rate: {report.get(\"cache_summary\", {}).get(\"avg_hit_rate_percent\", \"N/A\")}%')
"

# Check system bottlenecks
iostat -x 1 5
sar -u 1 5
```

**Solutions**:
1. **Database Optimization**:
   ```bash
   # Rebuild indexes
   sqlite3 /opt/dpibs/knowledge/dpibs.db "REINDEX;"
   
   # Clean old data
   sqlite3 /opt/dpibs/knowledge/dpibs.db "
   DELETE FROM dpibs_performance_metrics 
   WHERE timestamp < datetime('now', '-30 days');
   "
   ```

2. **Cache Optimization**:
   ```bash
   # Increase cache sizes
   # Edit config/monitoring.yaml
   l1_cache:
     max_size_mb: 300
   l2_cache:
     max_size_mb: 800
   ```

### Log Analysis

#### Critical Log Patterns
```bash
# Find error patterns
journalctl -u dpibs-monitoring | grep -E "(ERROR|CRITICAL)" | tail -20

# Monitor real-time errors
journalctl -u dpibs-monitoring -f | grep --color=always -E "(ERROR|CRITICAL|WARNING)"

# Analyze performance degradation
journalctl -u dpibs-monitoring --since "1 hour ago" | grep "response_time" | \
  awk '{print $NF}' | sort -n | tail -10
```

#### Log Rotation
```bash
# Configure logrotate for DPIBS logs
sudo tee /etc/logrotate.d/dpibs > /dev/null << 'EOF'
/var/log/dpibs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    sharedscripts
    postrotate
        /bin/systemctl reload dpibs-monitoring > /dev/null 2>&1 || true
    endscript
}
EOF
```

---

## Security Operations

### Security Hardening

#### File Permissions
```bash
# Secure configuration files
sudo chmod 600 /opt/dpibs/.env.production
sudo chmod 600 /opt/dpibs/config/monitoring.yaml
sudo chown root:dpibs /opt/dpibs/.env.production

# Secure database
sudo chmod 640 /opt/dpibs/knowledge/dpibs.db
sudo chown dpibs:dpibs /opt/dpibs/knowledge/dpibs.db
```

#### Network Security
```bash
# Configure firewall (UFW example)
sudo ufw allow from 10.0.0.0/8 to any port 8080 comment 'DPIBS Dashboard - Internal'
sudo ufw deny from any to any port 8080 comment 'DPIBS Dashboard - External'

# Configure iptables for specific access
sudo iptables -A INPUT -p tcp --dport 8080 -s 10.0.0.0/8 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8080 -j DROP
```

#### SSL/TLS Configuration
```bash
# Generate SSL certificate
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /opt/dpibs/ssl/dpibs.key \
    -out /opt/dpibs/ssl/dpibs.crt

# Configure nginx proxy with SSL
sudo tee /etc/nginx/sites-available/dpibs > /dev/null << 'EOF'
server {
    listen 443 ssl;
    server_name dpibs.company.com;
    
    ssl_certificate /opt/dpibs/ssl/dpibs.crt;
    ssl_certificate_key /opt/dpibs/ssl/dpibs.key;
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF
```

### Access Control

#### Authentication Setup
```yaml
# config/monitoring.yaml - Add authentication
security:
  authentication:
    enabled: true
    type: basic_auth
    users:
      admin:
        password_hash: "$2b$12$..."
        permissions: ["read", "write", "admin"]
      readonly:
        password_hash: "$2b$12$..."
        permissions: ["read"]
```

#### API Security
```bash
# Generate API keys
python3 -c "import secrets; print(secrets.token_hex(32))"

# Configure rate limiting
# Add to config/monitoring.yaml
api:
  rate_limiting:
    enabled: true
    requests_per_minute: 60
    requests_per_hour: 1000
```

### Security Auditing

#### Regular Security Checks
```bash
#!/bin/bash
# /opt/dpibs/scripts/security-audit.sh

echo "DPIBS Security Audit - $(date)"
echo "=================================="

# Check file permissions
echo "Checking file permissions..."
find /opt/dpibs -type f -perm /o+w -exec ls -la {} \;

# Check for world-readable config files
echo "Checking configuration file permissions..."
find /opt/dpibs/config -type f -perm /o+r -exec ls -la {} \;

# Check SSL certificate expiration
echo "Checking SSL certificate..."
openssl x509 -in /opt/dpibs/ssl/dpibs.crt -noout -dates

# Check for failed login attempts (if authentication enabled)
echo "Checking failed login attempts..."
journalctl -u dpibs-monitoring | grep "authentication failed" | tail -10

# Check open ports
echo "Checking open network ports..."
netstat -tlnp | grep :8080
```

---

## Capacity Planning

### Resource Monitoring

#### System Resource Trends
```bash
# Monitor CPU trends
sar -u 1 0 | awk 'NR>3{sum+=$3} END{print "Average CPU:", sum/(NR-3)"%"}'

# Monitor memory trends
free -h && echo "Memory usage trend:"
sar -r 1 5

# Monitor disk usage trends
df -h /opt/dpibs
du -sh /opt/dpibs/knowledge/

# Monitor network I/O
sar -n DEV 1 5
```

#### Application Resource Usage
```bash
# Monitor DPIBS-specific resource usage
ps -eo pid,ppid,cmd,%mem,%cpu,rss --sort=-%mem | grep dpibs

# Check cache memory consumption
curl -s http://localhost:8080/cache/stats | jq '{
  total_cache_mb: (.l1_cache.size_mb + .l2_cache.size_mb + .l3_cache.size_mb),
  cache_efficiency: (.l1_cache.hit_rate_percent + .l2_cache.hit_rate_percent + .l3_cache.hit_rate_percent) / 3
}'
```

### Scaling Recommendations

#### Vertical Scaling Triggers

| Resource | Scale Up When | Recommended Action |
|----------|---------------|-------------------|
| CPU | >80% sustained | Add 2+ cores |
| Memory | >85% usage | Add 4+ GB RAM |
| Disk | >80% full | Add 20+ GB storage |
| Network | >80% bandwidth | Upgrade connection |

#### Horizontal Scaling Considerations

```bash
# Load balancer configuration (nginx example)
upstream dpibs_backend {
    server 10.0.0.10:8080 weight=3;
    server 10.0.0.11:8080 weight=3;
    server 10.0.0.12:8080 weight=2 backup;
}

server {
    listen 80;
    location / {
        proxy_pass http://dpibs_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Capacity Planning Metrics

#### Storage Growth Projection
```bash
# Calculate database growth rate
sqlite3 /opt/dpibs/knowledge/dpibs.db "
SELECT 
  DATE(timestamp) as date,
  COUNT(*) as records,
  (SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()) / 1024.0 / 1024.0 as size_mb
FROM dpibs_performance_metrics 
GROUP BY DATE(timestamp)
ORDER BY date DESC
LIMIT 7;
"
```

#### Performance Capacity
```bash
# Calculate request handling capacity
echo "Current performance metrics:"
curl -s http://localhost:8080/metrics | jq '{
  requests_per_second: .performance.requests_per_second,
  avg_response_time: .performance.avg_response_time_ms,
  cache_hit_rate: .performance.cache_hit_rate_percent
}'

# Estimate maximum capacity
echo "Estimated maximum capacity:"
echo "With current response time, max sustainable RPS: $((1000 / avg_response_time))"
```

---

## Maintenance Procedures

### Scheduled Maintenance

#### Daily Tasks
```bash
#!/bin/bash
# /opt/dpibs/scripts/daily-maintenance.sh

# Check service health
systemctl is-active --quiet dpibs-monitoring || {
    echo "ALERT: DPIBS service is down"
    systemctl restart dpibs-monitoring
}

# Backup database
/opt/dpibs/scripts/backup-database.sh

# Clean old logs
find /var/log -name "*dpibs*" -mtime +7 -delete

# Check disk space
DISK_USAGE=$(df /opt/dpibs | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 85 ]; then
    echo "WARNING: Disk usage is ${DISK_USAGE}%"
fi

# Performance check
RESPONSE_TIME=$(curl -w "%{time_total}" -s http://localhost:8080/health -o /dev/null)
if (( $(echo "$RESPONSE_TIME > 0.5" | bc -l) )); then
    echo "WARNING: High response time: ${RESPONSE_TIME}s"
fi
```

#### Weekly Tasks
```bash
#!/bin/bash
# /opt/dpibs/scripts/weekly-maintenance.sh

# Optimize database
/opt/dpibs/scripts/optimize-database.sh

# Rotate cache to test cold start performance
curl -X POST http://localhost:8080/cache/clear

# Security audit
/opt/dpibs/scripts/security-audit.sh

# Generate performance report
python3 -c "
from systems.dpibs_monitoring_dashboard import DPIBSMonitoringDashboard
dashboard = DPIBSMonitoringDashboard()
report = dashboard.export_metrics_report(hours=168)
print('Weekly Performance Report:', report)
"
```

#### Monthly Tasks
```bash
#!/bin/bash
# /opt/dpibs/scripts/monthly-maintenance.sh

# Full system backup
tar -czf "/backup/dpibs-full-$(date +%Y%m).tar.gz" /opt/dpibs/

# Update system packages (if automated updates disabled)
# sudo apt update && sudo apt upgrade -y

# Review and archive old logs
journalctl --vacuum-time=30d

# Capacity planning review
echo "Monthly Capacity Review:"
du -sh /opt/dpibs/
df -h
free -h

# Performance baseline update
echo "Updating performance baselines..."
# Archive current metrics for trend analysis
```

### Maintenance Mode

#### Enable Maintenance Mode
```bash
#!/bin/bash
# /opt/dpibs/scripts/enable-maintenance.sh

echo "Enabling maintenance mode..."

# Create maintenance page
sudo tee /var/www/html/maintenance.html > /dev/null << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>DPIBS Maintenance</title>
</head>
<body>
    <h1>System Under Maintenance</h1>
    <p>DPIBS is currently undergoing scheduled maintenance.</p>
    <p>Service will be restored shortly.</p>
</body>
</html>
EOF

# Update nginx to serve maintenance page
sudo nginx -s reload

# Stop DPIBS services
sudo systemctl stop dpibs-monitoring

echo "Maintenance mode enabled"
```

#### Disable Maintenance Mode
```bash
#!/bin/bash
# /opt/dpibs/scripts/disable-maintenance.sh

echo "Disabling maintenance mode..."

# Start DPIBS services
sudo systemctl start dpibs-monitoring

# Wait for service to be ready
sleep 30

# Test service health
curl -f http://localhost:8080/health || {
    echo "ERROR: Service health check failed"
    exit 1
}

# Restore normal nginx configuration
sudo nginx -s reload

# Remove maintenance page
sudo rm -f /var/www/html/maintenance.html

echo "Maintenance mode disabled - service restored"
```

### Emergency Procedures

#### Service Recovery
```bash
#!/bin/bash
# /opt/dpibs/scripts/emergency-recovery.sh

echo "Starting emergency recovery procedure..."

# Stop all DPIBS processes
pkill -f dpibs

# Check for corrupted database
sqlite3 /opt/dpibs/knowledge/dpibs.db "PRAGMA integrity_check;" || {
    echo "Database corrupted, restoring from backup..."
    LATEST_BACKUP=$(ls -t /backup/dpibs/*/dpibs_*.db.gz | head -n1)
    gunzip -c "$LATEST_BACKUP" > /opt/dpibs/knowledge/dpibs.db
}

# Reset cache directory
rm -rf /tmp/dpibs-cache-*

# Start service
systemctl start dpibs-monitoring

# Verify recovery
sleep 30
curl -f http://localhost:8080/health && echo "Recovery successful" || echo "Recovery failed"
```

---

## Appendix

### Configuration Templates

#### Production monitoring.yaml
```yaml
# Production DPIBS Configuration
performance_monitoring:
  enabled: true
  collection_interval_seconds: 60
  metrics_retention_hours: 168
  
cache_configuration:
  l1_cache:
    max_size_mb: 200
    ttl_seconds: 300
    eviction_policy: "LRU"
  l2_cache:
    max_size_mb: 800
    ttl_seconds: 1800
    compression_enabled: true
  l3_cache:
    max_size_mb: 4096
    ttl_seconds: 86400
    persistent_storage: true

alerts:
  enabled: true
  thresholds:
    response_time_ms: 200
    cache_hit_rate_percent: 70
    cpu_usage_percent: 70
    memory_usage_percent: 75
    error_rate_percent: 1
  
dashboard:
  enabled: true
  port: 8080
  auto_start: true
  
logging:
  level: "INFO"
  file: "/var/log/dpibs/monitoring.log"
  rotation: "daily"
  max_files: 30
```

### Service Templates

#### systemd service file
```ini
[Unit]
Description=DPIBS Monitoring Dashboard
Documentation=https://docs.dpibs.dev
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=dpibs
Group=dpibs
WorkingDirectory=/opt/dpibs
Environment=ENVIRONMENT=production
Environment=DPIBS_LOG_LEVEL=INFO
Environment=DPIBS_CONFIG_PATH=/opt/dpibs/config/monitoring.yaml
ExecStart=/usr/bin/python3 systems/dpibs_monitoring_dashboard.py
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=dpibs-monitoring

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/dpibs

[Install]
WantedBy=multi-user.target
```

### Monitoring Scripts

#### Health check script
```bash
#!/bin/bash
# /opt/dpibs/scripts/health-check.sh

# Exit codes: 0=OK, 1=WARNING, 2=CRITICAL

# Check service status
if ! systemctl is-active --quiet dpibs-monitoring; then
    echo "CRITICAL: DPIBS service is not running"
    exit 2
fi

# Check HTTP endpoint
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health)
if [ "$HTTP_CODE" != "200" ]; then
    echo "CRITICAL: HTTP health check failed (code: $HTTP_CODE)"
    exit 2
fi

# Check response time
RESPONSE_TIME=$(curl -w "%{time_total}" -s http://localhost:8080/health -o /dev/null)
if (( $(echo "$RESPONSE_TIME > 0.5" | bc -l) )); then
    echo "WARNING: High response time: ${RESPONSE_TIME}s"
    exit 1
fi

# Check cache hit rate
CACHE_HIT_RATE=$(curl -s http://localhost:8080/cache/stats | jq -r '.overall_hit_rate // 0')
if (( $(echo "$CACHE_HIT_RATE < 50" | bc -l) )); then
    echo "WARNING: Low cache hit rate: ${CACHE_HIT_RATE}%"
    exit 1
fi

echo "OK: All health checks passed"
exit 0
```

---

**DPIBS Operations Guide v1.0**  
*Development Process Intelligence & Benchmarking System*  
*© 2024 - Operations Documentation*