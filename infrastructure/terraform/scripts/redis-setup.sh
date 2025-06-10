#!/bin/bash
# Redis Setup Script for PlusEV Wealth Engine 6
# This script installs and configures Redis on Amazon Linux 2

set -e

# Variables
PROJECT_NAME="${project_name}"
LOG_FILE="/var/log/redis-setup.log"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a $LOG_FILE
}

log "Starting Redis setup for $PROJECT_NAME"

# Update system
log "Updating system packages..."
yum update -y

# Install Redis
log "Installing Redis..."
yum install -y redis

# Configure Redis
log "Configuring Redis..."

# Backup original config
cp /etc/redis.conf /etc/redis.conf.backup

# Configure Redis for production
cat > /etc/redis.conf << 'EOF'
# Redis configuration for PlusEV Wealth Engine 6
# Optimized for algorithmic trading performance

# Network
bind 0.0.0.0
port 6379
tcp-backlog 511
timeout 0
tcp-keepalive 300

# General
daemonize yes
supervised systemd
pidfile /var/run/redis/redis.pid
loglevel notice
logfile /var/log/redis/redis.log
databases 16

# Snapshotting
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /var/lib/redis/

# Security
# requirepass WealthEngine123!  # Uncomment for password auth

# Memory management
maxmemory 400mb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# Lazy freeing
lazyfree-lazy-eviction yes
lazyfree-lazy-expire yes
lazyfree-lazy-server-del yes

# Append only file
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
aof-load-truncated yes
aof-use-rdb-preamble yes

# Performance tuning for trading system
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
list-compress-depth 0
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64
hll-sparse-max-bytes 3000
stream-node-max-bytes 4096
stream-node-max-entries 100

# Client output buffer limits
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit replica 256mb 64mb 60
client-output-buffer-limit pubsub 32mb 8mb 60

# Slow log
slowlog-log-slower-than 10000
slowlog-max-len 128

# Latency monitoring
latency-monitor-threshold 100
EOF

# Create Redis user and directories
log "Setting up Redis user and directories..."
useradd -r -s /bin/false redis || true
mkdir -p /var/lib/redis /var/log/redis /var/run/redis
chown -R redis:redis /var/lib/redis /var/log/redis /var/run/redis
chmod 755 /var/lib/redis /var/log/redis /var/run/redis

# Set up logrotate for Redis
cat > /etc/logrotate.d/redis << 'EOF'
/var/log/redis/*.log {
    weekly
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    copytruncate
}
EOF

# Configure systemd service
log "Configuring Redis systemd service..."
cat > /etc/systemd/system/redis.service << 'EOF'
[Unit]
Description=Redis In-Memory Data Store for PlusEV Wealth Engine
After=network.target

[Service]
User=redis
Group=redis
ExecStart=/usr/bin/redis-server /etc/redis.conf
ExecStop=/usr/bin/redis-cli shutdown
Restart=always
RestartSec=3
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF

# Enable and start Redis
log "Starting Redis service..."
systemctl daemon-reload
systemctl enable redis
systemctl start redis

# Wait for Redis to start
sleep 5

# Test Redis
log "Testing Redis connection..."
if redis-cli ping | grep -q PONG; then
    log "Redis is running successfully!"
else
    log "ERROR: Redis is not responding!"
    exit 1
fi

# Install monitoring tools
log "Installing Redis monitoring tools..."
yum install -y htop iotop

# Configure system limits for Redis
log "Configuring system limits..."
echo "redis soft nofile 65535" >> /etc/security/limits.conf
echo "redis hard nofile 65535" >> /etc/security/limits.conf

# Disable transparent hugepages (recommended for Redis)
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag

# Make hugepage settings persistent
cat >> /etc/rc.local << 'EOF'
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag
EOF
chmod +x /etc/rc.local

# Configure sysctl for Redis optimization
cat >> /etc/sysctl.conf << 'EOF'
# Redis optimizations
vm.overcommit_memory = 1
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
EOF
sysctl -p

# Set up basic monitoring script
cat > /usr/local/bin/redis-monitor.sh << 'EOF'
#!/bin/bash
# Basic Redis monitoring script

REDIS_CLI="/usr/bin/redis-cli"
THRESHOLD_MEM=80  # Alert if memory usage > 80%
THRESHOLD_CONN=1000  # Alert if connections > 1000

# Get Redis stats
MEMORY_USED=$($REDIS_CLI info memory | grep used_memory_human | cut -d: -f2 | tr -d '\r')
MEMORY_PEAK=$($REDIS_CLI info memory | grep used_memory_peak_human | cut -d: -f2 | tr -d '\r')
CONNECTED_CLIENTS=$($REDIS_CLI info clients | grep connected_clients | cut -d: -f2 | tr -d '\r')
TOTAL_COMMANDS=$($REDIS_CLI info stats | grep total_commands_processed | cut -d: -f2 | tr -d '\r')

echo "=== Redis Status Report ==="
echo "Memory Used: $MEMORY_USED"
echo "Memory Peak: $MEMORY_PEAK"
echo "Connected Clients: $CONNECTED_CLIENTS"
echo "Total Commands: $TOTAL_COMMANDS"
echo "=========================="

# Log to syslog
logger "Redis Monitor - Memory: $MEMORY_USED, Clients: $CONNECTED_CLIENTS, Commands: $TOTAL_COMMANDS"
EOF

chmod +x /usr/local/bin/redis-monitor.sh

# Set up cron job for monitoring
echo "*/5 * * * * /usr/local/bin/redis-monitor.sh" | crontab -

# Final status check
log "Final Redis setup verification..."
systemctl status redis --no-pager
redis-cli info server | head -10

log "Redis setup completed successfully for $PROJECT_NAME!"
log "Redis is ready for high-performance algorithmic trading!"

# Create success marker
touch /tmp/redis-setup-complete

log "Setup complete. Redis is running on port 6379."