#!/bin/bash
# TimescaleDB Setup Script for PlusEV Wealth Engine 6
# This script installs and configures TimescaleDB on Amazon Linux 2

set -e

# Variables
PROJECT_NAME="${project_name}"
DB_PASSWORD="${db_password}"
LOG_FILE="/var/log/timescaledb-setup.log"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a $LOG_FILE
}

log "Starting TimescaleDB setup for $PROJECT_NAME"

# Update system
log "Updating system packages..."
yum update -y

# Install required packages
log "Installing required packages..."
yum install -y gcc gcc-c++ make wget curl git

# Install PostgreSQL 14
log "Installing PostgreSQL 14..."
yum install -y https://download.postgresql.org/pub/repos/yum/reporpms/EL-7-x86_64/pgdg-redhat-repo-latest.noarch.rpm
yum install -y postgresql14-server postgresql14-devel postgresql14-contrib

# Initialize PostgreSQL
log "Initializing PostgreSQL..."
/usr/pgsql-14/bin/postgresql-14-setup initdb

# Enable and start PostgreSQL
log "Starting PostgreSQL service..."
systemctl enable postgresql-14
systemctl start postgresql-14

# Wait for PostgreSQL to start
sleep 10

# Install TimescaleDB
log "Installing TimescaleDB..."

# Add TimescaleDB repository
cat > /etc/yum.repos.d/timescale_timescaledb.repo << 'EOF'
[timescale_timescaledb]
name=timescale_timescaledb
baseurl=https://packagecloud.io/timescale/timescaledb/el/7/$basearch
repo_gpgcheck=1
gpgcheck=0
enabled=1
gpgkey=https://packagecloud.io/timescale/timescaledb/gpgkey
sslverify=1
sslcacert=/etc/pki/tls/certs/ca-bundle.crt
metadata_expire=300
EOF

# Install TimescaleDB extension
yum install -y timescaledb-2-postgresql-14

# Configure PostgreSQL for TimescaleDB
log "Configuring PostgreSQL for TimescaleDB..."

# Backup original config
cp /var/lib/pgsql/14/data/postgresql.conf /var/lib/pgsql/14/data/postgresql.conf.backup
cp /var/lib/pgsql/14/data/pg_hba.conf /var/lib/pgsql/14/data/pg_hba.conf.backup

# Configure PostgreSQL
cat >> /var/lib/pgsql/14/data/postgresql.conf << 'EOF'

# TimescaleDB Configuration for PlusEV Wealth Engine 6
# Optimized for time-series market data

# Extensions
shared_preload_libraries = 'timescaledb'

# Memory settings (optimized for t3.micro)
shared_buffers = 128MB
effective_cache_size = 384MB
work_mem = 4MB
maintenance_work_mem = 64MB

# WAL settings
wal_buffers = 16MB
checkpoint_completion_target = 0.9
wal_writer_delay = 200ms

# Connection settings
max_connections = 100
listen_addresses = '*'
port = 5432

# Logging
log_destination = 'stderr'
logging_collector = on
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_rotation_age = 1d
log_rotation_size = 100MB
log_min_duration_statement = 1000
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_statement = 'ddl'

# Performance tuning for time-series data
random_page_cost = 1.1
effective_io_concurrency = 200
default_statistics_target = 100

# TimescaleDB specific settings
timescaledb.max_background_workers = 8
EOF

# Configure pg_hba.conf for network access
cat > /var/lib/pgsql/14/data/pg_hba.conf << 'EOF'
# PostgreSQL Client Authentication Configuration File
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# "local" is for Unix domain socket connections only
local   all             all                                     peer

# IPv4 local connections:
host    all             all             127.0.0.1/32            md5

# IPv4 VPC connections (adjust CIDR as needed)
host    all             all             10.0.0.0/16             md5

# IPv6 local connections:
host    all             all             ::1/128                 md5
EOF

# Restart PostgreSQL to apply configuration
log "Restarting PostgreSQL with new configuration..."
systemctl restart postgresql-14

# Wait for PostgreSQL to restart
sleep 10

# Set up PostgreSQL user and database
log "Setting up PostgreSQL user and database..."

# Switch to postgres user and set up database
sudo -u postgres bash << EOF
# Set password for postgres user
psql -c "ALTER USER postgres PASSWORD '$DB_PASSWORD';"

# Create market_data database
createdb market_data

# Connect to market_data database and enable TimescaleDB
psql -d market_data -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# Create sample tables for market data
psql -d market_data << 'SQL'
-- Create market data table
CREATE TABLE market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    price DECIMAL(15,8) NOT NULL,
    volume DECIMAL(20,8),
    bid DECIMAL(15,8),
    ask DECIMAL(15,8),
    bid_size DECIMAL(20,8),
    ask_size DECIMAL(20,8),
    exchange VARCHAR(50),
    data_source VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable (TimescaleDB)
SELECT create_hypertable('market_data', 'time');

-- Create indexes for performance
CREATE INDEX idx_market_data_symbol_time ON market_data (symbol, time DESC);
CREATE INDEX idx_market_data_time ON market_data (time DESC);
CREATE INDEX idx_market_data_symbol ON market_data (symbol);

-- Create trades table
CREATE TABLE trades (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    price DECIMAL(15,8) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    side VARCHAR(10) NOT NULL, -- 'buy' or 'sell'
    trade_id VARCHAR(100),
    exchange VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert trades to hypertable
SELECT create_hypertable('trades', 'time');

-- Create indexes for trades
CREATE INDEX idx_trades_symbol_time ON trades (symbol, time DESC);
CREATE INDEX idx_trades_time ON trades (time DESC);

-- Create orderbook table
CREATE TABLE orderbook (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL, -- 'bid' or 'ask'
    price DECIMAL(15,8) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    level INTEGER,
    exchange VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert orderbook to hypertable
SELECT create_hypertable('orderbook', 'time');

-- Create indexes for orderbook
CREATE INDEX idx_orderbook_symbol_time ON orderbook (symbol, time DESC);
CREATE INDEX idx_orderbook_time ON orderbook (time DESC);

-- Create aggregated data table (for performance)
CREATE TABLE market_data_1min (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open DECIMAL(15,8),
    high DECIMAL(15,8),
    low DECIMAL(15,8),
    close DECIMAL(15,8),
    volume DECIMAL(20,8),
    vwap DECIMAL(15,8),
    count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('market_data_1min', 'time');

-- Create indexes
CREATE INDEX idx_market_data_1min_symbol_time ON market_data_1min (symbol, time DESC);

-- Create continuous aggregate for 1-minute OHLCV
CREATE MATERIALIZED VIEW market_data_1min_view
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 minute', time) AS time,
       symbol,
       first(price, time) AS open,
       max(price) AS high,
       min(price) AS low,
       last(price, time) AS close,
       sum(volume) AS volume,
       (sum(price * volume) / sum(volume)) AS vwap,
       count(*) AS count
FROM market_data
GROUP BY time_bucket('1 minute', time), symbol;

-- Add refresh policy
SELECT add_continuous_aggregate_policy('market_data_1min_view',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute');

-- Create data retention policy (keep raw data for 30 days)
SELECT add_retention_policy('market_data', INTERVAL '30 days');
SELECT add_retention_policy('trades', INTERVAL '30 days');
SELECT add_retention_policy('orderbook', INTERVAL '7 days'); -- Orderbook data kept for shorter time

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Insert sample data for testing
INSERT INTO market_data (time, symbol, price, volume, exchange, data_source) VALUES
(NOW(), 'BTCUSD', 45000.00, 1.5, 'binance', 'websocket'),
(NOW() - INTERVAL '1 minute', 'BTCUSD', 44950.00, 2.1, 'binance', 'websocket'),
(NOW() - INTERVAL '2 minutes', 'BTCUSD', 45100.00, 0.8, 'binance', 'websocket');

SQL
EOF

# Set up TimescaleDB monitoring
log "Setting up TimescaleDB monitoring..."

# Create monitoring script
cat > /usr/local/bin/timescaledb-monitor.sh << 'EOF'
#!/bin/bash
# TimescaleDB monitoring script

export PGPASSWORD="$DB_PASSWORD"
PSQL="/usr/pgsql-14/bin/psql -h localhost -U postgres -d market_data"

echo "=== TimescaleDB Status Report ==="
echo "Database: market_data"
echo "Time: $(date)"
echo

# Check database size
echo "Database sizes:"
$PSQL -c "SELECT datname, pg_size_pretty(pg_database_size(datname)) as size FROM pg_database WHERE datname IN ('market_data', 'postgres');"
echo

# Check hypertable info
echo "Hypertable information:"
$PSQL -c "SELECT hypertable_name, num_chunks, table_bytes, index_bytes, total_bytes FROM timescaledb_information.hypertables;"
echo

# Check recent data
echo "Recent market data (last 10 records):"
$PSQL -c "SELECT time, symbol, price, volume FROM market_data ORDER BY time DESC LIMIT 10;"
echo

# Check connections
echo "Active connections:"
$PSQL -c "SELECT count(*) as active_connections FROM pg_stat_activity WHERE state = 'active';"
echo

echo "=== End Report ==="

# Log to syslog
logger "TimescaleDB Monitor - Database healthy, monitoring market data"
EOF

chmod +x /usr/local/bin/timescaledb-monitor.sh

# Set up cron job for monitoring
echo "*/10 * * * * /usr/local/bin/timescaledb-monitor.sh" | crontab -

# Configure firewall (if running)
log "Configuring firewall..."
systemctl status firewalld &>/dev/null && {
    firewall-cmd --permanent --add-port=5432/tcp
    firewall-cmd --reload
} || log "Firewall not running, skipping configuration"

# Set up logrotate for PostgreSQL
cat > /etc/logrotate.d/postgresql-14 << 'EOF'
/var/lib/pgsql/14/data/log/*.log {
    weekly
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    copytruncate
    su postgres postgres
}
EOF

# Create backup script
log "Creating backup script..."
cat > /usr/local/bin/timescaledb-backup.sh << 'EOF'
#!/bin/bash
# TimescaleDB backup script

BACKUP_DIR="/var/backups/timescaledb"
DATE=$(date +%Y%m%d_%H%M%S)
DATABASE="market_data"

mkdir -p $BACKUP_DIR

export PGPASSWORD="$DB_PASSWORD"

# Create backup
/usr/pgsql-14/bin/pg_dump -h localhost -U postgres -d $DATABASE -f $BACKUP_DIR/market_data_$DATE.sql

# Compress backup
gzip $BACKUP_DIR/market_data_$DATE.sql

# Remove backups older than 7 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete

echo "Backup completed: market_data_$DATE.sql.gz"
logger "TimescaleDB backup completed: market_data_$DATE.sql.gz"
EOF

chmod +x /usr/local/bin/timescaledb-backup.sh

# Set up daily backup cron job
echo "0 2 * * * /usr/local/bin/timescaledb-backup.sh" | crontab -

# Final verification
log "Final TimescaleDB setup verification..."
systemctl status postgresql-14 --no-pager

# Test connection
export PGPASSWORD="$DB_PASSWORD"
if /usr/pgsql-14/bin/psql -h localhost -U postgres -d market_data -c "SELECT version();" &>/dev/null; then
    log "TimescaleDB connection test successful!"
else
    log "ERROR: TimescaleDB connection test failed!"
    exit 1
fi

# Create success marker
touch /tmp/timescaledb-setup-complete

log "TimescaleDB setup completed successfully for $PROJECT_NAME!"
log "Database ready for high-performance time-series market data!"
log "Access: psql -h localhost -U postgres -d market_data"