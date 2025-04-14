# Database Migration System

This directory contains the Alembic-based migration system for the Trading Platform. It allows for versioned database schema management, with support for both PostgreSQL and TimescaleDB.

## Features

- Multi-database support (PostgreSQL and TimescaleDB)
- TimescaleDB hypertable support for high-performance time-series data
- Reversible migrations with proper downgrade paths
- Audit triggers for critical tables
- Data validation and constraints for financial integrity
- Helper utilities for common migration patterns

## Directory Structure

```
app/db/migrations/
├── env.py                # Alembic environment configuration
├── README.md             # This file
├── script.py.mako        # Template for migration scripts
├── helpers/              # Helper modules
│   └── timescale.py      # TimescaleDB-specific helpers
└── versions/             # Migration script files
    ├── 20250411_0001_init_schemas.py
    ├── 20250411_0002_init_timescaledb.py
    ├── 20250411_0003_create_market_data_tables.py
    ├── 20250411_0004_create_auth_and_trading_tables.py
    └── 20250411_0005_add_audit_triggers.py
```

## Quick Start

### Initialize the Database

To initialize both PostgreSQL and TimescaleDB databases:

```bash
python -m app.db.commands init
```

If you need to start fresh (drop existing tables):

```bash
python -m app.db.commands init --drop-first
```

If you don't have TimescaleDB installed:

```bash
python -m app.db.commands init --skip-timescale
```

### Create a New Migration

To create a new migration for PostgreSQL:

```bash
python -m app.db.commands revision -m "Add user preferences table" --database postgres
```

To autogenerate a migration based on model changes:

```bash
python -m app.db.commands revision -m "Update order table" -a --database postgres
```

To create a migration for TimescaleDB:

```bash
python -m app.db.commands revision -m "Add trade history table" --database timescale
```

### Apply Migrations

To upgrade to the latest version:

```bash
python -m app.db.commands upgrade --database postgres
```

To upgrade to a specific version:

```bash
python -m app.db.commands upgrade -r 20250411_0004 --database postgres
```

### Revert Migrations

To downgrade one version:

```bash
python -m app.db.commands downgrade --database postgres
```

To downgrade to a specific version:

```bash
python -m app.db.commands downgrade -r 20250411_0001 --database postgres
```

### Check Migration Status

To see the current migration version:

```bash
python -m app.db.commands current --database postgres
```

To see the migration history:

```bash
python -m app.db.commands history --database postgres
```

## Writing Migrations

### General Guidelines

1. **Always include both upgrade and downgrade paths**
2. **Use appropriate schemas** (auth, trading, market_data, etc.)
3. **Add indexes for commonly queried columns**
4. **Include appropriate constraints** for data integrity
5. **Use explicit naming conventions** for constraints and indexes

### TimescaleDB-Specific Guidelines

1. **Create regular tables first**, then convert to hypertables
2. **Choose appropriate chunk intervals** based on data rate and query patterns
3. **Compress older chunks** for storage efficiency
4. **Create continuous aggregates** for common time-bucket queries
5. **Set up retention policies** for automatic data management

### Example: Creating a TimescaleDB Table

```python
def upgrade():
    # Create regular table first
    op.create_table('trades',
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('symbol', sa.String(20), nullable=False, index=True),
        sa.Column('price', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('quantity', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('side', sa.String(4), nullable=False),
        sa.PrimaryKeyConstraint('timestamp', 'symbol'),
        schema='market_data'
    )
    
    # Convert to hypertable
    create_hypertable(
        'market_data.trades', 
        'timestamp', 
        chunk_time_interval='1 hour'
    )
    
    # Add compression
    add_hypertable_compression(
        'market_data.trades',
        compress_after='7 days',
        compress_segmentby=['symbol'],
        compress_orderby='timestamp'
    )
    
    # Create a continuous aggregate
    create_continuous_aggregate(
        'market_data.hourly_trades',
        'market_data.trades',
        """
        SELECT
            time_bucket('1 hour', timestamp) AS bucket,
            symbol,
            first(price, timestamp) AS open_price,
            last(price, timestamp) AS close_price,
            count(*) AS trade_count,
            sum(quantity) AS volume
        FROM market_data.trades
        GROUP BY bucket, symbol
        """,
        refresh_interval='5 minutes'
    )
```

## Best Practices

### Performance Considerations

- **Add indexes selectively** - Only add indexes for common query patterns
- **Use hypertables for time-series data** - Enables TimescaleDB optimizations
- **Consider partitioning** - For very large tables, add space partitioning
- **Use continuous aggregates** - Pre-aggregate data for common time ranges
- **Compress older data** - Significant storage savings for historical data

### Financial Data Integrity

- **Use numeric type for monetary values** - Never use float for financial data
- **Add appropriate constraints** - Check constraints for valid ranges
- **Include audit triggers** - Track all changes to critical financial data
- **Use transactions** - Ensure atomic operations for related changes

### Migration Dependencies

- **Use proper revision chaining** - Set correct `down_revision` values
- **Use database-specific migrations** - Set the `database` attribute
- **Manage cross-database dependencies** - Use the `depends_on` attribute

## Troubleshooting

### Common Issues

1. **Migration fails with "relation already exists"**
   - Make sure you're not recreating an existing table
   - Check that the table doesn't exist in another schema

2. **TimescaleDB functions not found**
   - Ensure TimescaleDB extension is installed
   - Check that the database user has appropriate permissions

3. **Downgrade fails**
   - Ensure all downgrade paths are implemented correctly
   - Check for dependencies that might prevent dropping objects

4. **Autogenerate produces unwanted changes**
   - Update your models to match the current database state
   - Consider selective autogeneration with `include_object`

### Getting Help

For more information:
- Check the [Alembic documentation](https://alembic.sqlalchemy.org/en/latest/)
- For TimescaleDB issues, see the [TimescaleDB documentation](https://docs.timescale.com/)