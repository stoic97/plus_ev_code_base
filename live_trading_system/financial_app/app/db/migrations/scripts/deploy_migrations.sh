#!/bin/bash
# Database migration deployment script
# This script handles database migrations for different environments

set -e  # Exit on any error

# Default values
ENV="development"
DATABASES=("postgres" "timescale")
ACTION="upgrade"
REVISION="head"
VERBOSE=false
DRY_RUN=false

# Display help
function show_help {
    echo "Database Migration Deployment Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -e, --environment ENV   Set environment (development, staging, production)"
    echo "  -d, --database DB       Specify database (postgres, timescale, all)"
    echo "  -a, --action ACTION     Action to perform (upgrade, downgrade, current, history)"
    echo "  -r, --revision REV      Revision identifier (default: head for upgrade, -1 for downgrade)"
    echo "  -v, --verbose           Enable verbose output"
    echo "  --dry-run               Just print what would be done without executing"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -e production -a upgrade                  # Upgrade all databases in production"
    echo "  $0 -e staging -d postgres -a downgrade -r -1 # Downgrade PostgreSQL in staging by 1 version"
    echo "  $0 -e production -d timescale -a current     # Check current TimescaleDB version in production"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENV="$2"
            shift 2
            ;;
        -d|--database)
            if [[ "$2" == "all" ]]; then
                DATABASES=("postgres" "timescale")
            else
                DATABASES=("$2")
            fi
            shift 2
            ;;
        -a|--action)
            ACTION="$2"
            shift 2
            ;;
        -r|--revision)
            REVISION="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate environment
if [[ "$ENV" != "development" && "$ENV" != "staging" && "$ENV" != "production" ]]; then
    echo "Error: Invalid environment '$ENV'. Must be development, staging, or production."
    exit 1
fi

# Validate action
if [[ "$ACTION" != "upgrade" && "$ACTION" != "downgrade" && 
      "$ACTION" != "current" && "$ACTION" != "history" ]]; then
    echo "Error: Invalid action '$ACTION'. Must be upgrade, downgrade, current, or history."
    exit 1
fi

# Set up environment-specific configuration
echo "Loading $ENV environment configuration..."

# Load appropriate .env file based on environment
if [[ -f ".env.$ENV" ]]; then
    ENV_FILE=".env.$ENV"
elif [[ -f "../../../.env.$ENV" ]]; then
    ENV_FILE="../../../.env.$ENV"
else
    echo "Warning: Environment file .env.$ENV not found. Using default .env file."
    if [[ -f ".env" ]]; then
        ENV_FILE=".env"
    elif [[ -f "../../../.env" ]]; then
        ENV_FILE="../../../.env"
    else
        echo "Error: No .env file found!"
        exit 1
    fi
fi

# Source environment variables if not in dry run mode
if [[ "$DRY_RUN" == "false" ]]; then
    export $(grep -v '^#' $ENV_FILE | xargs)
    
    # Add additional safety check for production
    if [[ "$ENV" == "production" && "$ACTION" == "downgrade" ]]; then
        read -p "⚠️ You are about to DOWNGRADE the PRODUCTION database. Are you sure? (y/N): " confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            echo "Operation cancelled."
            exit 0
        fi
    fi
fi

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"

# Function to run a migration command
function run_migration_command {
    local db=$1
    local cmd="python -m app.db.commands $ACTION --database $db"
    
    # Add revision parameter if needed
    if [[ "$ACTION" == "upgrade" || "$ACTION" == "downgrade" ]]; then
        cmd="$cmd -r $REVISION"
    fi
    
    # Print command if verbose
    if [[ "$VERBOSE" == "true" || "$DRY_RUN" == "true" ]]; then
        echo "Command: $cmd"
    fi
    
    # Execute command if not in dry run mode
    if [[ "$DRY_RUN" == "false" ]]; then
        if [[ "$VERBOSE" == "true" ]]; then
            $cmd
        else
            $cmd > /dev/null
        fi
    fi
}

# Change to project root directory
cd "$PROJECT_ROOT"

# Run migrations for each database
for db in "${DATABASES[@]}"; do
    echo "Running '$ACTION' on $db database for $ENV environment..."
    
    if [[ "$db" == "timescale" && "$ENV" != "development" ]]; then
        # Special case for TimescaleDB in non-development environments
        # We need to ensure the extension is available
        if [[ "$DRY_RUN" == "false" ]]; then
            # Try to connect and check extension
            if ! python -c "from sqlalchemy import create_engine, text; from app.core.config import get_settings; settings = get_settings(); engine = create_engine(str(settings.db.TIMESCALE_URI)); conn = engine.connect(); result = conn.execute(text(\"SELECT extname FROM pg_extension WHERE extname = 'timescaledb'\")).fetchone(); exit(0 if result else 1)" &>/dev/null; then
                echo "Warning: TimescaleDB extension not detected in $ENV environment."
                echo "Make sure TimescaleDB is installed before running migrations."
                
                if [[ "$ENV" == "production" ]]; then
                    read -p "Continue anyway? (y/N): " confirm
                    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
                        echo "Skipping TimescaleDB migrations."
                        continue
                    fi
                fi
            fi
        fi
    fi
    
    # Run the command
    run_migration_command $db
    
    if [[ "$DRY_RUN" == "false" ]]; then
        echo "✅ Successfully completed '$ACTION' on $db database."
    else
        echo "(Dry run - no changes made)"
    fi
done

echo "Migration deployment completed successfully."