#!/bin/bash

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install pytest boto3 moto pytest-mock pytest-cov

# Run tests with coverage
echo "Running tests..."
PYTHONPATH=$PYTHONPATH:$(pwd)/live_trading_system pytest live_trading_system/financial_app/tests/consumers/market_data/test_*.py -v --cov=app.consumers.market_data --cov-report=term-missing

# Deactivate virtual environment
deactivate 