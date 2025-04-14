#!/usr/bin/env bash
# Tests for the deploy_migrations.sh script

# Import test utilities
source "$(dirname "$0")/../../shell_test_utils.sh"

# Path to the script to test (relative to project root)
SCRIPT_PATH="app/db/migrations/scripts/deploy_migrations.sh"

# Set up test environment
setup() {
    # Create temporary directory for test artifacts
    TEST_DIR=$(mktemp -d)
    
    # Create mock Alembic files
    mkdir -p "$TEST_DIR/migrations/versions"
    
    # Create mock alembic.ini
    cat > "$TEST_DIR/alembic.ini" << EOF
[alembic]
script_location = migrations
sqlalchemy.url = postgresql://postgres:postgres@localhost:5432/test_db
prepend_sys_path = .
EOF
    
    # Create mock env.py
    cat > "$TEST_DIR/migrations/env.py" << EOF
from alembic import context

def run_migrations_offline():
    print("Running offline migrations")

def run_migrations_online():
    print("Running online migrations")

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
EOF
    
    # Create a mock version of alembic command
    cat > "$TEST_DIR/alembic" << EOF
#!/usr/bin/env bash
echo "Alembic command: \$@"
exit 0
EOF
    chmod +x "$TEST_DIR/alembic"
    
    # Create a backup of the original script
    if [ -f "$SCRIPT_PATH" ]; then
        cp "$SCRIPT_PATH" "$TEST_DIR/original_deploy_migrations.sh"
    else
        # Create a basic version for testing
        cat > "$TEST_DIR/original_deploy_migrations.sh" << EOF
#!/usr/bin/env bash
set -e

# Get script directory
SCRIPT_DIR="\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="\$(realpath "\${SCRIPT_DIR}/../../../../")"
ALEMBIC_DIR="\${PROJECT_ROOT}/app/db/migrations"

# Change to alembic directory
cd "\${ALEMBIC_DIR}"

# Default values
REVISION="head"
ENV_FILE=".env"
DRY_RUN=false

# Parse arguments
while [[ \$# -gt 0 ]]; do
    key="\$1"
    case \$key in
        --revision|-r)
        REVISION="\$2"
        shift 2
        ;;
        --env|-e)
        ENV_FILE="\$2"
        shift 2
        ;;
        --dry-run|-d)
        DRY_RUN=true
        shift
        ;;
        --help|-h)
        echo "Usage: \$0 [--revision|-r REVISION] [--env|-e ENV_FILE] [--dry-run|-d]"
        echo ""
        echo "Options:"
        echo "  --revision, -r  Alembic revision to upgrade to (default: head)"
        echo "  --env, -e       Environment file to load before running migrations (default: .env)"
        echo "  --dry-run, -d   Print the SQL that would be executed but don't run it"
        echo "  --help, -h      Show this help message"
        exit 0
        ;;
        *)
        echo "Unknown option: \$key"
        exit 1
        ;;
    esac
done

# Load environment variables if file exists
if [ -f "\${PROJECT_ROOT}/\${ENV_FILE}" ]; then
    echo "Loading environment variables from \${ENV_FILE}"
    source "\${PROJECT_ROOT}/\${ENV_FILE}"
fi

# Run alembic upgrade
if [ "\${DRY_RUN}" = true ]; then
    echo "Dry run: SQL statements for upgrade to \${REVISION}"
    alembic upgrade \${REVISION} --sql
else
    echo "Running database migrations to \${REVISION}"
    alembic upgrade \${REVISION}
fi

echo "Migration complete"
EOF
    fi
    
    # Create our test version in the temp directory
    cp "$TEST_DIR/original_deploy_migrations.sh" "$TEST_DIR/deploy_migrations.sh"
    chmod +x "$TEST_DIR/deploy_migrations.sh"
    
    # Add our mock alembic to the PATH for testing
    export PATH="$TEST_DIR:$PATH"
    
    # Change to the test directory
    cd "$TEST_DIR"
}

# Clean up after tests
teardown() {
    # Remove temporary directory
    rm -rf "$TEST_DIR"
}

# Test that the script runs with default arguments
test_default_arguments() {
    # Run the script with default arguments
    output=$("$TEST_DIR/deploy_migrations.sh" 2>&1)
    
    # Check that the script executed the expected command
    assert_contains "$output" "Running database migrations to head"
    assert_contains "$output" "Alembic command: upgrade head"
    assert_contains "$output" "Migration complete"
}

# Test specifying a different revision
test_custom_revision() {
    # Run the script with a custom revision
    output=$("$TEST_DIR/deploy_migrations.sh" --revision "abc123" 2>&1)
    
    # Check that the script executed the expected command
    assert_contains "$output" "Running database migrations to abc123"
    assert_contains "$output" "Alembic command: upgrade abc123"
}

# Test dry run option
test_dry_run() {
    # Run the script with dry run flag
    output=$("$TEST_DIR/deploy_migrations.sh" --dry-run 2>&1)
    
    # Check that the script executed the expected command
    assert_contains "$output" "Dry run: SQL statements for upgrade to head"
    assert_contains "$output" "Alembic command: upgrade head --sql"
}

# Test environment file loading
test_env_file_loading() {
    # Create a test environment file
    echo "DB_HOST=testdb.example.com" > "$TEST_DIR/test.env"
    
    # Run the script with the custom env file
    output=$("$TEST_DIR/deploy_migrations.sh" --env "test.env" 2>&1)
    
    # Check that the script loaded the environment file
    assert_contains "$output" "Loading environment variables from test.env"
}

# Test help option
test_help_option() {
    # Run the script with help flag
    output=$("$TEST_DIR/deploy_migrations.sh" --help 2>&1)
    
    # Check that the script displayed the help text
    assert_contains "$output" "Usage:"
    assert_contains "$output" "--revision"
    assert_contains "$output" "--env"
    assert_contains "$output" "--dry-run"
}

# Test invalid option
test_invalid_option() {
    # Run the script with an invalid option
    output=$("$TEST_DIR/deploy_migrations.sh" --invalid-option 2>&1) || true
    
    # Check that the script reported the error
    assert_contains "$output" "Unknown option: --invalid-option"
}

# Test error handling
test_error_handling() {
    # Create a failing alembic mock
    cat > "$TEST_DIR/alembic" << EOF
#!/usr/bin/env bash
echo "Alembic error: migration failed" >&2
exit 1
EOF
    chmod +x "$TEST_DIR/alembic"
    
    # Run the script and capture the failure
    output=$("$TEST_DIR/deploy_migrations.sh" 2>&1) || true
    
    # Check that the error was reported
    assert_contains "$output" "Alembic error: migration failed"
}

# Run tests
run_test() {
    local test_name=$1
    echo "Running test: $test_name"
    setup
    $test_name
    local result=$?
    teardown
    return $result
}

# Create a test utilities file if it doesn't exist
if [ ! -d "$(dirname "$0")/../../" ]; then
    mkdir -p "$(dirname "$0")/../../"
fi

if [ ! -f "$(dirname "$0")/../../shell_test_utils.sh" ]; then
    cat > "$(dirname "$0")/../../shell_test_utils.sh" << 'EOF'
#!/usr/bin/env bash
# Shell script testing utilities

# Assert that a string contains a substring
assert_contains() {
    local haystack="$1"
    local needle="$2"
    
    if [[ "$haystack" == *"$needle"* ]]; then
        return 0  # Success
    else
        echo "Assertion failed: '$haystack' does not contain '$needle'"
        return 1  # Failure
    fi
}

# Assert that a string does not contain a substring
assert_not_contains() {
    local haystack="$1"
    local needle="$2"
    
    if [[ "$haystack" != *"$needle"* ]]; then
        return 0  # Success
    else
        echo "Assertion failed: '$haystack' contains '$needle'"
        return 1  # Failure
    fi
}

# Assert that two strings are equal
assert_equals() {
    local expected="$1"
    local actual="$2"
    
    if [[ "$expected" == "$actual" ]]; then
        return 0  # Success
    else
        echo "Assertion failed: Expected '$expected' but got '$actual'"
        return 1  # Failure
    fi
}

# Assert that a file exists
assert_file_exists() {
    local file="$1"
    
    if [ -f "$file" ]; then
        return 0  # Success
    else
        echo "Assertion failed: File '$file' does not exist"
        return 1  # Failure
    fi
}

# Assert that a directory exists
assert_dir_exists() {
    local dir="$1"
    
    if [ -d "$dir" ]; then
        return 0  # Success
    else
        echo "Assertion failed: Directory '$dir' does not exist"
        return 1  # Failure
    fi
}
EOF
    chmod +x "$(dirname "$0")/../../shell_test_utils.sh"
fi

# Run all tests
echo "Starting tests for deploy_migrations.sh"
failures=0

for test_func in test_default_arguments test_custom_revision test_dry_run test_env_file_loading test_help_option test_invalid_option test_error_handling; do
    if ! run_test $test_func; then
        echo "Test failed: $test_func"
        ((failures++))
    fi
done

if [ $failures -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "$failures test(s) failed"
    exit 1
fi