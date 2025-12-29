#!/bin/bash
# Meridian Quickstart Script

set -e

echo "Meridian Quickstart"
echo "======================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required"
    exit 1
fi

echo "Python found: $(python3 --version)"

# Create virtual environment if needed
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate

# Install
echo "Installing Meridian..."
pip install -e . -q

# Run a quick test
echo ""
echo "Running quick test (math_short with GPT-2)..."
echo "This may take a minute on first run (downloading model)..."
echo ""

python -m core.cli run --suite math_short --model deepseek_chat

echo ""
echo "Quickstart complete."
echo ""
echo "Next steps:"
echo "  1. Start the UI:     make run-ui"
echo "  2. Run more suites:  make run-suite SUITE=instruction_following"
echo "  3. View results:     Open http://localhost:8501"
echo ""
