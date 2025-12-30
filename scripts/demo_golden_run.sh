#!/bin/bash
# =============================================================================
# Meridian Golden Run Demo
# 
# This script demonstrates the complete attestation workflow:
# 1. Run an evaluation with attestation
# 2. Verify the attestation
# 3. Sign the attestation (optional, requires cryptography)
# 4. Replay with drift detection
# =============================================================================

set -e

echo "============================================"
echo "Meridian Golden Run Demo"
echo "============================================"
echo ""

# Check for API key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "ERROR: DEEPSEEK_API_KEY not set"
    echo "Get a free key at: https://platform.deepseek.com/"
    echo "Then run: export DEEPSEEK_API_KEY=your_key"
    exit 1
fi

echo "Step 1: Run evaluation with attestation"
echo "----------------------------------------"
python -m meridian.cli run --suite edge_cases --model deepseek_chat --attest

# Get the latest run ID
RUN_ID=$(ls -t data/results/ | head -1)
echo ""
echo "Run ID: $RUN_ID"
echo ""

echo "Step 2: Verify attestation integrity"
echo "------------------------------------"
python -m meridian.cli verify --id $RUN_ID

echo ""
echo "Step 3: Replay with drift detection"
echo "------------------------------------"
python -m meridian.cli replay --id $RUN_ID --mode drift

echo ""
echo "Step 4: Generate verification badge"
echo "------------------------------------"
python -m meridian.cli certify-run --id $RUN_ID --save

echo ""
echo "============================================"
echo "Demo complete!"
echo ""
echo "Artifacts created:"
echo "  - data/results/$RUN_ID/attestation.json"
echo "  - data/results/$RUN_ID/manifest.json"
echo "  - suite_certifications/*.json"
echo "  - suite_certifications/*.svg"
echo ""
echo "To sign (requires 'pip install cryptography'):"
echo "  python -m meridian.cli keygen"
echo "  python -m meridian.cli sign --id $RUN_ID"
echo "============================================"
