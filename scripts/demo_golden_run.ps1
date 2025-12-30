# Meridian Golden Run Demo (PowerShell)
# 
# Demonstrates the complete attestation workflow:
# 1. Run evaluation with attestation
# 2. Verify the attestation
# 3. Sign (optional)
# 4. Replay with drift detection

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Meridian Golden Run Demo" -ForegroundColor Cyan
Write-Host "============================================"
Write-Host ""

# Check for API key
if (-not $env:DEEPSEEK_API_KEY) {
    Write-Host "ERROR: DEEPSEEK_API_KEY not set" -ForegroundColor Red
    Write-Host "Get a free key at: https://platform.deepseek.com/"
    Write-Host 'Then run: $env:DEEPSEEK_API_KEY = "your_key"'
    exit 1
}

Write-Host "Step 1: Run evaluation with attestation" -ForegroundColor Yellow
Write-Host "----------------------------------------"
python -m meridian.cli run --suite edge_cases --model deepseek_chat --attest

# Get the latest run ID
$RUN_ID = (Get-ChildItem data\results -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name
Write-Host ""
Write-Host "Run ID: $RUN_ID" -ForegroundColor Green
Write-Host ""

Write-Host "Step 2: Verify attestation integrity" -ForegroundColor Yellow
Write-Host "------------------------------------"
python -m meridian.cli verify --id $RUN_ID

Write-Host ""
Write-Host "Step 3: Replay with drift detection" -ForegroundColor Yellow
Write-Host "------------------------------------"
python -m meridian.cli replay --id $RUN_ID --mode drift

Write-Host ""
Write-Host "Step 4: Generate verification badge" -ForegroundColor Yellow
Write-Host "------------------------------------"
python -m meridian.cli certify-run --id $RUN_ID --save

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Demo complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Artifacts created:"
Write-Host "  - data\results\$RUN_ID\attestation.json"
Write-Host "  - data\results\$RUN_ID\manifest.json"
Write-Host "  - suite_certifications\*.json"
Write-Host "  - suite_certifications\*.svg"
Write-Host ""
Write-Host "To sign (requires 'pip install cryptography'):"
Write-Host "  python -m meridian.cli keygen"
Write-Host "  python -m meridian.cli sign --id $RUN_ID"
Write-Host "============================================" -ForegroundColor Cyan
