"""Show last run results"""
from core.storage.db import get_db

db = get_db()
runs = db.get_runs(limit=1)

if runs:
    run = runs[0]
    print(f"Run: {run['suite_name']} / {run['model_id']}")
    print(f"Accuracy: {run['accuracy']*100:.0f}%")
    print()
    
    results = db.get_results(run['run_id'])
    for r in results:
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        print(f"  {r['test_id']}: {status}")
