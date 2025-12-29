"""
Meridian - HTML Report Generator
Creates beautiful HTML reports from run results
"""

from pathlib import Path
from datetime import datetime
from typing import Optional
import json


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meridian Report - {suite_name}</title>
    <style>
        :root {{
            --dark: #111827;
            --gray: #6B7280;
            --light: #F3F4F6;
            --border: #E5E7EB;
            --success: #059669;
            --danger: #DC2626;
            --accent: #6366F1;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #FAFAFA;
            color: var(--dark);
            line-height: 1.6;
            padding: 40px 20px;
        }}
        
        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        
        h1 {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        
        .subtitle {{
            color: var(--gray);
            font-size: 14px;
            margin-bottom: 32px;
        }}
        
        .card {{
            background: white;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        
        .card-title {{
            font-size: 11px;
            font-weight: 600;
            color: var(--gray);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 16px;
        }}
        
        .kpi-row {{
            display: flex;
            gap: 24px;
            flex-wrap: wrap;
        }}
        
        .kpi {{
            flex: 1;
            min-width: 120px;
        }}
        
        .kpi-value {{
            font-size: 36px;
            font-weight: 700;
            line-height: 1;
        }}
        
        .kpi-value.success {{ color: var(--success); }}
        .kpi-value.danger {{ color: var(--danger); }}
        
        .kpi-label {{
            font-size: 12px;
            color: var(--gray);
            margin-top: 4px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            text-align: left;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
        }}
        
        th {{
            font-size: 11px;
            font-weight: 600;
            color: var(--gray);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        td {{
            font-size: 14px;
        }}
        
        .status {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }}
        
        .status.pass {{
            background: #D1FAE5;
            color: var(--success);
        }}
        
        .status.fail {{
            background: #FEE2E2;
            color: var(--danger);
        }}
        
        .ci-bar {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 16px;
        }}
        
        .ci-bar-track {{
            flex: 1;
            height: 8px;
            background: var(--light);
            border-radius: 4px;
            position: relative;
        }}
        
        .ci-bar-fill {{
            position: absolute;
            height: 100%;
            background: var(--accent);
            border-radius: 4px;
            opacity: 0.3;
        }}
        
        .ci-bar-point {{
            position: absolute;
            width: 12px;
            height: 12px;
            background: var(--dark);
            border-radius: 50%;
            top: -2px;
            transform: translateX(-50%);
        }}
        
        .footer {{
            text-align: center;
            color: var(--gray);
            font-size: 12px;
            margin-top: 40px;
        }}
        
        .test-id {{
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 13px;
        }}
        
        .latency {{
            color: var(--gray);
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Meridian Evaluation Report</h1>
        <p class="subtitle">{suite_name} • {model_id} • {timestamp}</p>
        
        <!-- KPIs -->
        <div class="card">
            <div class="card-title">Summary</div>
            <div class="kpi-row">
                <div class="kpi">
                    <div class="kpi-value {accuracy_class}">{accuracy}%</div>
                    <div class="kpi-label">Accuracy</div>
                </div>
                <div class="kpi">
                    <div class="kpi-value">{passed}/{total}</div>
                    <div class="kpi-label">Tests Passed</div>
                </div>
                <div class="kpi">
                    <div class="kpi-value">{latency_p50:.1f}s</div>
                    <div class="kpi-label">p50 Latency</div>
                </div>
                <div class="kpi">
                    <div class="kpi-value">{latency_p95:.1f}s</div>
                    <div class="kpi-label">p95 Latency</div>
                </div>
            </div>
            
            <!-- CI Bar -->
            <div class="ci-bar">
                <span style="font-size: 12px; color: var(--gray);">{ci_lower}%</span>
                <div class="ci-bar-track">
                    <div class="ci-bar-fill" style="left: {ci_left}%; width: {ci_width}%;"></div>
                    <div class="ci-bar-point" style="left: {accuracy}%;"></div>
                </div>
                <span style="font-size: 12px; color: var(--gray);">{ci_upper}%</span>
            </div>
            <p style="text-align: center; font-size: 12px; color: var(--gray); margin-top: 8px;">
                95% Confidence Interval: [{ci_lower}%, {ci_upper}%]
            </p>
        </div>
        
        <!-- Results Table -->
        <div class="card">
            <div class="card-title">Test Results</div>
            <table>
                <thead>
                    <tr>
                        <th>Test ID</th>
                        <th>Status</th>
                        <th>Score</th>
                        <th>Latency</th>
                    </tr>
                </thead>
                <tbody>
                    {results_rows}
                </tbody>
            </table>
        </div>
        
        <!-- Failed Tests -->
        {failed_section}
        
        <div class="footer">
            <p>Generated by Meridian v0.3.0 • {timestamp}</p>
            <p><a href="https://github.com/Chrissis-Tech/Meridian">github.com/Chrissis-Tech/Meridian</a></p>
        </div>
    </div>
</body>
</html>
"""


def generate_html_report(
    run_id: str,
    output_path: Optional[str] = None
) -> str:
    """
    Generate an HTML report for a run.
    
    Args:
        run_id: The run ID to generate report for
        output_path: Optional path to save the report
        
    Returns:
        Path to the generated report
    """
    from core.storage.db import get_db
    import numpy as np
    
    db = get_db()
    run = db.get_run(run_id)
    results = db.get_results(run_id)
    
    if not run:
        raise ValueError(f"Run '{run_id}' not found")
    
    # Calculate stats
    accuracy = run['accuracy'] * 100 if run['accuracy'] else 0
    passed = run['passed_tests']
    failed = run['failed_tests']
    total = passed + failed
    
    ci_lower = run['accuracy_ci_lower'] * 100 if run['accuracy_ci_lower'] else accuracy
    ci_upper = run['accuracy_ci_upper'] * 100 if run['accuracy_ci_upper'] else accuracy
    
    latencies = [r['latency_ms'] / 1000 for r in results if r['latency_ms']]
    latency_p50 = np.percentile(latencies, 50) if latencies else 0
    latency_p95 = np.percentile(latencies, 95) if latencies else 0
    
    # Accuracy class
    accuracy_class = "success" if accuracy >= 70 else "danger" if accuracy < 50 else ""
    
    # Results rows
    results_rows = ""
    for r in results:
        status_class = "pass" if r['passed'] else "fail"
        status_text = "PASS" if r['passed'] else "FAIL"
        latency = r['latency_ms'] / 1000 if r['latency_ms'] else 0
        score = f"{r['score']:.2f}" if r['score'] else "—"
        
        results_rows += f"""
        <tr>
            <td class="test-id">{r['test_id']}</td>
            <td><span class="status {status_class}">{status_text}</span></td>
            <td>{score}</td>
            <td class="latency">{latency:.2f}s</td>
        </tr>
        """
    
    # Failed tests section
    failed_results = [r for r in results if not r['passed']]
    if failed_results:
        failed_section = """
        <div class="card">
            <div class="card-title">Failed Tests Details</div>
        """
        for r in failed_results[:5]:
            output = r.get('output', 'No output')[:200]
            failed_section += f"""
            <div style="margin-bottom: 16px; padding: 12px; background: #FEF2F2; border-radius: 8px;">
                <div style="font-weight: 600; font-family: monospace; margin-bottom: 8px;">{r['test_id']}</div>
                <div style="font-size: 12px; color: var(--gray); white-space: pre-wrap;">{output}...</div>
            </div>
            """
        failed_section += "</div>"
    else:
        failed_section = ""
    
    # Fill template
    html = HTML_TEMPLATE.format(
        suite_name=run['suite_name'],
        model_id=run['model_id'],
        timestamp=run['started_at'][:19],
        accuracy=f"{accuracy:.0f}",
        accuracy_class=accuracy_class,
        passed=passed,
        total=total,
        latency_p50=latency_p50,
        latency_p95=latency_p95,
        ci_lower=f"{ci_lower:.0f}",
        ci_upper=f"{ci_upper:.0f}",
        ci_left=ci_lower,
        ci_width=ci_upper - ci_lower,
        results_rows=results_rows,
        failed_section=failed_section
    )
    
    # Save
    if not output_path:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        output_path = reports_dir / f"{run_id}.html"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    return str(output_path)


def generate_comparison_report(
    run_a_id: str,
    run_b_id: str,
    output_path: Optional[str] = None
) -> str:
    """Generate comparison HTML report for two runs."""
    from core.storage.db import get_db
    from core.runner import SuiteRunner
    
    db = get_db()
    run_a = db.get_run(run_a_id)
    run_b = db.get_run(run_b_id)
    
    runner = SuiteRunner()
    comparison = runner.compare_runs(run_a_id, run_b_id)
    
    acc_a = run_a['accuracy'] * 100 if run_a['accuracy'] else 0
    acc_b = run_b['accuracy'] * 100 if run_b['accuracy'] else 0
    delta = acc_b - acc_a
    delta_class = "success" if delta > 0 else "danger" if delta < 0 else ""
    delta_str = f"+{delta:.0f}%" if delta > 0 else f"{delta:.0f}%"
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Meridian Comparison Report</title>
    <style>
        body {{ font-family: Inter, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }}
        .card {{ background: white; border: 1px solid #E5E7EB; border-radius: 12px; padding: 24px; margin: 20px 0; }}
        .kpi-row {{ display: flex; gap: 24px; }}
        .kpi {{ flex: 1; text-align: center; }}
        .kpi-value {{ font-size: 36px; font-weight: 700; }}
        .kpi-label {{ color: #6B7280; font-size: 12px; }}
        .success {{ color: #059669; }}
        .danger {{ color: #DC2626; }}
    </style>
</head>
<body>
    <h1>Comparison Report</h1>
    <p style="color: #6B7280;">{comparison.suite_name} • {run_a['started_at'][:10]}</p>
    
    <div class="card">
        <div class="kpi-row">
            <div class="kpi">
                <div class="kpi-value">{acc_a:.0f}%</div>
                <div class="kpi-label">A: {comparison.model_a}</div>
            </div>
            <div class="kpi">
                <div class="kpi-value">{acc_b:.0f}%</div>
                <div class="kpi-label">B: {comparison.model_b}</div>
            </div>
            <div class="kpi">
                <div class="kpi-value {delta_class}">{delta_str}</div>
                <div class="kpi-label">Δ Accuracy</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h3>Changes</h3>
        <p><strong>{len(comparison.improvements)}</strong> improved · <strong>{len(comparison.regressions)}</strong> regressed</p>
    </div>
    
    <p style="text-align: center; color: #6B7280; font-size: 12px;">Generated by Meridian v0.3.0</p>
</body>
</html>
"""
    
    if not output_path:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        output_path = reports_dir / f"compare_{run_a_id[:8]}_{run_b_id[:8]}.html"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    return str(output_path)
