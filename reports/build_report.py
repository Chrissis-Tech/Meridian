"""
Meridian Report Builder

Generates HTML and Markdown reports from evaluation results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from jinja2 import Template

from core.types import SuiteResult, ComparisonResult
from core.config import REPORTS_DIR


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meridian Report - {{ suite_name }}</title>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --success: #3fb950;
            --failure: #f85149;
            --warning: #d29922;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { 
            font-size: 2rem; 
            margin-bottom: 1rem;
            color: var(--accent);
        }
        h2 { font-size: 1.5rem; margin: 2rem 0 1rem; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        .metric-card {
            background: var(--bg-secondary);
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #30363d;
        }
        .metric-value { font-size: 2rem; font-weight: bold; color: var(--accent); }
        .metric-label { color: var(--text-secondary); font-size: 0.875rem; }
        .ci-text { font-size: 0.875rem; color: var(--text-secondary); }
        table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
        th, td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d; }
        th { background: var(--bg-secondary); }
        .passed { color: var(--success); }
        .failed { color: var(--failure); }
        .tag { 
            display: inline-block; 
            padding: 0.25rem 0.5rem; 
            background: #30363d; 
            border-radius: 4px; 
            font-size: 0.75rem;
            margin: 0.125rem;
        }
        .footer { margin-top: 3rem; color: var(--text-secondary); font-size: 0.875rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Meridian Report</h1>
        <p><strong>Suite:</strong> {{ suite_name }} | <strong>Model:</strong> {{ model_id }} | <strong>Run ID:</strong> {{ run_id }}</p>
        <p><strong>Generated:</strong> {{ timestamp }}</p>
        
        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="metric-card">
                <div class="metric-value">{{ "%.1f"|format(accuracy * 100) }}%</div>
                <div class="metric-label">Accuracy</div>
                {% if accuracy_ci %}
                <div class="ci-text">95% CI: [{{ "%.1f"|format(accuracy_ci[0] * 100) }}%, {{ "%.1f"|format(accuracy_ci[1] * 100) }}%]</div>
                {% endif %}
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ passed_tests }}/{{ total_tests }}</div>
                <div class="metric-label">Tests Passed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.0f"|format(mean_latency_ms) }}ms</div>
                <div class="metric-label">Mean Latency</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${{ "%.4f"|format(total_cost) }}</div>
                <div class="metric-label">Total Cost</div>
            </div>
        </div>
        
        <h2>Results</h2>
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
                {% for result in results %}
                <tr>
                    <td>{{ result.test_id }}</td>
                    <td class="{{ 'passed' if result.passed else 'failed' }}">{{ 'PASS' if result.passed else 'FAIL' }}</td>
                    <td>{{ "%.2f"|format(result.score) }}</td>
                    <td>{{ "%.0f"|format(result.latency_ms) }}ms</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        {% if failed_tests %}
        <h2>Failed Tests</h2>
        {% for result in failed_tests[:10] %}
        <div class="metric-card" style="margin-bottom: 1rem;">
            <strong>{{ result.test_id }}</strong>
            <p style="color: var(--text-secondary); margin-top: 0.5rem;">{{ result.output[:300] }}...</p>
        </div>
        {% endfor %}
        {% endif %}
        
        <div class="footer">
            <p>Generated by Meridian v0.3.0</p>
        </div>
    </div>
</body>
</html>
"""


def build_html_report(result: SuiteResult, output_path: Optional[Path] = None) -> str:
    """Generate HTML report from suite result."""
    template = Template(HTML_TEMPLATE)
    
    failed_tests = [r for r in result.results if not r.passed]
    
    html = template.render(
        suite_name=result.suite_name,
        model_id=result.model_id,
        run_id=result.run_id,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        accuracy=result.accuracy,
        accuracy_ci=result.accuracy_ci,
        passed_tests=result.passed_tests,
        total_tests=result.total_tests,
        mean_latency_ms=result.mean_latency_ms,
        total_cost=result.total_cost,
        results=result.results,
        failed_tests=failed_tests,
    )
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)
    
    return html


def build_markdown_report(result: SuiteResult, output_path: Optional[Path] = None) -> str:
    """Generate Markdown report from suite result."""
    lines = [
        f"# Meridian Report: {result.suite_name}",
        "",
        f"**Model:** {result.model_id}  ",
        f"**Run ID:** {result.run_id}  ",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Accuracy | {result.accuracy:.1%} |",
    ]
    
    if result.accuracy_ci:
        lines.append(f"| 95% CI | [{result.accuracy_ci[0]:.1%}, {result.accuracy_ci[1]:.1%}] |")
    
    lines.extend([
        f"| Passed | {result.passed_tests}/{result.total_tests} |",
        f"| Mean Latency | {result.mean_latency_ms:.0f}ms |",
        f"| Total Cost | ${result.total_cost:.4f} |",
        "",
        "## Results",
        "",
        "| Test ID | Status | Score | Latency |",
        "|---------|--------|-------|---------|",
    ])
    
    for r in result.results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"| {r.test_id} | {status} | {r.score:.2f} | {r.latency_ms:.0f}ms |")
    
    md = "\n".join(lines)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md)
    
    return md
