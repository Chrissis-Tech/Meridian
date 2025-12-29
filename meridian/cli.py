"""
Meridian CLI - Command Line Interface
"""

import click
import json
from pathlib import Path


@click.group()
@click.version_option(version="1.0.0")
def main():
    """Meridian - LLM Evaluation and Interpretability Framework"""
    pass


@main.command()
@click.option("--suite", "-s", required=True, help="Suite name")
@click.option("--model", "-m", default="deepseek_chat", help="Model ID")
@click.option("--temperature", "-t", default=0.0, help="Temperature")
@click.option("--output", "-o", help="Output file path")
def run(suite: str, model: str, temperature: float, output: str):
    """Run a test suite."""
    from .runner import run_suite
    from .config import SUITES_DIR
    
    click.echo(f"Running suite '{suite}' with model '{model}'...")
    
    suite_path = SUITES_DIR / f"{suite}.jsonl"
    if not suite_path.exists():
        click.echo(f"Suite not found: {suite_path}", err=True)
        return
    
    from .types import RunConfig
    config = RunConfig(model_id=model, temperature=temperature)
    
    result = run_suite(suite, model, run_config=config)
    
    click.echo(f"\nCompleted: {result.passed_tests}/{result.total_tests} passed")
    click.echo(f"  Accuracy: {result.accuracy:.1%}")
    if result.accuracy_ci:
        click.echo(f"  95% CI: [{result.accuracy_ci[0]:.1%}, {result.accuracy_ci[1]:.1%}]")
    click.echo(f"  Mean Latency: {result.mean_latency_ms:.0f}ms")
    
    if output:
        from .storage.jsonl import export_results_jsonl
        export_results_jsonl(result, output)
        click.echo(f"  Saved to: {output}")


@main.command()
@click.option("--baseline", "-b", required=True, help="Baseline JSON file")
@click.option("--suite", "-s", help="Specific suite to check")
@click.option("--model", "-m", default="deepseek_chat", help="Model ID")
def check(baseline: str, suite: str, model: str):
    """Validate against baseline thresholds."""
    from .runner import run_suite
    from .storage.db import get_db
    
    baseline_path = Path(baseline)
    if not baseline_path.exists():
        click.echo(f"Baseline not found: {baseline}", err=True)
        return
    
    with open(baseline_path) as f:
        baselines = json.load(f)
    
    thresholds = baselines.get("thresholds", {})
    suites_to_check = [suite] if suite else list(thresholds.keys())
    
    all_passed = True
    for suite_name in suites_to_check:
        if suite_name not in thresholds:
            continue
        
        click.echo(f"Checking {suite_name}...")
        result = run_suite(suite_name, model)
        
        suite_thresh = thresholds[suite_name]
        violations = []
        
        if "accuracy_min" in suite_thresh and result.accuracy < suite_thresh["accuracy_min"]:
            violations.append(f"accuracy {result.accuracy:.1%} < {suite_thresh['accuracy_min']:.1%}")
        
        if "latency_max_ms" in suite_thresh and result.mean_latency_ms > suite_thresh["latency_max_ms"]:
            violations.append(f"latency {result.mean_latency_ms:.0f}ms > {suite_thresh['latency_max_ms']}ms")
        
        if violations:
            click.echo(f"  FAILED: {', '.join(violations)}")
            all_passed = False
        else:
            click.echo(f"  PASSED")
    
    if all_passed:
        click.echo("\nAll checks passed.")
    else:
        click.echo("\nSome checks failed.", err=True)
        raise SystemExit(1)


@main.command()
@click.option("--run-a", required=True, help="First run ID")
@click.option("--run-b", required=True, help="Second run ID")
def compare(run_a: str, run_b: str):
    """Compare two runs."""
    from .runner import SuiteRunner
    
    runner = SuiteRunner()
    result = runner.compare_runs(run_a, run_b)
    
    click.echo(f"\nComparison: {run_a} vs {run_b}")
    click.echo(f"Accuracy delta: {result.accuracy_delta:+.1%}")
    click.echo(f"Regressions: {len(result.regressions)}")
    click.echo(f"Improvements: {len(result.improvements)}")
    
    if result.regressions:
        click.echo(f"\nRegressed tests: {', '.join(result.regressions[:5])}")


@main.command()
@click.option("--id", required=True, help="Golden run ID (e.g., 2025-12-28_deepseek_chat)")
def reproduce(id: str):
    """Reproduce a golden run."""
    from pathlib import Path
    
    golden_dir = Path("golden_runs") / id
    if not golden_dir.exists():
        click.echo(f"Golden run not found: {id}", err=True)
        click.echo("Available golden runs:")
        for d in Path("golden_runs").iterdir():
            if d.is_dir():
                click.echo(f"  - {d.name}")
        return
    
    # Load golden run config
    summary_path = golden_dir / "summary.json"
    if not summary_path.exists():
        click.echo(f"Summary not found: {summary_path}", err=True)
        return
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    suite_name = summary.get("suite_name", "Meridian_core_50")
    model_id = summary.get("model_id", "deepseek_chat")
    config = summary.get("configuration", {})
    
    click.echo(f"Reproducing golden run: {id}")
    click.echo(f"  Suite: {suite_name}")
    click.echo(f"  Model: {model_id}")
    click.echo(f"  Config: {config}")
    
    from .runner import run_suite
    from .types import RunConfig
    
    run_config = RunConfig(
        model_id=model_id,
        temperature=config.get("temperature", 0.0),
        max_tokens=config.get("max_tokens", 256),
    )
    
    result = run_suite(suite_name, model_id, run_config=run_config)
    
    click.echo(f"\nNew run completed:")
    click.echo(f"  Accuracy: {result.accuracy:.1%}")
    click.echo(f"  Passed: {result.passed_tests}/{result.total_tests}")
    
    # Compare with golden
    golden_acc = summary.get("results", {}).get("accuracy", 0)
    delta = result.accuracy - golden_acc
    
    click.echo(f"\nComparison with golden run:")
    click.echo(f"  Golden accuracy: {golden_acc:.1%}")
    click.echo(f"  Current accuracy: {result.accuracy:.1%}")
    click.echo(f"  Delta: {delta:+.1%}")
    
    if abs(delta) <= 0.05:
        click.echo("\nResult: REPRODUCIBLE (within 5% tolerance)")
    else:
        click.echo("\nResult: VARIANCE DETECTED (check environment)")


@main.command()
def list_suites():
    """List available test suites."""
    from .config import SUITES_DIR
    
    click.echo("Available suites:")
    for path in sorted(SUITES_DIR.glob("*.jsonl")):
        click.echo(f"  - {path.stem}")


@main.command()
def list_models():
    """List available models."""
    from .config import get_available_models
    
    models = get_available_models()
    click.echo("Available models:")
    for model_id, info in models.items():
        status = "[available]" if info.get("available") else "[requires API key]"
        click.echo(f"  {model_id}: {info.get('display_name', model_id)} {status}")


@main.command()
@click.option("--run-id", "-r", required=True, help="Run ID to generate report for")
@click.option("--output", "-o", help="Output file path")
@click.option("--open", "open_browser", is_flag=True, help="Open in browser after generating")
def report(run_id: str, output: str, open_browser: bool):
    """Generate HTML report for a run."""
    from .reports import generate_html_report
    
    click.echo(f"Generating report for run '{run_id}'...")
    
    try:
        report_path = generate_html_report(run_id, output)
        click.echo(f"Report saved to: {report_path}")
        
        if open_browser:
            import webbrowser
            webbrowser.open(f"file://{Path(report_path).absolute()}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", "-p", default=8000, help="Port to bind")
def serve(host: str, port: int):
    """Start the REST API server."""
    try:
        from api.server import start_server
        click.echo(f"Starting Meridian API on http://{host}:{port}")
        click.echo(f"Docs: http://{host}:{port}/docs")
        start_server(host, port)
    except ImportError:
        click.echo("FastAPI not installed. Run: pip install fastapi uvicorn", err=True)


if __name__ == "__main__":
    main()

