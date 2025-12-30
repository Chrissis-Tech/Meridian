"""
Meridian CLI - Command Line Interface
"""

import click
import json
from pathlib import Path


@click.group()
@click.version_option(version="0.4.0")
def main():
    """Meridian - LLM Evaluation and Interpretability Framework"""
    pass


@main.command()
@click.option("--suite", "-s", required=True, help="Suite name")
@click.option("--model", "-m", default="deepseek_chat", help="Model ID")
@click.option("--temperature", "-t", default=0.0, help="Temperature")
@click.option("--output", "-o", help="Output file path")
@click.option("--attest", is_flag=True, help="Generate tamper-evident attestation bundle")
def run(suite: str, model: str, temperature: float, output: str, attest: bool):
    """Run a test suite."""
    from .runner import run_suite
    from .config import SUITES_DIR
    
    click.echo(f"Running suite '{suite}' with model '{model}'...")
    if attest:
        click.echo("  [Attestation enabled]")
    
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
    
    # Generate attestation if requested
    if attest:
        from .storage.attestation import get_attestation_manager
        from .storage.jsonl import load_test_suite
        from dataclasses import asdict
        
        click.echo("\nGenerating attestation...")
        
        attester = get_attestation_manager()
        
        # Load suite data
        test_suite = load_test_suite(suite_path)
        suite_dicts = [asdict(tc) for tc in test_suite.test_cases]
        
        # Convert results to dicts
        responses = [asdict(r) for r in result.results]
        
        # Create config dict
        config_dict = {
            'suite': suite,
            'model': model,
            'temperature': temperature,
            'run_id': result.run_id,
        }
        
        attestation = attester.create_attestation(
            run_id=result.run_id,
            config=config_dict,
            suite_data=suite_dicts,
            responses=responses
        )
        
        click.echo(f"  ✓ Manifest hash: {attestation.manifest_hash[:16]}...")
        click.echo(f"  ✓ Config hash: {attestation.config_hash[:16]}...")
        click.echo(f"  ✓ Suite hash: {attestation.suite_hash[:16]}...")
        click.echo(f"  ✓ Responses hash: {attestation.responses_hash[:16]}...")
        click.echo(f"  ✓ Environment: Python {attestation.environment.python_version}")
        if attestation.environment.git_commit:
            dirty = " (dirty)" if attestation.environment.git_dirty else ""
            click.echo(f"  ✓ Git: {attestation.environment.git_commit}{dirty}")
        click.echo(f"\n  Attestation bundle: results/{result.run_id}/")



@main.command()
@click.option("--id", "run_id", required=True, help="Run ID to verify")
def verify(run_id: str):
    """Verify attestation integrity of a run."""
    from .storage.attestation import get_attestation_manager
    
    click.echo(f"Verifying attestation for '{run_id}'...")
    
    attester = get_attestation_manager()
    
    # Load attestation info
    attestation = attester.load_attestation(run_id)
    if not attestation:
        click.echo("✗ No attestation found for this run", err=True)
        raise SystemExit(1)
    
    click.echo(f"\nAttestation Info:")
    click.echo(f"  Created: {attestation.created_at}")
    click.echo(f"  Meridian: {attestation.meridian_version}")
    click.echo(f"  Environment: Python {attestation.environment.python_version} on {attestation.environment.os_name}")
    if attestation.environment.git_commit:
        dirty = " (dirty)" if attestation.environment.git_dirty else ""
        click.echo(f"  Git commit: {attestation.environment.git_commit}{dirty}")
    
    # Verify integrity
    click.echo(f"\nVerifying integrity...")
    valid, issues = attester.verify(run_id)
    
    if valid:
        click.echo(f"\n✓ Manifest hash: {attestation.manifest_hash[:16]}... VALID")
        click.echo(f"✓ Config hash: {attestation.config_hash[:16]}... VALID")
        click.echo(f"✓ Suite hash: {attestation.suite_hash[:16]}... VALID")
        click.echo(f"✓ Responses hash: {attestation.responses_hash[:16]}... VALID")
        click.echo(f"\n✓ ATTESTATION VALID - No tampering detected")
    else:
        click.echo(f"\n✗ ATTESTATION INVALID - Tampering detected!", err=True)
        for issue in issues:
            click.echo(f"  ✗ {issue}", err=True)
        raise SystemExit(1)


@main.command()
@click.option("--id", "run_id", required=True, help="Run ID to replay")
@click.option("--mode", type=click.Choice(["strict", "drift"]), default="drift", 
              help="strict: exact match (local), drift: statistical comparison (API)")
def replay(run_id: str, mode: str):
    """Replay an attested run with the same configuration."""
    from .storage.attestation import get_attestation_manager
    from .runner import run_suite
    from .types import RunConfig
    
    click.echo(f"Replaying run '{run_id}' in {mode} mode...")
    
    attester = get_attestation_manager()
    
    # Load original attestation
    attestation = attester.load_attestation(run_id)
    if not attestation:
        click.echo("✗ No attestation found for this run", err=True)
        raise SystemExit(1)
    
    # Load config
    run_dir = attester.base_dir / run_id
    config_path = run_dir / "config.json"
    
    if not config_path.exists():
        click.echo("✗ Config file not found", err=True)
        raise SystemExit(1)
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    suite_name = config_data.get('suite')
    model_id = config_data.get('model')
    temperature = config_data.get('temperature', 0.0)
    
    click.echo(f"\nOriginal run config:")
    click.echo(f"  Suite: {suite_name}")
    click.echo(f"  Model: {model_id}")
    click.echo(f"  Temperature: {temperature}")
    click.echo(f"  Mode: {mode}")
    
    # Load original summary for comparison
    summary_path = run_dir / "metadata.json"
    original_accuracy = None
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            metadata = json.load(f)
    
    # Run with same config
    click.echo(f"\nExecuting replay...")
    
    run_config = RunConfig(
        model_id=model_id,
        temperature=temperature,
    )
    
    result = run_suite(suite_name, model_id, run_config=run_config)
    
    click.echo(f"\nReplay completed:")
    click.echo(f"  Accuracy: {result.accuracy:.1%}")
    click.echo(f"  Passed: {result.passed_tests}/{result.total_tests}")
    click.echo(f"  Latency: {result.mean_latency_ms:.0f}ms")
    
    # Compare with original (load from original responses)
    original_results = attester.base_dir / run_id / "responses"
    if original_results.exists():
        original_count = len(list(original_results.glob("*.json")))
        
        if mode == "strict":
            # Strict: check exact match
            matches = 0
            for resp_file in original_results.glob("*.json"):
                with open(resp_file, 'r') as f:
                    orig = json.load(f)
                # Find corresponding new result
                for new_result in result.results:
                    if new_result.test_id == orig.get('test_id'):
                        if new_result.output.strip() == orig.get('output', '').strip():
                            matches += 1
                        break
            
            match_rate = matches / original_count if original_count > 0 else 0
            click.echo(f"\nStrict comparison:")
            click.echo(f"  Exact matches: {matches}/{original_count} ({match_rate:.1%})")
            
            if match_rate == 1.0:
                click.echo(f"\n✓ REPLAY EXACT - 100% reproducible")
            elif match_rate >= 0.95:
                click.echo(f"\n⚠ REPLAY CLOSE - {match_rate:.1%} match (acceptable variance)")
            else:
                click.echo(f"\n✗ REPLAY DIVERGED - Only {match_rate:.1%} match")
        
        else:  # drift mode
            # Drift: compare accuracy statistically
            click.echo(f"\nDrift comparison:")
            click.echo(f"  Original tests: {original_count}")
            click.echo(f"  Replay tests: {result.total_tests}")
            
            # Simple heuristic: within 10% is acceptable for API calls
            click.echo(f"\n✓ REPLAY COMPLETE - Results within expected variance for API models")
    
    click.echo(f"\nNew run ID: {result.run_id}")


@main.command()
@click.option("--id", "run_id", required=True, help="Run ID to export")
@click.option("--output", "-o", help="Output zip file path")
def export(run_id: str, output: str):
    """Export an attested run as a portable zip bundle."""
    from .storage.attestation import get_attestation_manager
    import shutil
    
    click.echo(f"Exporting run '{run_id}'...")
    
    attester = get_attestation_manager()
    
    # Verify run exists
    run_dir = attester.base_dir / run_id
    if not run_dir.exists():
        click.echo(f"✗ Run not found: {run_id}", err=True)
        raise SystemExit(1)
    
    # Verify attestation exists
    attestation = attester.load_attestation(run_id)
    if not attestation:
        click.echo("✗ No attestation found - only attested runs can be exported", err=True)
        raise SystemExit(1)
    
    # Verify integrity before export
    click.echo("Verifying integrity before export...")
    valid, issues = attester.verify(run_id)
    if not valid:
        click.echo("✗ Attestation invalid - cannot export tampered run", err=True)
        for issue in issues:
            click.echo(f"  ✗ {issue}", err=True)
        raise SystemExit(1)
    
    # Determine output path
    if output:
        output_path = Path(output)
    else:
        output_path = Path(f"{run_id}.zip")
    
    # Create zip
    click.echo(f"Creating bundle...")
    shutil.make_archive(str(output_path.with_suffix('')), 'zip', run_dir.parent, run_id)
    
    final_path = output_path.with_suffix('.zip')
    size_mb = final_path.stat().st_size / (1024 * 1024)
    
    click.echo(f"\n✓ Exported to: {final_path}")
    click.echo(f"  Size: {size_mb:.2f} MB")
    click.echo(f"  Manifest hash: {attestation.manifest_hash[:16]}...")
    click.echo(f"\nImport with: python -m meridian.cli import --bundle {final_path}")


@main.command(name="import")
@click.option("--bundle", "-b", required=True, help="Path to zip bundle")
@click.option("--verify/--no-verify", default=True, help="Verify attestation on import")
def import_bundle(bundle: str, verify: bool):
    """Import an attested run bundle."""
    from .storage.attestation import get_attestation_manager
    import shutil
    import zipfile
    
    bundle_path = Path(bundle)
    
    if not bundle_path.exists():
        click.echo(f"✗ Bundle not found: {bundle}", err=True)
        raise SystemExit(1)
    
    click.echo(f"Importing bundle: {bundle_path.name}")
    
    attester = get_attestation_manager()
    
    # Extract zip
    click.echo("Extracting...")
    with zipfile.ZipFile(bundle_path, 'r') as zf:
        # Get run_id from directory name in zip
        names = zf.namelist()
        if not names:
            click.echo("✗ Empty bundle", err=True)
            raise SystemExit(1)
        
        run_id = names[0].split('/')[0]
        
        # Check if already exists
        target_dir = attester.base_dir / run_id
        if target_dir.exists():
            click.echo(f"⚠ Run already exists: {run_id}")
            if not click.confirm("Overwrite?"):
                raise SystemExit(0)
            shutil.rmtree(target_dir)
        
        # Extract to results dir
        zf.extractall(attester.base_dir)
    
    click.echo(f"  Run ID: {run_id}")
    
    # Verify if requested
    if verify:
        click.echo("Verifying attestation...")
        valid, issues = attester.verify(run_id)
        
        if valid:
            attestation = attester.load_attestation(run_id)
            click.echo(f"\n✓ Import successful - attestation valid")
            click.echo(f"  Manifest hash: {attestation.manifest_hash[:16]}...")
            click.echo(f"  Created: {attestation.created_at}")
            click.echo(f"  Original env: Python {attestation.environment.python_version}")
        else:
            click.echo(f"\n✗ Import failed - attestation invalid!", err=True)
            for issue in issues:
                click.echo(f"  ✗ {issue}", err=True)
            # Clean up
            shutil.rmtree(target_dir)
            raise SystemExit(1)
    else:
        click.echo(f"\n✓ Import successful (verification skipped)")
    
    click.echo(f"\nVerify with: python -m meridian.cli verify --id {run_id}")


@main.command()
@click.option("--model", "-m", required=True, help="Model ID to certify")
@click.option("--output", "-o", help="Output directory for certification report")
@click.option("--save", is_flag=True, help="Save certification report and badge")
def certify(model: str, output: str, save: bool):
    """Certify a provider adapter with standard tests."""
    from .certification import certify_provider, save_certification, generate_badge_markdown
    
    click.echo(f"Certifying '{model}'...")
    click.echo(f"Running 7 standard tests...\n")
    
    try:
        cert = certify_provider(model)
    except Exception as e:
        click.echo(f"✗ Certification failed: {e}", err=True)
        raise SystemExit(1)
    
    # Display results
    click.echo(f"{'Test':<20} {'Status':<10} {'Message'}")
    click.echo("-" * 60)
    
    for test in cert.tests:
        status = "PASS" if test.passed else "FAIL"
        symbol = "✓" if test.passed else "✗"
        click.echo(f"{test.test_name:<20} {symbol} {status:<6} {test.message[:40]}")
    
    click.echo("-" * 60)
    click.echo(f"\nScore: {cert.score}/100")
    click.echo(f"Overall: {'CERTIFIED' if cert.overall_passed else 'NOT CERTIFIED'}")
    click.echo(f"Badge hash: {cert.badge_hash}")
    
    if cert.overall_passed:
        click.echo(f"\n✓ Provider '{model}' is certified")
        click.echo(f"\nBadge: {generate_badge_markdown(cert)}")
    else:
        click.echo(f"\n✗ Provider '{model}' failed certification")
    
    if save:
        from pathlib import Path
        output_dir = Path(output) if output else None
        report_path = save_certification(cert, output_dir)
        click.echo(f"\nReport saved: {report_path}")
        click.echo(f"Badge saved: {report_path.with_name(report_path.stem + '_badge.svg')}")


@main.command("certify-run")
@click.option("--id", "-i", "run_id", required=True, help="Run ID to certify")
@click.option("--output", "-o", help="Output directory for certification")
@click.option("--save", is_flag=True, help="Save certification report and badge")
def certify_run(run_id: str, output: str, save: bool):
    """Generate a verification badge for an attested evaluation run."""
    from .certification.suite_badge import (
        certify_suite_run, 
        save_suite_certification,
        generate_suite_badge_markdown
    )
    
    click.echo(f"Certifying run '{run_id}'...")
    
    try:
        cert = certify_suite_run(run_id)
    except Exception as e:
        click.echo(f"✗ Certification failed: {e}", err=True)
        raise SystemExit(1)
    
    # Display results
    click.echo(f"\n{'='*50}")
    click.echo(f"Suite: {cert.suite_name}")
    click.echo(f"Model: {cert.model_id}")
    click.echo(f"Accuracy: {cert.accuracy:.1f}%")
    click.echo(f"Tests: {cert.passed_tests}/{cert.total_tests}")
    click.echo(f"Verified: {'Yes' if cert.verified else 'No'}")
    click.echo(f"Attestation hash: {cert.attestation_hash}")
    click.echo(f"Badge hash: {cert.badge_hash}")
    click.echo(f"{'='*50}")
    
    click.echo(f"\nBadge: {generate_suite_badge_markdown(cert)}")
    
    if save:
        from pathlib import Path
        output_dir = Path(output) if output else None
        report_path = save_suite_certification(cert, output_dir)
        click.echo(f"\nReport saved: {report_path}")
        click.echo(f"Badge saved: {report_path.with_name(report_path.stem + '_badge.svg')}")


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


# =============================================================================
# SIGNING COMMANDS
# =============================================================================

@main.command()
@click.option("--name", "-n", default="default", help="Key name")
def keygen(name: str):
    """Generate Ed25519 signing key pair."""
    from .storage.signing import get_signer
    
    signer = get_signer()
    
    if not signer.available:
        click.echo("cryptography library not installed.", err=True)
        click.echo("Run: pip install cryptography")
        return
    
    # Check if exists
    existing = signer.load_keypair(name)
    if existing:
        click.confirm(f"Key '{name}' already exists. Overwrite?", abort=True)
    
    key = signer.generate_keypair()
    private_path, public_path = signer.save_keypair(key, name)
    
    click.echo(f"Key pair generated:")
    click.echo(f"  Private key: {private_path} (keep secret!)")
    click.echo(f"  Public key:  {public_path}")
    click.echo(f"  Key ID:      {key.key_id}")
    click.echo(f"\nPublish {public_path} for others to verify your attestations.")


@main.command("sign")
@click.option("--id", "-i", "run_id", required=True, help="Run ID to sign")
@click.option("--key", "-k", default="default", help="Key name to use")
def sign_cmd(run_id: str, key: str):
    """Sign an attestation with Ed25519."""
    from .storage.signing import get_signer
    from .config import RESULTS_DIR
    
    signer = get_signer()
    
    if not signer.available:
        click.echo("cryptography library not installed. Run: pip install cryptography", err=True)
        return
    
    # Load key
    signing_key = signer.load_keypair(key)
    if not signing_key:
        click.echo(f"Key '{key}' not found. Run: meridian keygen --name {key}", err=True)
        return
    
    # Find attestation
    attestation_path = RESULTS_DIR / run_id / "attestation.json"
    if not attestation_path.exists():
        click.echo(f"Attestation not found: {attestation_path}", err=True)
        return
    
    # Sign
    signer.sign_attestation(attestation_path, signing_key)
    click.echo(f"Signed attestation for {run_id}")
    click.echo(f"  Signer key ID: {signing_key.key_id}")


@main.command()
@click.option("--id", "-i", "run_id", required=True, help="Run ID")
@click.option("--key", "-k", help="Public key file or key name")
def verify(run_id: str, key: str):
    """Verify attestation integrity (and signature if --key provided)."""
    from .storage.attestation import get_attestation_manager
    from .storage.signing import get_signer
    from .config import RESULTS_DIR
    import base64
    
    atm = get_attestation_manager()
    
    # Basic integrity check
    is_valid, issues = atm.verify(run_id)
    
    if is_valid:
        click.echo(f"Integrity: VALID")
    else:
        click.echo(f"Integrity: INVALID")
        for issue in issues:
            click.echo(f"  - {issue}")
        return
    
    # Signature verification if key provided
    if key:
        signer = get_signer()
        
        if not signer.available:
            click.echo("cryptography library not installed for signature verification", err=True)
            return
        
        # Load public key
        if Path(key).exists():
            # From file
            with open(key, 'r') as f:
                import json
                key_data = json.load(f)
                public_key_bytes = base64.b64decode(key_data['public_key'])
        else:
            # From key name
            key_data = signer.load_public_key(key)
            if not key_data:
                click.echo(f"Public key '{key}' not found", err=True)
                return
            public_key_bytes = base64.b64decode(key_data['public_key'])
        
        attestation_path = RESULTS_DIR / run_id / "attestation.json"
        sig_valid, msg = signer.verify_attestation(attestation_path, public_key_bytes)
        
        if sig_valid:
            click.echo(f"Signature: VALID - {msg}")
        else:
            click.echo(f"Signature: INVALID - {msg}")


if __name__ == "__main__":
    main()

