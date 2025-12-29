#!/usr/bin/env python
"""
Example: Evaluate OpenAI prompt pack for production deployment.

This script demonstrates how to evaluate a set of prompts before deployment.
Copy and adapt for your use case.
"""

import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.runner import run_suite
from core.types import RunConfig
from reports.build_report import build_html_report


def evaluate_prompt_pack(
    suite_path: str,
    model: str = "openai_gpt35",
    output_dir: str = "reports/output"
):
    """
    Evaluate a prompt pack and generate a report.
    
    Args:
        suite_path: Path to your JSONL test suite
        model: Model to evaluate
        output_dir: Directory for output reports
    """
    # Verify API key
    if "openai" in model and not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)
    
    # Configuration for production evaluation
    config = RunConfig(
        model_id=model,
        temperature=0.0,          # Deterministic for reproducibility
        max_tokens=512,           # Adjust based on expected output
        num_consistency_runs=3,   # Check for consistency
    )
    
    print(f"Evaluating: {suite_path}")
    print(f"Model: {model}")
    print("-" * 40)
    
    # Run evaluation
    suite_name = Path(suite_path).stem
    result = run_suite(suite_name, model, run_config=config)
    
    # Print summary
    print(f"\nResults:")
    print(f"  Accuracy: {result.accuracy:.1%}")
    if result.accuracy_ci:
        print(f"  95% CI: [{result.accuracy_ci[0]:.1%}, {result.accuracy_ci[1]:.1%}]")
    print(f"  Passed: {result.passed_tests}/{result.total_tests}")
    print(f"  Mean Latency: {result.mean_latency_ms:.0f}ms")
    print(f"  Total Cost: ${result.total_cost:.4f}")
    
    # Generate report
    output_path = Path(output_dir) / f"{suite_name}_{model}_{result.run_id}.html"
    build_html_report(result, output_path)
    print(f"\nReport saved: {output_path}")
    
    # Return pass/fail based on threshold
    threshold = 0.80  # 80% accuracy required
    if result.accuracy >= threshold:
        print(f"\nPASSED: Accuracy >= {threshold:.0%}")
        return 0
    else:
        print(f"\nFAILED: Accuracy < {threshold:.0%}")
        return 1


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        suite = sys.argv[1]
    else:
        suite = "Meridian_core_50"
    
    model = sys.argv[2] if len(sys.argv) > 2 else "deepseek_chat"
    
    sys.exit(evaluate_prompt_pack(suite, model))
