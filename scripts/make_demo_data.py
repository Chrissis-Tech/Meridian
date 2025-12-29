#!/usr/bin/env python
"""
Generate demo data for Meridian.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("Generating demo data for Meridian...")
    
    from core.runner import run_suite
    from core.types import RunConfig
    from reports.build_report import build_html_report, build_markdown_report
    from core.config import REPORTS_DIR
    
    print("\nRunning math_short suite with GPT-2...")
    config = RunConfig(model_id="deepseek_chat", temperature=0.0, max_tokens=50)
    
    try:
        result = run_suite("math_short", "deepseek_chat", run_config=config)
        
        print(f"Completed: {result.passed_tests}/{result.total_tests} passed")
        print(f"Accuracy: {result.accuracy:.1%}")
        
        print("\nGenerating reports...")
        
        html_path = REPORTS_DIR / f"demo_{result.run_id}.html"
        md_path = REPORTS_DIR / f"demo_{result.run_id}.md"
        
        build_html_report(result, html_path)
        build_markdown_report(result, md_path)
        
        print(f"HTML: {html_path}")
        print(f"Markdown: {md_path}")
        
        print("\nDemo data generated successfully.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nEnsure transformers and torch are installed:")
        print("  pip install transformers torch")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
