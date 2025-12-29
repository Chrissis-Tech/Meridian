#!/usr/bin/env python
"""
Example: Evaluate RAG system with policy compliance checks.

This script demonstrates how to evaluate a RAG pipeline for:
- Answer accuracy
- Source citation compliance
- Hallucination detection
- Policy adherence

Adapt for your RAG implementation.
"""

import os
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.types import TestCase, ExpectedOutput, RunResult
from core.scoring import (
    contains_match,
    check_hallucination_heuristics,
    validate_json,
)


def create_rag_test_suite(
    questions: list[dict],
    output_path: str = "suites/rag_policy.jsonl"
):
    """
    Create a test suite from RAG questions.
    
    Args:
        questions: List of {"question": str, "expected_answer": str, "sources": list}
        output_path: Where to save the test suite
    """
    import json
    
    tests = []
    for i, q in enumerate(questions):
        test = {
            "id": f"RAG-{i+1:03d}",
            "prompt": q["question"],
            "expected": {
                "type": "contains",
                "value": q["expected_answer"]
            },
            "metadata": {
                "required_sources": q.get("sources", []),
                "policy": q.get("policy", "standard")
            }
        }
        tests.append(test)
    
    with open(output_path, "w") as f:
        for test in tests:
            f.write(json.dumps(test) + "\n")
    
    print(f"Created {len(tests)} tests at {output_path}")
    return output_path


def evaluate_rag_output(
    question: str,
    rag_output: str,
    expected_answer: str,
    required_sources: list[str],
) -> dict:
    """
    Evaluate a single RAG output for policy compliance.
    
    Returns dict with:
        - answer_correct: bool
        - sources_cited: bool
        - hallucination_risk: float
        - policy_compliant: bool
    """
    # Check answer correctness
    answer_result = contains_match(rag_output, expected_answer)
    
    # Check source citations
    sources_cited = all(
        source.lower() in rag_output.lower()
        for source in required_sources
    )
    
    # Check for hallucination indicators
    hallucination_result = check_hallucination_heuristics(rag_output)
    
    # Policy compliance (stricter)
    policy_compliant = (
        answer_result.passed and
        sources_cited and
        hallucination_result.passed
    )
    
    return {
        "answer_correct": answer_result.passed,
        "sources_cited": sources_cited,
        "hallucination_risk": 1.0 - hallucination_result.score,
        "policy_compliant": policy_compliant,
        "details": {
            "answer_score": answer_result.score,
            "hallucination_details": hallucination_result.details,
        }
    }


def run_rag_evaluation(
    rag_function: Callable[[str], str],
    test_cases: list[dict],
) -> dict:
    """
    Run full RAG evaluation.
    
    Args:
        rag_function: Function that takes question and returns RAG output
        test_cases: List of test case dicts
    
    Returns:
        Aggregate results
    """
    results = []
    
    for test in test_cases:
        output = rag_function(test["question"])
        
        result = evaluate_rag_output(
            question=test["question"],
            rag_output=output,
            expected_answer=test["expected_answer"],
            required_sources=test.get("sources", []),
        )
        result["test_id"] = test.get("id", "unknown")
        results.append(result)
    
    # Aggregate
    n = len(results)
    return {
        "total_tests": n,
        "answer_accuracy": sum(r["answer_correct"] for r in results) / n,
        "citation_rate": sum(r["sources_cited"] for r in results) / n,
        "mean_hallucination_risk": sum(r["hallucination_risk"] for r in results) / n,
        "policy_compliance_rate": sum(r["policy_compliant"] for r in results) / n,
        "results": results,
    }


# Example usage
if __name__ == "__main__":
    # Mock RAG function for demonstration
    def mock_rag(question: str) -> str:
        return f"Based on the documentation, the answer is: example response. (Source: docs.example.com)"
    
    # Example test cases
    test_cases = [
        {
            "id": "RAG-001",
            "question": "What is the API rate limit?",
            "expected_answer": "rate limit",
            "sources": ["docs.example.com"],
        },
        {
            "id": "RAG-002",
            "question": "How do I authenticate?",
            "expected_answer": "API key",
            "sources": ["auth.example.com"],
        },
    ]
    
    results = run_rag_evaluation(mock_rag, test_cases)
    
    print("RAG Evaluation Results:")
    print(f"  Answer Accuracy: {results['answer_accuracy']:.1%}")
    print(f"  Citation Rate: {results['citation_rate']:.1%}")
    print(f"  Hallucination Risk: {results['mean_hallucination_risk']:.1%}")
    print(f"  Policy Compliance: {results['policy_compliance_rate']:.1%}")
