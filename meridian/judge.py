"""
Meridian - LLM Judge
Evaluate model outputs using a judge LLM
"""

from dataclasses import dataclass
from typing import Optional
import json
import re


@dataclass
class JudgeVerdict:
    """Result of LLM judge evaluation"""
    overall_score: float  # 0-100
    passed: bool
    correctness: float  # 0-100
    completeness: float  # 0-100
    format_adherence: float  # 0-100
    no_hallucination: float  # 0-100
    reasoning: str
    issues: list[str]


JUDGE_PROMPT = """You are an expert evaluator. Your task is to evaluate the RESPONSE against the given CONTEXT and REQUIREMENTS.

CONTEXT (source of truth):
{context}

ORIGINAL TASK:
{prompt}

REQUIREMENTS TO CHECK:
{requirements}

RESPONSE TO EVALUATE:
{response}

---

Evaluate the response on these criteria (score 0-100 each):

1. CORRECTNESS: Does the response accurately reflect the context? No made-up facts?
2. COMPLETENESS: Does it address all parts of the task?
3. FORMAT: Does it follow the required format (JSON, length, structure)?
4. NO_HALLUCINATION: Does it avoid adding facts not in the context?

Output your evaluation as valid JSON:
{{
  "correctness": <0-100>,
  "completeness": <0-100>,
  "format_adherence": <0-100>,
  "no_hallucination": <0-100>,
  "reasoning": "<2-3 sentence explanation>",
  "issues": ["<issue 1>", "<issue 2>"]
}}

Only output the JSON, nothing else."""


def evaluate_with_judge(
    context: str,
    prompt: str,
    response: str,
    requirements: str,
    judge_adapter,
    pass_threshold: float = 70.0
) -> JudgeVerdict:
    """
    Evaluate a response using an LLM as judge.
    
    Args:
        context: The context/document provided
        prompt: The original task/prompt
        response: The model's response to evaluate
        requirements: Specific requirements to check
        judge_adapter: Model adapter to use for judging
        pass_threshold: Minimum overall score to pass
        
    Returns:
        JudgeVerdict with scores and reasoning
    """
    from core.model_adapters.base import GenerationConfig
    
    judge_prompt = JUDGE_PROMPT.format(
        context=context[:4000] if context else "(No context provided)",
        prompt=prompt[:1000],
        requirements=requirements if requirements else "(General quality check)",
        response=response[:3000]
    )
    
    config = GenerationConfig(
        temperature=0.0,
        max_tokens=500
    )
    
    try:
        result = judge_adapter.generate(judge_prompt, config)
        
        # Parse JSON from response
        json_match = re.search(r'\{[^{}]*\}', result.output, re.DOTALL)
        if json_match:
            verdict_data = json.loads(json_match.group())
        else:
            # Try to parse the whole output
            verdict_data = json.loads(result.output)
        
        correctness = float(verdict_data.get("correctness", 50))
        completeness = float(verdict_data.get("completeness", 50))
        format_adherence = float(verdict_data.get("format_adherence", 50))
        no_hallucination = float(verdict_data.get("no_hallucination", 50))
        
        # Overall is weighted average
        overall = (
            correctness * 0.35 +
            completeness * 0.25 +
            format_adherence * 0.15 +
            no_hallucination * 0.25
        )
        
        return JudgeVerdict(
            overall_score=overall,
            passed=overall >= pass_threshold,
            correctness=correctness,
            completeness=completeness,
            format_adherence=format_adherence,
            no_hallucination=no_hallucination,
            reasoning=verdict_data.get("reasoning", ""),
            issues=verdict_data.get("issues", [])
        )
        
    except Exception as e:
        return JudgeVerdict(
            overall_score=0,
            passed=False,
            correctness=0,
            completeness=0,
            format_adherence=0,
            no_hallucination=0,
            reasoning=f"Judge evaluation failed: {e}",
            issues=["Judge evaluation error"]
        )


def quick_keyword_check(response: str, keywords: str) -> tuple[bool, list[str]]:
    """
    Quick keyword check for simple validation.
    
    Returns (passed, missing_keywords)
    """
    if not keywords:
        return True, []
    
    keyword_list = [k.strip().lower() for k in keywords.split(",") if k.strip()]
    response_lower = response.lower()
    
    missing = [k for k in keyword_list if k not in response_lower]
    
    return len(missing) == 0, missing
