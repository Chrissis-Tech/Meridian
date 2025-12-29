"""
Meridian Runner - Test Suite Execution Engine

Executes test suites against models with full metrics.
"""

from datetime import datetime
from typing import Optional, Callable
from tqdm import tqdm

from .types import (
    TestCase, TestSuite, RunResult, SuiteResult, ConsistencyResult,
    ComparisonResult, RunConfig, GenerationResult,
)
from .model_adapters import ModelAdapter, get_adapter, GenerationConfig
from .storage import get_db, load_test_suite, get_artifact_manager
from .storage.jsonl import export_results_jsonl
from .utils import generate_run_id, hash_prompt, hash_result, Timer, now_iso
from .config import config, SUITES_DIR


class SuiteRunner:
    """Executes test suites and collects results."""
    
    def __init__(self, model: Optional[ModelAdapter] = None):
        self.model = model
        self.db = get_db()
        self.artifacts = get_artifact_manager()
    
    def run_suite(
        self,
        suite_path: str,
        model_id: Optional[str] = None,
        run_config: Optional[RunConfig] = None,
        progress_callback: Optional[Callable] = None,
    ) -> SuiteResult:
        """Run a complete test suite."""
        # Load suite
        suite = load_test_suite(suite_path)
        
        # Get model
        if model_id:
            model = get_adapter(model_id)
        elif self.model:
            model = self.model
        else:
            model = get_adapter(config.model.default_model)
        
        run_config = run_config or RunConfig(model_id=model.model_id)
        run_id = generate_run_id()
        
        # Create run in DB
        self.db.create_run(
            run_id=run_id,
            suite_name=suite.name,
            model_id=model.model_id,
            config={"temperature": run_config.temperature, "max_tokens": run_config.max_tokens}
        )
        
        results = []
        passed = 0
        failed = 0
        errors = 0
        total_tokens = 0
        total_cost = 0.0
        latencies = []
        
        # Run each test
        iterator = tqdm(suite.test_cases, desc=f"Running {suite.name}")
        for test in iterator:
            try:
                if test.num_runs > 1:
                    # Consistency test
                    result = self._run_consistency_test(test, model, run_config, run_id)
                    results.append(result)
                else:
                    result = self._run_single_test(test, model, run_config, run_id)
                    results.append(result)
                
                if result.passed:
                    passed += 1
                else:
                    failed += 1
                
                total_tokens += result.tokens_in + result.tokens_out
                total_cost += result.cost_estimate
                latencies.append(result.latency_ms)
                
            except Exception as e:
                errors += 1
                error_result = RunResult(
                    test_id=test.id,
                    model_id=model.model_id,
                    output="",
                    tokens_in=0,
                    tokens_out=0,
                    latency_ms=0,
                    cost_estimate=0,
                    timestamp=now_iso(),
                    run_hash="error",
                    error=str(e),
                )
                results.append(error_result)
            
            if progress_callback:
                progress_callback(len(results), len(suite.test_cases))
        
        # Calculate aggregates
        accuracy = passed / len(suite.test_cases) if suite.test_cases else 0
        mean_latency = sum(latencies) / len(latencies) if latencies else 0
        
        # Bootstrap CI for accuracy
        accuracy_ci = self._bootstrap_ci([1 if r.passed else 0 for r in results])
        
        suite_result = SuiteResult(
            suite_name=suite.name,
            model_id=model.model_id,
            run_id=run_id,
            timestamp=now_iso(),
            total_tests=len(suite.test_cases),
            passed_tests=passed,
            failed_tests=failed,
            error_tests=errors,
            accuracy=accuracy,
            accuracy_ci=accuracy_ci,
            mean_latency_ms=mean_latency,
            total_tokens=total_tokens,
            total_cost=total_cost,
            results=results,
        )
        
        # Complete run in DB
        self.db.complete_run(run_id, suite_result)
        
        return suite_result
    
    def _run_single_test(
        self,
        test: TestCase,
        model: ModelAdapter,
        run_config: RunConfig,
        run_id: str,
    ) -> RunResult:
        """Run a single test case."""
        gen_config = GenerationConfig(
            temperature=run_config.temperature,
            max_tokens=run_config.max_tokens,
            return_attention=run_config.capture_attention,
            return_hidden_states=run_config.capture_hidden_states,
        )
        
        # Generate
        gen_result = model.generate(test.prompt, gen_config)
        
        # Score
        score_result = self._score_output(test, gen_result.output)
        
        result = RunResult(
            test_id=test.id,
            model_id=model.model_id,
            output=gen_result.output,
            tokens_in=gen_result.tokens_in,
            tokens_out=gen_result.tokens_out,
            latency_ms=gen_result.latency_ms,
            cost_estimate=model.estimate_cost(gen_result.tokens_in, gen_result.tokens_out),
            timestamp=now_iso(),
            run_hash=hash_result({"test_id": test.id, "output": gen_result.output}),
            passed=score_result.passed,
            score=score_result.score,
            scoring_method=score_result.method,
            scoring_details=score_result.details,
        )
        
        # Save to DB
        self.db.save_result(run_id, result)
        
        return result
    
    def _run_consistency_test(
        self,
        test: TestCase,
        model: ModelAdapter,
        run_config: RunConfig,
        run_id: str,
    ) -> RunResult:
        """Run consistency test with multiple runs."""
        from .scoring import measure_self_consistency, detect_contradictions
        
        outputs = []
        total_latency = 0
        total_tokens_in = 0
        total_tokens_out = 0
        
        gen_config = GenerationConfig(
            temperature=run_config.consistency_temperature,
            max_tokens=run_config.max_tokens,
        )
        
        for i in range(test.num_runs):
            gen_result = model.generate(test.prompt, gen_config)
            outputs.append(gen_result.output)
            total_latency += gen_result.latency_ms
            total_tokens_in += gen_result.tokens_in
            total_tokens_out += gen_result.tokens_out
        
        # Score consistency
        consistency = measure_self_consistency(outputs)
        contradictions = detect_contradictions(outputs)
        
        passed = consistency.passed and contradictions.passed
        
        result = RunResult(
            test_id=test.id,
            model_id=model.model_id,
            output=outputs[0],  # First output as representative
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            latency_ms=total_latency,
            cost_estimate=model.estimate_cost(total_tokens_in, total_tokens_out),
            timestamp=now_iso(),
            run_hash=hash_result({"test_id": test.id, "outputs": outputs}),
            passed=passed,
            score=consistency.score,
            scoring_method="self_consistency",
            scoring_details={
                "consistency": consistency.details,
                "contradictions": contradictions.details,
                "num_runs": test.num_runs,
                "all_outputs": outputs,
            },
        )
        
        self.db.save_result(run_id, result)
        return result
    
    def _score_output(self, test: TestCase, output: str):
        """Score an output based on test configuration."""
        from .scoring import (
            exact_match, contains_match, numeric_match,
            regex_match, validate_json,
            check_length_constraints, check_word_restrictions,
            detect_hallucination_heuristics, check_refusal_appropriateness,
        )
        
        method = test.scoring.method
        expected = test.expected
        
        if method == "exact_match":
            return exact_match(output, expected.value or "")
        elif method == "contains_match":
            return contains_match(output, expected.value or "")
        elif method == "numeric_match":
            return numeric_match(output, float(expected.value or 0))
        elif method == "regex_match":
            return regex_match(output, expected.pattern or ".*")
        elif method == "json_schema":
            return validate_json(output, expected.schema, expected.required_keys if hasattr(expected, 'required_keys') else None)
        elif method == "length_constraint":
            return check_length_constraints(
                output,
                max_words=expected.max_words if hasattr(expected, 'max_words') else None,
                max_sentences=expected.max_sentences if hasattr(expected, 'max_sentences') else None,
            )
        elif method == "word_restriction":
            return check_word_restrictions(
                output,
                forbidden_words=expected.forbidden_words,
                required_words=expected.required_words,
            )
        elif method == "hallucination_heuristics":
            return detect_hallucination_heuristics(output)
        elif method == "refusal_check":
            return check_refusal_appropriateness(output, should_refuse=True)
        else:
            # Default to contains if expected value exists
            if expected.value:
                return contains_match(output, expected.value)
            return exact_match(output, "")
    
    def _bootstrap_ci(self, values: list, confidence: float = 0.95, n_bootstrap: int = 1000):
        """Calculate bootstrap confidence interval."""
        import numpy as np
        
        if len(values) < 2:
            mean = np.mean(values) if values else 0
            return (mean, mean)
        
        values = np.array(values)
        means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(means, alpha / 2 * 100)
        upper = np.percentile(means, (1 - alpha / 2) * 100)
        
        return (float(lower), float(upper))
    
    def compare_runs(
        self,
        run_a_id: str,
        run_b_id: str,
    ) -> ComparisonResult:
        """Compare two runs."""
        results_a = self.db.get_results(run_a_id)
        results_b = self.db.get_results(run_b_id)
        run_a = self.db.get_run(run_a_id)
        run_b = self.db.get_run(run_b_id)
        
        # Match by test_id
        a_by_id = {r["test_id"]: r for r in results_a}
        b_by_id = {r["test_id"]: r for r in results_b}
        
        common_ids = set(a_by_id.keys()) & set(b_by_id.keys())
        
        regressions = []
        improvements = []
        
        for test_id in common_ids:
            a_passed = a_by_id[test_id]["passed"]
            b_passed = b_by_id[test_id]["passed"]
            
            if a_passed and not b_passed:
                regressions.append(test_id)
            elif not a_passed and b_passed:
                improvements.append(test_id)
        
        accuracy_a = run_a["accuracy"] if run_a else 0
        accuracy_b = run_b["accuracy"] if run_b else 0
        
        return ComparisonResult(
            run_a_id=run_a_id,
            run_b_id=run_b_id,
            model_a=run_a["model_id"] if run_a else "",
            model_b=run_b["model_id"] if run_b else "",
            suite_name=run_a["suite_name"] if run_a else "",
            accuracy_delta=accuracy_b - accuracy_a,
            regressions=regressions,
            improvements=improvements,
        )


def run_suite(
    suite_name: str,
    model_id: str = "deepseek_chat",
    **kwargs
) -> SuiteResult:
    """Convenience function to run a suite."""
    suite_path = SUITES_DIR / f"{suite_name}.jsonl"
    runner = SuiteRunner()
    return runner.run_suite(str(suite_path), model_id, **kwargs)
