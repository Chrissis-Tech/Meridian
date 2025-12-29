"""
Meridian Storage - JSONL Operations

Handles JSONL file operations for test suites and result exports.
"""

import json
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from ..types import (
    TestCase,
    TestSuite,
    ExpectedOutput,
    ScoringConfig,
    TaskType,
    RunResult,
    SuiteResult,
)


class JSONLReader:
    """Reader for JSONL files."""
    
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
    
    def __iter__(self) -> Iterator[dict]:
        """Iterate over lines in the JSONL file."""
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    
    def read_all(self) -> list[dict]:
        """Read all items from the file."""
        return list(self)
    
    def count(self) -> int:
        """Count items in the file."""
        return sum(1 for _ in self)


class JSONLWriter:
    """Writer for JSONL files."""
    
    def __init__(self, path: Union[str, Path], mode: str = 'w'):
        self.path = Path(path)
        self.mode = mode
        self.path.parent.mkdir(parents=True, exist_ok=True)
    
    def write(self, item: dict) -> None:
        """Write a single item to the file."""
        with open(self.path, self.mode, encoding='utf-8') as f:
            f.write(json.dumps(item) + '\n')
    
    def write_all(self, items: list[dict]) -> None:
        """Write all items to the file."""
        with open(self.path, self.mode, encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item) + '\n')
    
    def append(self, item: dict) -> None:
        """Append an item to the file."""
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(item) + '\n')



def load_test_suite(path: Union[str, Path]) -> TestSuite:
    """Load a test suite from a JSONL file."""
    path = Path(path)
    reader = JSONLReader(path)
    
    test_cases = []
    suite_metadata = {}
    
    for item in reader:
        # Check if it's metadata (first line can be suite metadata)
        if 'suite_name' in item or 'description' in item:
            suite_metadata = item
            continue
        
        # Parse as test case
        test_case = parse_test_case(item)
        test_cases.append(test_case)
    
    # Determine task type from filename or metadata
    task_type = TaskType.INSTRUCTION_FOLLOWING
    if 'task_type' in suite_metadata:
        task_type = TaskType(suite_metadata['task_type'])
    else:
        # Infer from filename
        name = path.stem.lower()
        for tt in TaskType:
            if tt.value in name:
                task_type = tt
                break
    
    return TestSuite(
        name=suite_metadata.get('suite_name', path.stem),
        description=suite_metadata.get('description', f"Test suite: {path.stem}"),
        task_type=task_type,
        test_cases=test_cases,
        version=suite_metadata.get('version', '1.0.0'),
    )


def parse_test_case(item: dict) -> TestCase:
    """Parse a dict into a TestCase object."""
    # Parse expected output
    expected_data = item.get('expected', {})
    expected = ExpectedOutput(
        type=expected_data.get('type', 'exact'),
        value=expected_data.get('value'),
        schema=expected_data.get('schema'),
        pattern=expected_data.get('pattern'),
        min_length=expected_data.get('min_length'),
        max_length=expected_data.get('max_length'),
        forbidden_words=expected_data.get('forbidden_words'),
        required_words=expected_data.get('required_words'),
    )
    
    # Parse scoring config
    scoring_data = item.get('scoring', {})
    scoring = ScoringConfig(
        method=scoring_data.get('method', 'exact_match'),
        weight=scoring_data.get('weight', 1.0),
        params=scoring_data.get('params', {}),
    )
    
    return TestCase(
        id=item['id'],
        task=item.get('task', 'unknown'),
        prompt=item['prompt'],
        expected=expected,
        scoring=scoring,
        tags=item.get('tags', []),
        metadata=item.get('metadata', {}),
        num_runs=item.get('num_runs', 1),
        needle=item.get('needle'),
        haystack_size=item.get('haystack_size'),
    )


def save_test_suite(suite: TestSuite, path: Union[str, Path]) -> None:
    """Save a test suite to a JSONL file."""
    path = Path(path)
    writer = JSONLWriter(path)
    
    # Write metadata first
    metadata = {
        'suite_name': suite.name,
        'description': suite.description,
        'task_type': suite.task_type.value,
        'version': suite.version,
        'created_at': suite.created_at,
    }
    
    items = [metadata]
    
    # Convert test cases
    for tc in suite.test_cases:
        item = {
            'id': tc.id,
            'task': tc.task,
            'prompt': tc.prompt,
            'expected': {
                'type': tc.expected.type,
            },
            'scoring': {
                'method': tc.scoring.method,
                'weight': tc.scoring.weight,
                'params': tc.scoring.params,
            },
            'tags': tc.tags,
        }
        
        # Add optional expected fields
        if tc.expected.value:
            item['expected']['value'] = tc.expected.value
        if tc.expected.schema:
            item['expected']['schema'] = tc.expected.schema
        if tc.expected.pattern:
            item['expected']['pattern'] = tc.expected.pattern
        if tc.expected.min_length:
            item['expected']['min_length'] = tc.expected.min_length
        if tc.expected.max_length:
            item['expected']['max_length'] = tc.expected.max_length
        if tc.expected.forbidden_words:
            item['expected']['forbidden_words'] = tc.expected.forbidden_words
        if tc.expected.required_words:
            item['expected']['required_words'] = tc.expected.required_words
        
        # Add optional test case fields
        if tc.metadata:
            item['metadata'] = tc.metadata
        if tc.num_runs > 1:
            item['num_runs'] = tc.num_runs
        if tc.needle:
            item['needle'] = tc.needle
        if tc.haystack_size:
            item['haystack_size'] = tc.haystack_size
        
        items.append(item)
    
    writer.write_all(items)



def export_results_jsonl(
    suite_result: SuiteResult,
    path: Union[str, Path]
) -> None:
    """Export suite results to JSONL."""
    path = Path(path)
    writer = JSONLWriter(path)
    
    # Write summary first
    summary = {
        'type': 'summary',
        'suite_name': suite_result.suite_name,
        'model_id': suite_result.model_id,
        'run_id': suite_result.run_id,
        'timestamp': suite_result.timestamp,
        'total_tests': suite_result.total_tests,
        'passed_tests': suite_result.passed_tests,
        'failed_tests': suite_result.failed_tests,
        'error_tests': suite_result.error_tests,
        'accuracy': suite_result.accuracy,
        'accuracy_ci': suite_result.accuracy_ci,
        'mean_latency_ms': suite_result.mean_latency_ms,
        'total_tokens': suite_result.total_tokens,
        'total_cost': suite_result.total_cost,
    }
    
    items = [summary]
    
    # Convert individual results
    for result in suite_result.results:
        item = {
            'type': 'result',
            'test_id': result.test_id,
            'model_id': result.model_id,
            'output': result.output,
            'tokens_in': result.tokens_in,
            'tokens_out': result.tokens_out,
            'latency_ms': result.latency_ms,
            'cost_estimate': result.cost_estimate,
            'timestamp': result.timestamp,
            'run_hash': result.run_hash,
            'passed': result.passed,
            'score': result.score,
            'scoring_method': result.scoring_method,
            'scoring_details': result.scoring_details,
        }
        
        if result.confidence is not None:
            item['confidence'] = result.confidence
        if result.entropy is not None:
            item['entropy'] = result.entropy
        if result.error:
            item['error'] = result.error
        
        items.append(item)
    
    writer.write_all(items)


def load_results_jsonl(path: Union[str, Path]) -> SuiteResult:
    """Load suite results from JSONL."""
    path = Path(path)
    reader = JSONLReader(path)
    
    summary = None
    results = []
    
    for item in reader:
        if item.get('type') == 'summary':
            summary = item
        elif item.get('type') == 'result':
            result = RunResult(
                test_id=item['test_id'],
                model_id=item['model_id'],
                output=item['output'],
                tokens_in=item['tokens_in'],
                tokens_out=item['tokens_out'],
                latency_ms=item['latency_ms'],
                cost_estimate=item['cost_estimate'],
                timestamp=item['timestamp'],
                run_hash=item['run_hash'],
                passed=item['passed'],
                score=item['score'],
                scoring_method=item['scoring_method'],
                scoring_details=item.get('scoring_details', {}),
                confidence=item.get('confidence'),
                entropy=item.get('entropy'),
                error=item.get('error'),
            )
            results.append(result)
    
    if not summary:
        raise ValueError(f"No summary found in {path}")
    
    return SuiteResult(
        suite_name=summary['suite_name'],
        model_id=summary['model_id'],
        run_id=summary['run_id'],
        timestamp=summary['timestamp'],
        total_tests=summary['total_tests'],
        passed_tests=summary['passed_tests'],
        failed_tests=summary['failed_tests'],
        error_tests=summary['error_tests'],
        accuracy=summary['accuracy'],
        accuracy_ci=tuple(summary['accuracy_ci']) if summary.get('accuracy_ci') else None,
        mean_latency_ms=summary['mean_latency_ms'],
        total_tokens=summary['total_tokens'],
        total_cost=summary['total_cost'],
        results=results,
    )
