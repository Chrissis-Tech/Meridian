"""
Meridian Scoring - JSON Schema

Validates JSON output against schemas.
"""

import json
from typing import Any, Optional

from ..types import ScoringResult


def validate_json(
    output: str,
    schema: Optional[dict] = None,
    required_keys: Optional[list[str]] = None,
    strict: bool = True,
) -> ScoringResult:
    """
    Validate that output is valid JSON and optionally matches a schema.
    
    Args:
        output: Model output (should be JSON)
        schema: JSON Schema to validate against
        required_keys: List of keys that must be present
        strict: If False, try to extract JSON from mixed output
        
    Returns:
        ScoringResult with validation details
    """
    from ..utils import extract_json
    
    # Try to parse JSON
    if strict:
        try:
            data = json.loads(output.strip())
        except json.JSONDecodeError as e:
            return ScoringResult(
                passed=False,
                score=0.0,
                method="json_schema",
                details={
                    "valid_json": False,
                    "error": str(e),
                    "output_preview": output[:200],
                }
            )
    else:
        data = extract_json(output)
        if data is None:
            return ScoringResult(
                passed=False,
                score=0.0,
                method="json_schema",
                details={
                    "valid_json": False,
                    "error": "No valid JSON found in output",
                    "output_preview": output[:200],
                }
            )
    
    # Check required keys
    if required_keys:
        if not isinstance(data, dict):
            return ScoringResult(
                passed=False,
                score=0.0,
                method="json_schema",
                details={
                    "valid_json": True,
                    "error": "Output is not a JSON object",
                    "type": type(data).__name__,
                }
            )
        
        missing = [k for k in required_keys if k not in data]
        if missing:
            return ScoringResult(
                passed=False,
                score=1.0 - (len(missing) / len(required_keys)),
                method="json_schema",
                details={
                    "valid_json": True,
                    "missing_keys": missing,
                    "present_keys": list(data.keys()),
                }
            )
    
    # Validate against schema
    if schema:
        try:
            import jsonschema
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as e:
            return ScoringResult(
                passed=False,
                score=0.5,  # Partial credit for valid JSON
                method="json_schema",
                details={
                    "valid_json": True,
                    "schema_valid": False,
                    "schema_error": e.message,
                    "schema_path": list(e.absolute_path),
                }
            )
        except jsonschema.SchemaError as e:
            return ScoringResult(
                passed=False,
                score=0.0,
                method="json_schema",
                details={
                    "valid_json": True,
                    "error": f"Invalid schema: {e.message}",
                }
            )
    
    return ScoringResult(
        passed=True,
        score=1.0,
        method="json_schema",
        details={
            "valid_json": True,
            "schema_valid": schema is not None,
            "keys": list(data.keys()) if isinstance(data, dict) else None,
        }
    )


def validate_json_structure(
    output: str,
    expected_structure: dict,
) -> ScoringResult:
    """
    Validate JSON output matches an expected structure (type checking).
    
    Args:
        output: Model output
        expected_structure: Dict describing expected types
            e.g., {"name": str, "age": int, "items": list}
            
    Returns:
        ScoringResult with type validation details
    """
    from ..utils import extract_json
    
    data = extract_json(output)
    if data is None:
        return ScoringResult(
            passed=False,
            score=0.0,
            method="json_structure",
            details={"error": "Invalid JSON"}
        )
    
    if not isinstance(data, dict):
        return ScoringResult(
            passed=False,
            score=0.0,
            method="json_structure",
            details={"error": "Expected object, got " + type(data).__name__}
        )
    
    type_errors = []
    checked = 0
    passed_checks = 0
    
    for key, expected_type in expected_structure.items():
        checked += 1
        if key not in data:
            type_errors.append(f"Missing key: {key}")
        elif not isinstance(data[key], expected_type):
            type_errors.append(
                f"Key '{key}': expected {expected_type.__name__}, "
                f"got {type(data[key]).__name__}"
            )
        else:
            passed_checks += 1
    
    passed = len(type_errors) == 0
    score = passed_checks / checked if checked > 0 else 0.0
    
    return ScoringResult(
        passed=passed,
        score=score,
        method="json_structure",
        details={
            "type_errors": type_errors,
            "checked_keys": checked,
            "passed_checks": passed_checks,
        }
    )


def validate_json_array(
    output: str,
    min_items: int = 0,
    max_items: Optional[int] = None,
    item_schema: Optional[dict] = None,
) -> ScoringResult:
    """
    Validate JSON output is an array with optional constraints.
    
    Args:
        output: Model output
        min_items: Minimum number of items
        max_items: Maximum number of items
        item_schema: Schema for each item
        
    Returns:
        ScoringResult with array validation details
    """
    from ..utils import extract_json
    
    data = extract_json(output)
    if data is None:
        return ScoringResult(
            passed=False,
            score=0.0,
            method="json_array",
            details={"error": "Invalid JSON"}
        )
    
    if not isinstance(data, list):
        return ScoringResult(
            passed=False,
            score=0.0,
            method="json_array",
            details={"error": "Expected array, got " + type(data).__name__}
        )
    
    count = len(data)
    errors = []
    
    if count < min_items:
        errors.append(f"Too few items: {count} < {min_items}")
    
    if max_items is not None and count > max_items:
        errors.append(f"Too many items: {count} > {max_items}")
    
    # Validate items against schema
    if item_schema:
        import jsonschema
        invalid_items = []
        for i, item in enumerate(data):
            try:
                jsonschema.validate(item, item_schema)
            except jsonschema.ValidationError as e:
                invalid_items.append({"index": i, "error": e.message})
        
        if invalid_items:
            errors.extend([f"Item {e['index']}: {e['error']}" for e in invalid_items[:3]])
    
    passed = len(errors) == 0
    
    return ScoringResult(
        passed=passed,
        score=1.0 if passed else 0.5,
        method="json_array",
        details={
            "item_count": count,
            "min_items": min_items,
            "max_items": max_items,
            "errors": errors[:5],
        }
    )
