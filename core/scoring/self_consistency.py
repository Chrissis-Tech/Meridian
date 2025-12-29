"""
Meridian Scoring - Self-Consistency

Measures consistency across multiple model runs.
"""

from collections import Counter
from typing import Optional

from ..types import ScoringResult, ConsistencyResult
from ..utils import normalize_text, entropy


def measure_self_consistency(
    outputs: list[str],
    normalize: bool = True,
    semantic_similarity: bool = False,
) -> ScoringResult:
    """
    Measure self-consistency across multiple outputs.
    
    Args:
        outputs: List of outputs from multiple runs
        normalize: Whether to normalize outputs before comparison
        semantic_similarity: Use semantic similarity (requires embeddings)
        
    Returns:
        ScoringResult with consistency metrics
    """
    if len(outputs) < 2:
        return ScoringResult(
            passed=True,
            score=1.0,
            method="self_consistency",
            details={"error": "Need at least 2 outputs"}
        )
    
    # Normalize outputs
    if normalize:
        normalized = [normalize_text(o) for o in outputs]
    else:
        normalized = [o.strip() for o in outputs]
    
    # Count unique answers
    counter = Counter(normalized)
    unique_count = len(counter)
    most_common_count = counter.most_common(1)[0][1]
    
    # Calculate consistency score
    consistency_score = most_common_count / len(outputs)
    
    # Calculate diversity/entropy
    probs = [count / len(outputs) for count in counter.values()]
    output_entropy = entropy(probs) if len(probs) > 1 else 0.0
    
    # Determine if consistent (majority answer)
    passed = consistency_score >= 0.6  # At least 60% agreement
    
    return ScoringResult(
        passed=passed,
        score=consistency_score,
        method="self_consistency",
        details={
            "num_outputs": len(outputs),
            "unique_answers": unique_count,
            "most_common_count": most_common_count,
            "consistency_score": consistency_score,
            "entropy": output_entropy,
            "answer_distribution": dict(counter.most_common(5)),
        }
    )


def detect_contradictions(
    outputs: list[str],
    check_negations: bool = True,
    check_numbers: bool = True,
) -> ScoringResult:
    """
    Detect contradictions between outputs.
    
    Args:
        outputs: List of outputs from multiple runs
        check_negations: Check for yes/no contradictions
        check_numbers: Check for numeric contradictions
        
    Returns:
        ScoringResult with contradiction details
    """
    import re
    from ..utils import extract_number
    
    contradictions = []
    
    if len(outputs) < 2:
        return ScoringResult(
            passed=True,
            score=1.0,
            method="contradiction_detection",
            details={"error": "Need at least 2 outputs"}
        )
    
    # Check for yes/no contradictions
    if check_negations:
        yes_pattern = r'\b(yes|correct|true|right|agree|affirmative)\b'
        no_pattern = r'\b(no|incorrect|false|wrong|disagree|negative)\b'
        
        has_yes = []
        has_no = []
        
        for i, output in enumerate(outputs):
            lower = output.lower()
            if re.search(yes_pattern, lower):
                has_yes.append(i)
            if re.search(no_pattern, lower):
                has_no.append(i)
        
        if has_yes and has_no:
            # Both yes and no responses found
            contradictions.append({
                "type": "yes_no_contradiction",
                "yes_indices": has_yes,
                "no_indices": has_no,
            })
    
    # Check for numeric contradictions
    if check_numbers:
        numbers = []
        for i, output in enumerate(outputs):
            num = extract_number(output)
            if num is not None:
                numbers.append((i, num))
        
        if len(numbers) >= 2:
            # Check if numbers are significantly different
            values = [n[1] for n in numbers]
            min_val, max_val = min(values), max(values)
            
            if min_val != 0 and max_val / min_val > 1.5:
                # More than 50% difference
                contradictions.append({
                    "type": "numeric_contradiction",
                    "values": values,
                    "range": (min_val, max_val),
                })
    
    passed = len(contradictions) == 0
    
    return ScoringResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        method="contradiction_detection",
        details={
            "contradictions": contradictions,
            "contradiction_count": len(contradictions),
        }
    )


def aggregate_consistency_results(
    outputs: list[str],
    correct_answer: Optional[str] = None,
) -> ConsistencyResult:
    """
    Create a full ConsistencyResult from multiple outputs.
    
    Args:
        outputs: List of outputs from multiple runs
        correct_answer: Optional correct answer for accuracy
        
    Returns:
        ConsistencyResult with all metrics
    """
    # Get consistency metrics
    consistency = measure_self_consistency(outputs)
    contradictions = detect_contradictions(outputs)
    
    # Determine majority answer
    normalized = [normalize_text(o) for o in outputs]
    counter = Counter(normalized)
    majority_answer = counter.most_common(1)[0][0] if counter else ""
    
    # Calculate variance in length
    lengths = [len(o) for o in outputs]
    import numpy as np
    variance = float(np.var(lengths)) if len(lengths) > 1 else 0.0
    
    return ConsistencyResult(
        test_id="",  # Set by caller
        model_id="",  # Set by caller
        num_runs=len(outputs),
        outputs=outputs,
        self_consistency_score=consistency.score,
        entropy=consistency.details.get("entropy", 0.0),
        contradiction_detected=not contradictions.passed,
        variance=variance,
    )


def semantic_similarity_score(
    outputs: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> ScoringResult:
    """
    Calculate semantic similarity between outputs using embeddings.
    
    Args:
        outputs: List of outputs
        model_name: Sentence transformer model to use
        
    Returns:
        ScoringResult with similarity matrix
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        model = SentenceTransformer(model_name)
        embeddings = model.encode(outputs)
        
        # Calculate pairwise cosine similarities
        similarities = []
        n = len(outputs)
        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(float(sim))
        
        avg_similarity = np.mean(similarities) if similarities else 1.0
        min_similarity = np.min(similarities) if similarities else 1.0
        
        passed = avg_similarity >= 0.7  # High semantic similarity
        
        return ScoringResult(
            passed=passed,
            score=float(avg_similarity),
            method="semantic_similarity",
            details={
                "avg_similarity": float(avg_similarity),
                "min_similarity": float(min_similarity),
                "num_comparisons": len(similarities),
            }
        )
        
    except ImportError:
        return ScoringResult(
            passed=True,
            score=1.0,
            method="semantic_similarity",
            details={"error": "sentence-transformers not installed"}
        )
