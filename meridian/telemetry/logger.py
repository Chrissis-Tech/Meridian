"""
Meridian Telemetry - Structured Logging

Provides structured logging for production observability.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core.config import config


class StructuredLogger:
    """Structured JSON logger for Meridian."""
    
    def __init__(
        self,
        name: str = "Meridian",
        log_file: Optional[Path] = None,
        level: int = logging.INFO,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Formatter for structured output
        formatter = logging.Formatter(
            '%(message)s'  # We format JSON ourselves
        )
        
        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        
        # File handler (optional)
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _format_log(
        self,
        level: str,
        event: str,
        **kwargs
    ) -> str:
        """Format log entry as JSON."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "event": event,
            **kwargs
        }
        return json.dumps(entry)
    
    def info(self, event: str, **kwargs):
        """Log info-level event."""
        self.logger.info(self._format_log("INFO", event, **kwargs))
    
    def warning(self, event: str, **kwargs):
        """Log warning-level event."""
        self.logger.warning(self._format_log("WARNING", event, **kwargs))
    
    def error(self, event: str, **kwargs):
        """Log error-level event."""
        self.logger.error(self._format_log("ERROR", event, **kwargs))
    
    def debug(self, event: str, **kwargs):
        """Log debug-level event."""
        self.logger.debug(self._format_log("DEBUG", event, **kwargs))
    
    # Meridian-specific methods
    def log_run_start(
        self,
        run_id: str,
        suite_name: str,
        model_id: str,
        config: dict,
    ):
        """Log evaluation run start."""
        self.info(
            "run_started",
            run_id=run_id,
            suite_name=suite_name,
            model_id=model_id,
            config=config,
        )
    
    def log_run_complete(
        self,
        run_id: str,
        accuracy: float,
        passed: int,
        total: int,
        duration_ms: float,
    ):
        """Log evaluation run completion."""
        self.info(
            "run_completed",
            run_id=run_id,
            accuracy=accuracy,
            passed=passed,
            total=total,
            duration_ms=duration_ms,
        )
    
    def log_test_result(
        self,
        run_id: str,
        test_id: str,
        passed: bool,
        score: float,
        latency_ms: float,
    ):
        """Log individual test result."""
        self.debug(
            "test_result",
            run_id=run_id,
            test_id=test_id,
            passed=passed,
            score=score,
            latency_ms=latency_ms,
        )
    
    def log_model_call(
        self,
        model_id: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        cost: float,
    ):
        """Log model API call."""
        self.debug(
            "model_call",
            model_id=model_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost=cost,
        )
    
    def log_regression(
        self,
        run_a: str,
        run_b: str,
        test_id: str,
        delta: float,
    ):
        """Log detected regression."""
        self.warning(
            "regression_detected",
            run_a=run_a,
            run_b=run_b,
            test_id=test_id,
            delta=delta,
        )


# Global logger instance
_logger: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    """Get or create the global logger."""
    global _logger
    if _logger is None:
        log_file = config.paths.get("logs", "data/logs/Meridian.jsonl")
        _logger = StructuredLogger(log_file=Path(log_file))
    return _logger


def export_metrics_json(
    run_id: str,
    output_path: Path,
):
    """Export run metrics to JSON for external consumption."""
    from core.storage.db import get_db
    
    db = get_db()
    run = db.get_run(run_id)
    results = db.get_results(run_id)
    
    if not run:
        raise ValueError(f"Run not found: {run_id}")
    
    metrics = {
        "run_id": run_id,
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "accuracy": run["accuracy"],
            "passed_tests": run["passed_tests"],
            "failed_tests": run["failed_tests"],
            "mean_latency_ms": run["mean_latency_ms"],
            "total_cost": run["total_cost"],
        },
        "test_results": [
            {
                "test_id": r["test_id"],
                "passed": bool(r["passed"]),
                "score": r["score"],
                "latency_ms": r["latency_ms"],
            }
            for r in results
        ],
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    
    return metrics


def export_metrics_csv(
    run_id: str,
    output_path: Path,
):
    """Export run metrics to CSV."""
    import csv
    from core.storage.db import get_db
    
    db = get_db()
    results = db.get_results(run_id)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "test_id", "passed", "score", "latency_ms", "tokens_in", "tokens_out"
        ])
        for r in results:
            writer.writerow([
                r["test_id"],
                1 if r["passed"] else 0,
                r["score"],
                r["latency_ms"],
                r.get("tokens_in", 0),
                r.get("tokens_out", 0),
            ])
    
    return output_path
