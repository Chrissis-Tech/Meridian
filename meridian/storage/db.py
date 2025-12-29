"""
Meridian Storage - SQLite Database

Handles persistent storage of runs, results, and metrics using SQLite.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..config import DATABASE_PATH
from ..types import (
    RunResult,
    SuiteResult,
    ComparisonResult,
    BaselineThresholds,
)


class Database:
    """SQLite database for Meridian storage."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
    
    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_schema(self):
        """Initialize database schema."""
        with self._connection() as conn:
            cursor = conn.cursor()
            
            # Runs table - high-level run metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    suite_name TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    config_hash TEXT,
                    config_json TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT DEFAULT 'running',
                    total_tests INTEGER DEFAULT 0,
                    passed_tests INTEGER DEFAULT 0,
                    failed_tests INTEGER DEFAULT 0,
                    error_tests INTEGER DEFAULT 0,
                    accuracy REAL,
                    accuracy_ci_lower REAL,
                    accuracy_ci_upper REAL,
                    mean_latency_ms REAL,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0
                )
            """)
            
            # Results table - individual test results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    test_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    output TEXT,
                    tokens_in INTEGER,
                    tokens_out INTEGER,
                    latency_ms REAL,
                    cost_estimate REAL,
                    timestamp TEXT NOT NULL,
                    result_hash TEXT,
                    passed INTEGER,
                    score REAL,
                    scoring_method TEXT,
                    scoring_details TEXT,
                    confidence REAL,
                    entropy REAL,
                    run_index INTEGER DEFAULT 0,
                    error TEXT,
                    interpretability_ref TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """)
            
            # Comparisons table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS comparisons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_a_id TEXT NOT NULL,
                    run_b_id TEXT NOT NULL,
                    suite_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    accuracy_delta REAL,
                    accuracy_significant INTEGER,
                    accuracy_p_value REAL,
                    latency_delta REAL,
                    cost_delta REAL,
                    regressions TEXT,
                    improvements TEXT,
                    cohens_d REAL,
                    FOREIGN KEY (run_a_id) REFERENCES runs(run_id),
                    FOREIGN KEY (run_b_id) REFERENCES runs(run_id)
                )
            """)
            
            # Baselines table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    suite_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    accuracy_min REAL,
                    pass_rate_min REAL,
                    latency_max_ms REAL,
                    hallucination_rate_max REAL,
                    attack_success_rate_max REAL,
                    accuracy_tolerance REAL,
                    UNIQUE(model_id, suite_name)
                )
            """)
            
            # Causal traces table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS causal_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    test_id TEXT NOT NULL,
                    layer INTEGER NOT NULL,
                    head INTEGER,
                    position INTEGER NOT NULL,
                    component_type TEXT NOT NULL,
                    impact REAL NOT NULL,
                    baseline_prob REAL,
                    patched_prob REAL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_test_id ON results(test_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_suite ON runs(suite_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_id)")
    
    # =========================================================================
    # Run Operations
    # =========================================================================
    
    def create_run(
        self,
        run_id: str,
        suite_name: str,
        model_id: str,
        config: dict
    ) -> str:
        """Create a new run record."""
        from ..utils import hash_config
        
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO runs (run_id, suite_name, model_id, config_hash, config_json, started_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                suite_name,
                model_id,
                hash_config(config),
                json.dumps(config),
                datetime.now().isoformat()
            ))
        return run_id
    
    def update_run(self, run_id: str, **updates) -> None:
        """Update a run record."""
        with self._connection() as conn:
            cursor = conn.cursor()
            
            set_clauses = []
            values = []
            for key, value in updates.items():
                set_clauses.append(f"{key} = ?")
                values.append(value)
            
            values.append(run_id)
            
            cursor.execute(f"""
                UPDATE runs SET {', '.join(set_clauses)}
                WHERE run_id = ?
            """, values)
    
    def complete_run(
        self,
        run_id: str,
        suite_result: SuiteResult
    ) -> None:
        """Mark a run as complete with final results."""
        accuracy_ci_lower = suite_result.accuracy_ci[0] if suite_result.accuracy_ci else None
        accuracy_ci_upper = suite_result.accuracy_ci[1] if suite_result.accuracy_ci else None
        
        self.update_run(
            run_id,
            completed_at=datetime.now().isoformat(),
            status="completed",
            total_tests=suite_result.total_tests,
            passed_tests=suite_result.passed_tests,
            failed_tests=suite_result.failed_tests,
            error_tests=suite_result.error_tests,
            accuracy=suite_result.accuracy,
            accuracy_ci_lower=accuracy_ci_lower,
            accuracy_ci_upper=accuracy_ci_upper,
            mean_latency_ms=suite_result.mean_latency_ms,
            total_tokens=suite_result.total_tokens,
            total_cost=suite_result.total_cost
        )
    
    def get_run(self, run_id: str) -> Optional[dict]:
        """Get a run by ID."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_runs(
        self,
        suite_name: Optional[str] = None,
        model_id: Optional[str] = None,
        limit: int = 100
    ) -> list[dict]:
        """Get runs with optional filters."""
        with self._connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM runs WHERE 1=1"
            params = []
            
            if suite_name:
                query += " AND suite_name = ?"
                params.append(suite_name)
            
            if model_id:
                query += " AND model_id = ?"
                params.append(model_id)
            
            query += " ORDER BY started_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # Result Operations
    # =========================================================================
    
    def save_result(self, run_id: str, result: RunResult) -> int:
        """Save a single test result."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO results (
                    run_id, test_id, model_id, output, tokens_in, tokens_out,
                    latency_ms, cost_estimate, timestamp, result_hash, passed,
                    score, scoring_method, scoring_details, confidence, entropy,
                    run_index, error, interpretability_ref
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                result.test_id,
                result.model_id,
                result.output,
                result.tokens_in,
                result.tokens_out,
                result.latency_ms,
                result.cost_estimate,
                result.timestamp,
                result.run_hash,
                1 if result.passed else 0,
                result.score,
                result.scoring_method,
                json.dumps(result.scoring_details),
                result.confidence,
                result.entropy,
                result.run_index,
                result.error,
                result.interpretability_ref
            ))
            return cursor.lastrowid
    
    def get_results(self, run_id: str) -> list[dict]:
        """Get all results for a run."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM results WHERE run_id = ?
                ORDER BY test_id, run_index
            """, (run_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_result(self, run_id: str, test_id: str) -> Optional[dict]:
        """Get a specific result."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM results WHERE run_id = ? AND test_id = ?
            """, (run_id, test_id))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    # =========================================================================
    # Comparison Operations
    # =========================================================================
    
    def save_comparison(self, comparison: ComparisonResult) -> int:
        """Save a comparison result."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO comparisons (
                    run_a_id, run_b_id, suite_name, created_at,
                    accuracy_delta, accuracy_significant, accuracy_p_value,
                    latency_delta, cost_delta, regressions, improvements, cohens_d
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                comparison.run_a_id,
                comparison.run_b_id,
                comparison.suite_name,
                datetime.now().isoformat(),
                comparison.accuracy_delta,
                1 if comparison.accuracy_significant else 0,
                comparison.accuracy_p_value,
                comparison.latency_delta,
                comparison.cost_delta,
                json.dumps(comparison.regressions),
                json.dumps(comparison.improvements),
                comparison.cohens_d
            ))
            return cursor.lastrowid
    
    def get_comparisons(
        self,
        run_id: Optional[str] = None,
        limit: int = 50
    ) -> list[dict]:
        """Get comparisons, optionally filtered by run ID."""
        with self._connection() as conn:
            cursor = conn.cursor()
            
            if run_id:
                cursor.execute("""
                    SELECT * FROM comparisons 
                    WHERE run_a_id = ? OR run_b_id = ?
                    ORDER BY created_at DESC LIMIT ?
                """, (run_id, run_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM comparisons 
                    ORDER BY created_at DESC LIMIT ?
                """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # Baseline Operations
    # =========================================================================
    
    def save_baseline(self, baseline: BaselineThresholds) -> None:
        """Save or update a baseline."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO baselines (
                    model_id, suite_name, created_at,
                    accuracy_min, pass_rate_min, latency_max_ms,
                    hallucination_rate_max, attack_success_rate_max, accuracy_tolerance
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                baseline.model_id,
                baseline.suite_name,
                datetime.now().isoformat(),
                baseline.accuracy_min,
                baseline.pass_rate_min,
                baseline.latency_max_ms,
                baseline.hallucination_rate_max,
                baseline.attack_success_rate_max,
                baseline.accuracy_tolerance
            ))
    
    def get_baseline(
        self,
        model_id: str,
        suite_name: str
    ) -> Optional[BaselineThresholds]:
        """Get a baseline for a model/suite combination."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM baselines 
                WHERE model_id = ? AND suite_name = ?
            """, (model_id, suite_name))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return BaselineThresholds(
                model_id=row["model_id"],
                suite_name=row["suite_name"],
                accuracy_min=row["accuracy_min"] or 0.5,
                pass_rate_min=row["pass_rate_min"] or 0.5,
                latency_max_ms=row["latency_max_ms"] or 5000,
                hallucination_rate_max=row["hallucination_rate_max"] or 0.3,
                attack_success_rate_max=row["attack_success_rate_max"] or 0.1,
                accuracy_tolerance=row["accuracy_tolerance"] or 0.05
            )
    
    # =========================================================================
    # Causal Trace Operations
    # =========================================================================
    
    def save_causal_traces(self, run_id: str, test_id: str, traces: list) -> None:
        """Save causal traces for a test."""
        with self._connection() as conn:
            cursor = conn.cursor()
            for trace in traces:
                cursor.execute("""
                    INSERT INTO causal_traces (
                        run_id, test_id, layer, head, position,
                        component_type, impact, baseline_prob, patched_prob
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    test_id,
                    trace.layer,
                    trace.head,
                    trace.position,
                    trace.component_type,
                    trace.impact,
                    trace.baseline_prob,
                    trace.patched_prob
                ))
    
    def get_causal_traces(
        self,
        run_id: str,
        test_id: Optional[str] = None
    ) -> list[dict]:
        """Get causal traces for a run."""
        with self._connection() as conn:
            cursor = conn.cursor()
            
            if test_id:
                cursor.execute("""
                    SELECT * FROM causal_traces 
                    WHERE run_id = ? AND test_id = ?
                    ORDER BY impact DESC
                """, (run_id, test_id))
            else:
                cursor.execute("""
                    SELECT * FROM causal_traces 
                    WHERE run_id = ?
                    ORDER BY impact DESC
                """, (run_id,))
            
            return [dict(row) for row in cursor.fetchall()]


# Singleton database instance
_db: Optional[Database] = None


def get_db() -> Database:
    """Get the global database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db
