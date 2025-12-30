"""
Meridian Custom Suites - Evaluation Pack Management

Handles user-created test suites with validation and anti-cheating features.
"""

import json
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, List
import random

from ..config import DATA_DIR


@dataclass
class TestCase:
    """Single test case in an evaluation pack."""
    id: str
    input: str
    expected: Optional[str] = None
    scorer: str = "exact"  # exact, contains, regex, llm_judge
    tags: List[str] = field(default_factory=list)
    rubric: Optional[str] = None  # Required for llm_judge
    
    def validate(self) -> List[str]:
        """Validate test case, return list of issues."""
        issues = []
        
        if not self.id:
            issues.append("Test case missing 'id'")
        if not self.input:
            issues.append(f"Test case '{self.id}' missing 'input'")
        
        # Leak detection
        if self.expected and self.expected.lower() in self.input.lower():
            issues.append(f"LEAK WARNING: Test '{self.id}' input contains expected answer")
        
        # LLM Judge safety
        if self.scorer == "llm_judge" and not self.rubric:
            issues.append(f"Test '{self.id}' uses llm_judge but missing 'rubric'")
        
        return issues


@dataclass
class EvaluationPack:
    """User-created evaluation suite."""
    name: str
    tests: List[TestCase]
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: int = 1
    
    @property
    def id(self) -> str:
        """Generate deterministic ID from name."""
        return hashlib.sha256(self.name.encode()).hexdigest()[:12]
    
    def validate(self) -> List[str]:
        """Validate entire pack, return list of issues."""
        issues = []
        
        if not self.name:
            issues.append("Suite missing 'name'")
        if not self.tests:
            issues.append("Suite has no test cases")
        
        # Check for duplicate IDs
        ids = [t.id for t in self.tests]
        if len(ids) != len(set(ids)):
            issues.append("Duplicate test IDs found")
        
        # Validate each test
        for test in self.tests:
            issues.extend(test.validate())
        
        return issues
    
    def get_holdout_split(self, eval_ratio: float = 0.2, seed: int = 42) -> tuple:
        """Split tests into dev (80%) and eval (20%) sets."""
        random.seed(seed)
        tests = self.tests.copy()
        random.shuffle(tests)
        
        split_idx = int(len(tests) * (1 - eval_ratio))
        dev_set = tests[:split_idx]
        eval_set = tests[split_idx:]
        
        return dev_set, eval_set
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tests": [asdict(t) for t in self.tests],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }
    
    def to_jsonl(self) -> str:
        """Export as JSONL for portability."""
        lines = []
        for test in self.tests:
            lines.append(json.dumps(asdict(test), ensure_ascii=False))
        return "\n".join(lines)
    
    @classmethod
    def from_jsonl(cls, name: str, content: str, description: str = "") -> "EvaluationPack":
        """Parse JSONL content into EvaluationPack."""
        tests = []
        
        for line_num, line in enumerate(content.strip().split("\n"), 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                test = TestCase(
                    id=data.get("id", f"test_{line_num}"),
                    input=data.get("input", data.get("prompt", "")),
                    expected=data.get("expected", data.get("answer", None)),
                    scorer=data.get("scorer", "exact"),
                    tags=data.get("tags", []),
                    rubric=data.get("rubric"),
                )
                tests.append(test)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_num}: Invalid JSON - {e}")
        
        return cls(name=name, tests=tests, description=description)
    
    @classmethod
    def from_csv(cls, name: str, content: str, description: str = "") -> "EvaluationPack":
        """Parse CSV content into EvaluationPack."""
        import csv
        from io import StringIO
        
        tests = []
        reader = csv.DictReader(StringIO(content))
        
        for row_num, row in enumerate(reader, 1):
            test = TestCase(
                id=row.get("id", f"test_{row_num}"),
                input=row.get("input", row.get("prompt", "")),
                expected=row.get("expected", row.get("answer", None)),
                scorer=row.get("scorer", "exact"),
                tags=row.get("tags", "").split(",") if row.get("tags") else [],
                rubric=row.get("rubric"),
            )
            tests.append(test)
        
        return cls(name=name, tests=tests, description=description)


class CustomSuiteManager:
    """Manages custom evaluation packs in SQLite."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (DATA_DIR / "custom_suites.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS suites (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    tests_json TEXT NOT NULL,
                    created_at TEXT,
                    updated_at TEXT,
                    version INTEGER DEFAULT 1
                )
            """)
    
    def save(self, pack: EvaluationPack) -> str:
        """Save evaluation pack to database."""
        with sqlite3.connect(self.db_path) as conn:
            tests_json = json.dumps([asdict(t) for t in pack.tests])
            
            conn.execute("""
                INSERT OR REPLACE INTO suites 
                (id, name, description, tests_json, created_at, updated_at, version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                pack.id,
                pack.name,
                pack.description,
                tests_json,
                pack.created_at,
                datetime.utcnow().isoformat(),
                pack.version
            ))
        
        return pack.id
    
    def get(self, suite_id: str) -> Optional[EvaluationPack]:
        """Get evaluation pack by ID."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT name, description, tests_json, created_at, updated_at, version FROM suites WHERE id = ?",
                (suite_id,)
            ).fetchone()
        
        if not row:
            return None
        
        name, description, tests_json, created_at, updated_at, version = row
        tests_data = json.loads(tests_json)
        tests = [TestCase(**t) for t in tests_data]
        
        return EvaluationPack(
            name=name,
            tests=tests,
            description=description,
            created_at=created_at,
            updated_at=updated_at,
            version=version
        )
    
    def get_by_name(self, name: str) -> Optional[EvaluationPack]:
        """Get evaluation pack by name."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM suites WHERE name = ?",
                (name,)
            ).fetchone()
        
        if not row:
            return None
        
        return self.get(row[0])
    
    def list_all(self) -> List[dict]:
        """List all saved suites (summary only)."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT id, name, description, 
                       json_array_length(tests_json) as test_count,
                       created_at, updated_at
                FROM suites
                ORDER BY updated_at DESC
            """).fetchall()
        
        # json_array_length might not work, fallback to manual count
        result = []
        for row in rows:
            suite_id, name, description, test_count, created_at, updated_at = row
            
            # Get actual test count
            full_suite = self.get(suite_id)
            actual_count = len(full_suite.tests) if full_suite else 0
            
            result.append({
                "id": suite_id,
                "name": name,
                "description": description,
                "test_count": actual_count,
                "created_at": created_at,
                "updated_at": updated_at
            })
        
        return result
    
    def delete(self, suite_id: str) -> bool:
        """Delete a suite."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM suites WHERE id = ?", (suite_id,))
            return cursor.rowcount > 0
    
    def export_jsonl(self, suite_id: str) -> Optional[str]:
        """Export suite as JSONL."""
        pack = self.get(suite_id)
        if not pack:
            return None
        return pack.to_jsonl()


# Singleton
_manager: Optional[CustomSuiteManager] = None

def get_custom_suite_manager() -> CustomSuiteManager:
    """Get global custom suite manager."""
    global _manager
    if _manager is None:
        _manager = CustomSuiteManager()
    return _manager
