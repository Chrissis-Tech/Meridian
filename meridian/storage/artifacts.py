"""
Meridian Storage - Artifacts Management

Handles run artifacts, hashing, versioning, and file management.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from ..config import RESULTS_DIR, DATA_DIR
from ..utils import generate_run_id, hash_prompt, hash_config


class ArtifactManager:
    """Manages run artifacts and versioning."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or RESULTS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_run_dir(self, run_id: str) -> Path:
        """Get the directory for a run."""
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    def create_run(
        self,
        suite_name: str,
        model_id: str,
        config: dict,
        run_id: Optional[str] = None
    ) -> str:
        """Create a new run with artifacts directory."""
        run_id = run_id or generate_run_id()
        run_dir = self.get_run_dir(run_id)
        
        # Save run metadata
        metadata = {
            'run_id': run_id,
            'suite_name': suite_name,
            'model_id': model_id,
            'config': config,
            'config_hash': hash_config(config),
            'created_at': datetime.now().isoformat(),
            'status': 'running',
        }
        
        self._save_json(run_dir / 'metadata.json', metadata)
        
        return run_id
    
    def complete_run(self, run_id: str, summary: dict) -> None:
        """Mark a run as complete and save summary."""
        run_dir = self.get_run_dir(run_id)
        
        # Update metadata
        metadata_path = run_dir / 'metadata.json'
        metadata = self._load_json(metadata_path)
        metadata['status'] = 'completed'
        metadata['completed_at'] = datetime.now().isoformat()
        self._save_json(metadata_path, metadata)
        
        # Save summary
        self._save_json(run_dir / 'summary.json', summary)
    
    def save_result(
        self,
        run_id: str,
        test_id: str,
        result: dict,
        run_index: int = 0
    ) -> str:
        """Save a single test result."""
        run_dir = self.get_run_dir(run_id)
        results_dir = run_dir / 'results'
        results_dir.mkdir(exist_ok=True)
        
        # Create unique filename
        suffix = f"_{run_index}" if run_index > 0 else ""
        filename = f"{test_id}{suffix}.json"
        
        self._save_json(results_dir / filename, result)
        
        return filename
    
    def get_result(
        self,
        run_id: str,
        test_id: str,
        run_index: int = 0
    ) -> Optional[dict]:
        """Get a single test result."""
        run_dir = self.get_run_dir(run_id)
        
        suffix = f"_{run_index}" if run_index > 0 else ""
        filename = f"{test_id}{suffix}.json"
        
        result_path = run_dir / 'results' / filename
        
        if result_path.exists():
            return self._load_json(result_path)
        return None
    
    def get_all_results(self, run_id: str) -> list[dict]:
        """Get all results for a run."""
        run_dir = self.get_run_dir(run_id)
        results_dir = run_dir / 'results'
        
        if not results_dir.exists():
            return []
        
        results = []
        for path in sorted(results_dir.glob('*.json')):
            results.append(self._load_json(path))
        
        return results
    
    # =========================================================================
    # Interpretability Artifacts
    # =========================================================================
    
    def save_attention(
        self,
        run_id: str,
        test_id: str,
        attention_data: Any
    ) -> str:
        """Save attention patterns."""
        import numpy as np
        
        run_dir = self.get_run_dir(run_id)
        interp_dir = run_dir / 'interpretability' / test_id
        interp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy file
        attention_path = interp_dir / 'attention.npz'
        np.savez_compressed(attention_path, attention=attention_data)
        
        return str(attention_path)
    
    def save_hidden_states(
        self,
        run_id: str,
        test_id: str,
        hidden_states: Any
    ) -> str:
        """Save hidden states."""
        import numpy as np
        
        run_dir = self.get_run_dir(run_id)
        interp_dir = run_dir / 'interpretability' / test_id
        interp_dir.mkdir(parents=True, exist_ok=True)
        
        hidden_path = interp_dir / 'hidden_states.npz'
        np.savez_compressed(hidden_path, hidden_states=hidden_states)
        
        return str(hidden_path)
    
    def save_causal_trace(
        self,
        run_id: str,
        test_id: str,
        trace_data: dict
    ) -> str:
        """Save causal tracing results."""
        run_dir = self.get_run_dir(run_id)
        interp_dir = run_dir / 'interpretability' / test_id
        interp_dir.mkdir(parents=True, exist_ok=True)
        
        trace_path = interp_dir / 'causal_trace.json'
        self._save_json(trace_path, trace_data)
        
        return str(trace_path)
    
    def load_attention(
        self,
        run_id: str,
        test_id: str
    ) -> Optional[Any]:
        """Load attention patterns."""
        import numpy as np
        
        run_dir = self.get_run_dir(run_id)
        attention_path = run_dir / 'interpretability' / test_id / 'attention.npz'
        
        if attention_path.exists():
            data = np.load(attention_path)
            return data['attention']
        return None
    
    # =========================================================================
    # Prompt Versioning
    # =========================================================================
    
    def save_prompt_version(
        self,
        prompt: str,
        metadata: Optional[dict] = None
    ) -> str:
        """Save a prompt with version tracking."""
        prompts_dir = DATA_DIR / 'prompts'
        prompts_dir.mkdir(parents=True, exist_ok=True)
        
        prompt_hash = hash_prompt(prompt)
        prompt_path = prompts_dir / f"{prompt_hash}.json"
        
        version_data = {
            'hash': prompt_hash,
            'prompt': prompt,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {},
        }
        
        # Only save if new
        if not prompt_path.exists():
            self._save_json(prompt_path, version_data)
        
        return prompt_hash
    
    def get_prompt_by_hash(self, prompt_hash: str) -> Optional[str]:
        """Get a prompt by its hash."""
        prompts_dir = DATA_DIR / 'prompts'
        prompt_path = prompts_dir / f"{prompt_hash}.json"
        
        if prompt_path.exists():
            data = self._load_json(prompt_path)
            return data.get('prompt')
        return None
    
    # =========================================================================
    # Export
    # =========================================================================
    
    def export_run(
        self,
        run_id: str,
        output_path: Union[str, Path],
        include_interpretability: bool = True
    ) -> Path:
        """Export a run as a zip archive."""
        run_dir = self.get_run_dir(run_id)
        output_path = Path(output_path)
        
        if not include_interpretability:
            # Create temp copy without interpretability
            import tempfile
            with tempfile.TemporaryDirectory() as temp:
                temp_dir = Path(temp) / run_id
                shutil.copytree(run_dir, temp_dir, ignore=shutil.ignore_patterns('interpretability'))
                shutil.make_archive(str(output_path.with_suffix('')), 'zip', temp_dir.parent, run_id)
        else:
            shutil.make_archive(str(output_path.with_suffix('')), 'zip', run_dir.parent, run_id)
        
        return output_path.with_suffix('.zip')
    
    def delete_run(self, run_id: str) -> bool:
        """Delete a run and all its artifacts."""
        run_dir = self.get_run_dir(run_id)
        if run_dir.exists():
            shutil.rmtree(run_dir)
            return True
        return False
    
    def list_runs(self) -> list[str]:
        """List all run IDs."""
        runs = []
        for path in self.base_dir.iterdir():
            if path.is_dir() and path.name.startswith('run_'):
                runs.append(path.name)
        return sorted(runs, reverse=True)
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def _save_json(self, path: Path, data: dict) -> None:
        """Save JSON data to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _load_json(self, path: Path) -> dict:
        """Load JSON data from file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


# Singleton instance
_artifact_manager: Optional[ArtifactManager] = None


def get_artifact_manager() -> ArtifactManager:
    """Get the global artifact manager."""
    global _artifact_manager
    if _artifact_manager is None:
        _artifact_manager = ArtifactManager()
    return _artifact_manager
