"""
Meridian Attestation - Tamper-Evident Golden Runs

Creates verifiable, reproducible evaluation bundles with:
- Canonical JSON hashing
- Environment capture
- Manifest-based integrity verification
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from ..config import RESULTS_DIR


@dataclass
class EnvironmentInfo:
    """Captured environment for reproducibility."""
    python_version: str
    platform: str
    os_name: str
    meridian_version: str
    git_commit: Optional[str] = None
    git_dirty: bool = False
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


@dataclass
class Attestation:
    """Tamper-evident attestation for a run."""
    run_id: str
    manifest_hash: str
    config_hash: str
    suite_hash: str
    responses_hash: str
    environment: EnvironmentInfo
    created_at: str
    meridian_version: str
    signature: Optional[str] = None  # Reserved for future cryptographic signing
    verified: Optional[bool] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary with environment as nested dict."""
        d = asdict(self)
        return d


class AttestationManager:
    """Manages tamper-evident attestation for runs."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or RESULTS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Environment Capture
    # =========================================================================
    
    def capture_environment(self) -> EnvironmentInfo:
        """Capture current environment for reproducibility."""
        from .. import __version__
        
        git_commit = None
        git_dirty = False
        
        try:
            # Get git commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                git_commit = result.stdout.strip()[:12]
            
            # Check if dirty
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5
            )
            git_dirty = bool(result.stdout.strip())
        except Exception:
            pass  # Git not available
        
        return EnvironmentInfo(
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform=platform.platform(),
            os_name=platform.system(),
            meridian_version=__version__,
            git_commit=git_commit,
            git_dirty=git_dirty,
        )
    
    # =========================================================================
    # Canonical Hashing
    # =========================================================================
    
    def canonical_json(self, data: Any) -> str:
        """Convert data to canonical JSON (deterministic, ordered)."""
        return json.dumps(
            data,
            sort_keys=True,
            separators=(',', ':'),  # Compact, no extra spaces
            ensure_ascii=True,
            default=str  # Handle datetime, Path, etc.
        )
    
    def hash_data(self, data: Any) -> str:
        """Hash data using canonical JSON representation."""
        canonical = self.canonical_json(data)
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    
    def hash_file(self, path: Path) -> str:
        """Hash file contents."""
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def hash_directory(self, dir_path: Path) -> tuple[str, list[dict]]:
        """Hash all files in directory, return master hash and file list."""
        files = []
        
        for path in sorted(dir_path.rglob('*')):
            if path.is_file():
                rel_path = path.relative_to(dir_path).as_posix()
                file_hash = self.hash_file(path)
                files.append({
                    'path': rel_path,
                    'hash': file_hash,
                    'size': path.stat().st_size
                })
        
        # Master hash of all file hashes
        master = self.hash_data(files)
        return master, files
    
    # =========================================================================
    # Secret Redaction
    # =========================================================================
    
    def redact_secrets(self, config: dict) -> dict:
        """Remove sensitive data from config before attestation."""
        sensitive_keys = {
            'api_key', 'api_secret', 'token', 'password', 'secret',
            'openai_api_key', 'anthropic_api_key', 'deepseek_api_key',
            'mistral_api_key', 'groq_api_key', 'together_api_key'
        }
        
        def redact(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    k: '[REDACTED]' if k.lower() in sensitive_keys else redact(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [redact(item) for item in obj]
            return obj
        
        return redact(config)
    
    # =========================================================================
    # Manifest Generation
    # =========================================================================
    
    def generate_manifest(self, run_dir: Path) -> dict:
        """Generate canonical manifest for a run directory."""
        manifest = {
            'version': '1.0',
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'files': []
        }
        
        # Hash all files except manifest.json and attestation.json
        exclude = {'manifest.json', 'attestation.json'}
        
        for path in sorted(run_dir.rglob('*')):
            if path.is_file() and path.name not in exclude:
                rel_path = path.relative_to(run_dir).as_posix()
                manifest['files'].append({
                    'path': rel_path,
                    'hash': self.hash_file(path),
                    'size': path.stat().st_size
                })
        
        manifest['master_hash'] = self.hash_data(manifest['files'])
        return manifest
    
    # =========================================================================
    # Attestation Creation
    # =========================================================================
    
    def create_attestation(
        self,
        run_id: str,
        config: dict,
        suite_data: list[dict],
        responses: list[dict]
    ) -> Attestation:
        """Create tamper-evident attestation for a run."""
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Capture environment
        env = self.capture_environment()
        
        # Redact secrets from config
        safe_config = self.redact_secrets(config)
        
        # Save canonical config
        config_path = run_dir / 'config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(self.canonical_json(safe_config))
        
        # Save suite snapshot
        suite_path = run_dir / 'suite.jsonl'
        with open(suite_path, 'w', encoding='utf-8') as f:
            for test in suite_data:
                f.write(self.canonical_json(test) + '\n')
        
        # Save responses
        responses_dir = run_dir / 'responses'
        responses_dir.mkdir(exist_ok=True)
        for resp in responses:
            test_id = resp.get('test_id', 'unknown')
            resp_path = responses_dir / f'{test_id}.json'
            with open(resp_path, 'w', encoding='utf-8') as f:
                f.write(self.canonical_json(resp))
        
        # Generate manifest
        manifest = self.generate_manifest(run_dir)
        manifest_path = run_dir / 'manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            f.write(self.canonical_json(manifest))
        
        # Compute component hashes
        config_hash = self.hash_data(safe_config)
        suite_hash = self.hash_data(suite_data)
        responses_hash, _ = self.hash_directory(responses_dir)
        
        # Create attestation
        from .. import __version__
        attestation = Attestation(
            run_id=run_id,
            manifest_hash=manifest['master_hash'],
            config_hash=config_hash,
            suite_hash=suite_hash,
            responses_hash=responses_hash,
            environment=env,
            created_at=datetime.utcnow().isoformat() + 'Z',
            meridian_version=__version__,
            signature=None,  # Reserved for future
        )
        
        # Save attestation
        attestation_path = run_dir / 'attestation.json'
        with open(attestation_path, 'w', encoding='utf-8') as f:
            f.write(self.canonical_json(attestation.to_dict()))
        
        return attestation
    
    # =========================================================================
    # Verification
    # =========================================================================
    
    def verify(self, run_id: str) -> tuple[bool, list[str]]:
        """
        Verify attestation integrity.
        
        Returns:
            (valid, issues) where issues is empty if valid
        """
        run_dir = self.base_dir / run_id
        issues = []
        
        # Load attestation
        attestation_path = run_dir / 'attestation.json'
        if not attestation_path.exists():
            return False, ['Attestation file not found']
        
        with open(attestation_path, 'r', encoding='utf-8') as f:
            attestation_data = json.load(f)
        
        # Load manifest
        manifest_path = run_dir / 'manifest.json'
        if not manifest_path.exists():
            return False, ['Manifest file not found']
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        # Verify manifest hash
        recomputed_manifest = self.generate_manifest(run_dir)
        if recomputed_manifest['master_hash'] != manifest['master_hash']:
            issues.append(f"Manifest hash mismatch: expected {manifest['master_hash'][:16]}..., got {recomputed_manifest['master_hash'][:16]}...")
        
        # Verify individual files
        for file_entry in manifest['files']:
            file_path = run_dir / file_entry['path']
            if not file_path.exists():
                issues.append(f"Missing file: {file_entry['path']}")
                continue
            
            current_hash = self.hash_file(file_path)
            if current_hash != file_entry['hash']:
                issues.append(f"Tampered file: {file_entry['path']}")
        
        # Verify attestation matches manifest
        if attestation_data['manifest_hash'] != manifest['master_hash']:
            issues.append("Attestation manifest_hash does not match manifest")
        
        valid = len(issues) == 0
        return valid, issues
    
    # =========================================================================
    # Load Attestation
    # =========================================================================
    
    def load_attestation(self, run_id: str) -> Optional[Attestation]:
        """Load attestation for a run."""
        run_dir = self.base_dir / run_id
        attestation_path = run_dir / 'attestation.json'
        
        if not attestation_path.exists():
            return None
        
        with open(attestation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruct EnvironmentInfo
        env_data = data.get('environment', {})
        env = EnvironmentInfo(
            python_version=env_data.get('python_version', 'unknown'),
            platform=env_data.get('platform', 'unknown'),
            os_name=env_data.get('os_name', 'unknown'),
            meridian_version=env_data.get('meridian_version', 'unknown'),
            git_commit=env_data.get('git_commit'),
            git_dirty=env_data.get('git_dirty', False),
            timestamp=env_data.get('timestamp', ''),
        )
        
        return Attestation(
            run_id=data['run_id'],
            manifest_hash=data['manifest_hash'],
            config_hash=data['config_hash'],
            suite_hash=data['suite_hash'],
            responses_hash=data['responses_hash'],
            environment=env,
            created_at=data['created_at'],
            meridian_version=data['meridian_version'],
            signature=data.get('signature'),
        )


# Singleton
_attestation_manager: Optional[AttestationManager] = None


def get_attestation_manager() -> AttestationManager:
    """Get global attestation manager."""
    global _attestation_manager
    if _attestation_manager is None:
        _attestation_manager = AttestationManager()
    return _attestation_manager
