"""
Meridian Cryptographic Signing - Ed25519 Signatures

Provides cryptographic authenticity for attested runs.
"""

import base64
import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class SigningKey:
    """Ed25519 signing key pair."""
    private_key: bytes
    public_key: bytes
    key_id: str  # First 8 chars of public key hash
    
    def to_dict(self) -> dict:
        return {
            'public_key': base64.b64encode(self.public_key).decode('ascii'),
            'key_id': self.key_id,
        }


class AttestationSigner:
    """
    Ed25519 signing for attestation bundles.
    
    THREAT MODEL (what this protects against):
    =========================================
    
    1. DETECTS: Bundle tampering after creation
       - Any change to manifest/responses/config invalidates signature
       
    2. PROVES: Who signed the bundle (key holder identity)
       - Signature proves possession of private key
       - Public key can be registered/known in advance
       
    3. DOES NOT PROVE: Remote model behavior
       - Cannot prove OpenAI/DeepSeek actually ran the model
       - Cannot prove model wasn't locally faked
       - Cannot prove model version/weights unchanged
       
    4. DOES NOT PROTECT: Against key compromise
       - If private key is stolen, attacker can sign fake bundles
       
    RECOMMENDED USAGE:
    - Generate key pair once, protect private key
    - Publish public key in repo/DNS/keybase
    - Sign all production attestations
    - Verify before trusting any shared bundle
    """
    
    def __init__(self, keys_dir: Optional[Path] = None):
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey, Ed25519PublicKey
            )
            from cryptography.hazmat.primitives import serialization
            self._crypto_available = True
        except ImportError:
            self._crypto_available = False
        
        self.keys_dir = keys_dir or Path.home() / '.meridian' / 'keys'
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        
        self._private_key = None
        self._public_key = None
        self._key_id = None
    
    @property
    def available(self) -> bool:
        """Check if cryptography library is available."""
        return self._crypto_available
    
    def generate_keypair(self) -> SigningKey:
        """Generate new Ed25519 key pair."""
        if not self._crypto_available:
            raise RuntimeError("cryptography library not installed. Run: pip install cryptography")
        
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization
        
        # Generate key
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        
        # Serialize
        private_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        
        # Generate key ID
        key_id = hashlib.sha256(public_bytes).hexdigest()[:8]
        
        return SigningKey(
            private_key=private_bytes,
            public_key=public_bytes,
            key_id=key_id
        )
    
    def save_keypair(self, key: SigningKey, name: str = "default") -> Tuple[Path, Path]:
        """Save key pair to files."""
        private_path = self.keys_dir / f"{name}.key"
        public_path = self.keys_dir / f"{name}.pub"
        
        # Save private key (base64 encoded)
        with open(private_path, 'w') as f:
            f.write(base64.b64encode(key.private_key).decode('ascii'))
        
        # Restrict permissions (best effort on Windows)
        try:
            private_path.chmod(0o600)
        except:
            pass
        
        # Save public key with metadata
        with open(public_path, 'w') as f:
            json.dump({
                'key_id': key.key_id,
                'public_key': base64.b64encode(key.public_key).decode('ascii'),
                'algorithm': 'Ed25519',
            }, f, indent=2)
        
        return private_path, public_path
    
    def load_keypair(self, name: str = "default") -> Optional[SigningKey]:
        """Load key pair from files."""
        private_path = self.keys_dir / f"{name}.key"
        public_path = self.keys_dir / f"{name}.pub"
        
        if not private_path.exists() or not public_path.exists():
            return None
        
        with open(private_path, 'r') as f:
            private_bytes = base64.b64decode(f.read())
        
        with open(public_path, 'r') as f:
            public_data = json.load(f)
            public_bytes = base64.b64decode(public_data['public_key'])
            key_id = public_data['key_id']
        
        return SigningKey(
            private_key=private_bytes,
            public_key=public_bytes,
            key_id=key_id
        )
    
    def load_public_key(self, name: str = "default") -> Optional[dict]:
        """Load only public key (for verification)."""
        public_path = self.keys_dir / f"{name}.pub"
        
        if not public_path.exists():
            return None
        
        with open(public_path, 'r') as f:
            return json.load(f)
    
    def sign(self, data: bytes, key: SigningKey) -> str:
        """Sign data with Ed25519 private key."""
        if not self._crypto_available:
            raise RuntimeError("cryptography library not installed")
        
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        
        # Reconstruct private key
        private_key = Ed25519PrivateKey.from_private_bytes(key.private_key)
        
        # Sign
        signature = private_key.sign(data)
        
        return base64.b64encode(signature).decode('ascii')
    
    def verify(self, data: bytes, signature: str, public_key_bytes: bytes) -> bool:
        """Verify Ed25519 signature."""
        if not self._crypto_available:
            raise RuntimeError("cryptography library not installed")
        
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.exceptions import InvalidSignature
        
        try:
            public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            signature_bytes = base64.b64decode(signature)
            public_key.verify(signature_bytes, data)
            return True
        except InvalidSignature:
            return False
        except Exception:
            return False
    
    def sign_attestation(self, attestation_path: Path, key: SigningKey) -> dict:
        """
        Sign an attestation file.
        
        Returns updated attestation with signature.
        """
        with open(attestation_path, 'r', encoding='utf-8') as f:
            attestation = json.load(f)
        
        # Create signable payload (everything except signature)
        attestation.pop('signature', None)
        payload = json.dumps(attestation, sort_keys=True, separators=(',', ':')).encode('utf-8')
        
        # Sign
        signature = self.sign(payload, key)
        
        # Update attestation
        attestation['signature'] = signature
        attestation['signer_key_id'] = key.key_id
        
        # Save
        with open(attestation_path, 'w', encoding='utf-8') as f:
            json.dump(attestation, f, sort_keys=True, separators=(',', ':'))
        
        return attestation
    
    def verify_attestation(self, attestation_path: Path, public_key_bytes: bytes) -> Tuple[bool, str]:
        """
        Verify attestation signature.
        
        Returns (valid, message).
        """
        with open(attestation_path, 'r', encoding='utf-8') as f:
            attestation = json.load(f)
        
        signature = attestation.get('signature')
        if not signature:
            return False, "No signature found in attestation"
        
        # Reconstruct signable payload
        attestation_copy = dict(attestation)
        attestation_copy.pop('signature', None)
        attestation_copy.pop('signer_key_id', None)
        payload = json.dumps(attestation_copy, sort_keys=True, separators=(',', ':')).encode('utf-8')
        
        # Verify
        if self.verify(payload, signature, public_key_bytes):
            key_id = attestation.get('signer_key_id', 'unknown')
            return True, f"Valid signature from key {key_id}"
        else:
            return False, "Invalid signature - attestation may have been tampered"


# Singleton
_signer: Optional[AttestationSigner] = None

def get_signer() -> AttestationSigner:
    """Get global attestation signer."""
    global _signer
    if _signer is None:
        _signer = AttestationSigner()
    return _signer
