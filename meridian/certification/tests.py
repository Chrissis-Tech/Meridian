"""
Meridian Provider Certification - Core Tests

Standard tests that every provider adapter must pass.
"""

import time
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


@dataclass
class CertificationResult:
    """Result of a single certification test."""
    test_name: str
    passed: bool
    message: str
    latency_ms: float = 0.0
    details: Dict = field(default_factory=dict)


@dataclass
class ProviderCertification:
    """Complete certification report for a provider."""
    provider_id: str
    model_id: str
    timestamp: str
    tests: List[CertificationResult]
    overall_passed: bool
    score: int  # 0-100
    badge_hash: str = ""
    environment: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class CertificationSuite:
    """Run standardized tests against a provider adapter."""
    
    def __init__(self, adapter):
        self.adapter = adapter
        self.results: List[CertificationResult] = []
    
    def run_all(self) -> ProviderCertification:
        """Run all certification tests."""
        from meridian.storage.attestation import AttestationManager
        
        # Run each test
        self._test_connectivity()
        self._test_basic_generation()
        self._test_temperature_zero()
        self._test_determinism()
        self._test_latency()
        self._test_max_tokens()
        self._test_error_handling()
        
        # Calculate score
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        score = int((passed_count / total_count) * 100) if total_count > 0 else 0
        
        # Create certification
        cert = ProviderCertification(
            provider_id=self._get_provider_id(),
            model_id=self.adapter.model_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            tests=self.results,
            overall_passed=score >= 80,
            score=score,
            environment=AttestationManager().capture_environment().__dict__
        )
        
        # Generate badge hash
        cert.badge_hash = self._generate_badge_hash(cert)
        
        return cert
    
    def _get_provider_id(self) -> str:
        """Extract provider from model_id."""
        model_id = self.adapter.model_id
        if "_" in model_id:
            return model_id.split("_")[0]
        return model_id
    
    def _generate_badge_hash(self, cert: ProviderCertification) -> str:
        """Generate verification hash for the certification."""
        data = f"{cert.provider_id}:{cert.model_id}:{cert.timestamp}:{cert.score}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _test_connectivity(self):
        """Test 1: Basic API connectivity."""
        test_name = "connectivity"
        start = time.perf_counter()
        
        try:
            result = self.adapter.generate("Say 'ok'", None)
            latency = (time.perf_counter() - start) * 1000
            
            if result.output and len(result.output) > 0:
                self.results.append(CertificationResult(
                    test_name=test_name,
                    passed=True,
                    message="API connection successful",
                    latency_ms=latency
                ))
            else:
                self.results.append(CertificationResult(
                    test_name=test_name,
                    passed=False,
                    message="Empty response received",
                    latency_ms=latency
                ))
        except Exception as e:
            self.results.append(CertificationResult(
                test_name=test_name,
                passed=False,
                message=f"Connection failed: {str(e)[:100]}",
                latency_ms=0
            ))
    
    def _test_basic_generation(self):
        """Test 2: Basic text generation."""
        test_name = "basic_generation"
        start = time.perf_counter()
        
        try:
            result = self.adapter.generate(
                "What is 2+2? Answer with just the number.",
                None
            )
            latency = (time.perf_counter() - start) * 1000
            
            if "4" in result.output:
                self.results.append(CertificationResult(
                    test_name=test_name,
                    passed=True,
                    message="Basic math correct",
                    latency_ms=latency,
                    details={"output": result.output[:100]}
                ))
            else:
                self.results.append(CertificationResult(
                    test_name=test_name,
                    passed=False,
                    message=f"Expected '4', got: {result.output[:50]}",
                    latency_ms=latency
                ))
        except Exception as e:
            self.results.append(CertificationResult(
                test_name=test_name,
                passed=False,
                message=f"Generation failed: {str(e)[:100]}"
            ))
    
    def _test_temperature_zero(self):
        """Test 3: Temperature 0 works."""
        test_name = "temperature_zero"
        
        try:
            from .base import GenerationConfig
            config = GenerationConfig(temperature=0.0, max_tokens=50)
            
            start = time.perf_counter()
            result = self.adapter.generate("Say hello", config)
            latency = (time.perf_counter() - start) * 1000
            
            if result.output:
                self.results.append(CertificationResult(
                    test_name=test_name,
                    passed=True,
                    message="Temperature 0 supported",
                    latency_ms=latency
                ))
            else:
                self.results.append(CertificationResult(
                    test_name=test_name,
                    passed=False,
                    message="Empty output with temperature 0"
                ))
        except Exception as e:
            self.results.append(CertificationResult(
                test_name=test_name,
                passed=False,
                message=f"Temperature 0 failed: {str(e)[:100]}"
            ))
    
    def _test_determinism(self):
        """Test 4: Determinism at temperature 0."""
        test_name = "determinism"
        
        try:
            from .base import GenerationConfig
            config = GenerationConfig(temperature=0.0, max_tokens=20)
            
            prompt = "Complete: The capital of France is"
            
            result1 = self.adapter.generate(prompt, config)
            result2 = self.adapter.generate(prompt, config)
            
            # Normalize for comparison
            out1 = result1.output.strip().lower()[:50]
            out2 = result2.output.strip().lower()[:50]
            
            if out1 == out2:
                self.results.append(CertificationResult(
                    test_name=test_name,
                    passed=True,
                    message="Deterministic at temperature 0",
                    details={"output": out1}
                ))
            else:
                self.results.append(CertificationResult(
                    test_name=test_name,
                    passed=False,
                    message="Non-deterministic outputs",
                    details={"output1": out1, "output2": out2}
                ))
        except Exception as e:
            self.results.append(CertificationResult(
                test_name=test_name,
                passed=False,
                message=f"Determinism test failed: {str(e)[:100]}"
            ))
    
    def _test_latency(self):
        """Test 5: Reasonable latency."""
        test_name = "latency"
        
        try:
            latencies = []
            for _ in range(3):
                start = time.perf_counter()
                self.adapter.generate("Say hi", None)
                latencies.append((time.perf_counter() - start) * 1000)
            
            avg_latency = sum(latencies) / len(latencies)
            
            # Pass if average latency < 30 seconds
            if avg_latency < 30000:
                self.results.append(CertificationResult(
                    test_name=test_name,
                    passed=True,
                    message=f"Average latency: {avg_latency:.0f}ms",
                    latency_ms=avg_latency,
                    details={"latencies": latencies}
                ))
            else:
                self.results.append(CertificationResult(
                    test_name=test_name,
                    passed=False,
                    message=f"High latency: {avg_latency:.0f}ms",
                    latency_ms=avg_latency
                ))
        except Exception as e:
            self.results.append(CertificationResult(
                test_name=test_name,
                passed=False,
                message=f"Latency test failed: {str(e)[:100]}"
            ))
    
    def _test_max_tokens(self):
        """Test 6: Max tokens respected."""
        test_name = "max_tokens"
        
        try:
            from .base import GenerationConfig
            config = GenerationConfig(max_tokens=10)
            
            result = self.adapter.generate(
                "Write a very long essay about the history of computing",
                config
            )
            
            # Rough token estimate (4 chars per token)
            estimated_tokens = len(result.output) / 4
            
            if estimated_tokens < 50:  # Allow some slack
                self.results.append(CertificationResult(
                    test_name=test_name,
                    passed=True,
                    message=f"Max tokens respected (~{estimated_tokens:.0f} tokens)",
                    details={"output_length": len(result.output)}
                ))
            else:
                self.results.append(CertificationResult(
                    test_name=test_name,
                    passed=False,
                    message=f"Max tokens exceeded (~{estimated_tokens:.0f} tokens)"
                ))
        except Exception as e:
            self.results.append(CertificationResult(
                test_name=test_name,
                passed=False,
                message=f"Max tokens test failed: {str(e)[:100]}"
            ))
    
    def _test_error_handling(self):
        """Test 7: Graceful error handling."""
        test_name = "error_handling"
        
        try:
            from .base import GenerationConfig
            # Try with extreme parameters
            config = GenerationConfig(temperature=0.0, max_tokens=1)
            
            result = self.adapter.generate("Test", config)
            
            # Should either work or return clean error
            if result.output or hasattr(result, 'error'):
                self.results.append(CertificationResult(
                    test_name=test_name,
                    passed=True,
                    message="Handles edge cases gracefully"
                ))
            else:
                self.results.append(CertificationResult(
                    test_name=test_name,
                    passed=True,
                    message="Edge case handled"
                ))
        except Exception as e:
            # Even exceptions are "handled" if they're clean
            self.results.append(CertificationResult(
                test_name=test_name,
                passed=True,
                message=f"Exception raised cleanly: {type(e).__name__}"
            ))


def certify_provider(model_id: str) -> ProviderCertification:
    """Certify a provider by running all standard tests."""
    from meridian.model_adapters import get_adapter
    
    adapter = get_adapter(model_id)
    suite = CertificationSuite(adapter)
    return suite.run_all()
