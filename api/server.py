"""
Meridian REST API
FastAPI server for programmatic access
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.runner import SuiteRunner
from core.storage.db import get_db
from core.model_adapters import get_adapter
from core.model_adapters.base import GenerationConfig

app = FastAPI(
    title="Meridian API",
    description="LLM Evaluation Framework - REST API",
    version="0.3.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === MODELS ===

class RunRequest(BaseModel):
    suite: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 256

class GenerateRequest(BaseModel):
    prompt: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 256
    context: Optional[str] = None

class CompareRequest(BaseModel):
    run_a_id: str
    run_b_id: str

# === ENDPOINTS ===

@app.get("/")
def root():
    return {
        "name": "Meridian API",
        "version": "0.3.0",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

# --- Suites ---

@app.get("/api/suites")
def list_suites():
    """List all available test suites"""
    suites_dir = Path("suites")
    suites = []
    for f in suites_dir.glob("*.jsonl"):
        if not f.name.startswith("_"):
            suites.append(f.stem)
    return {"suites": sorted(suites)}

@app.get("/api/suites/{suite_name}")
def get_suite(suite_name: str):
    """Get suite details"""
    suite_path = Path(f"suites/{suite_name}.jsonl")
    if not suite_path.exists():
        raise HTTPException(404, f"Suite '{suite_name}' not found")
    
    import json
    with open(suite_path) as f:
        header = json.loads(f.readline())
        test_count = sum(1 for _ in f)
    
    return {
        "name": suite_name,
        "description": header.get("description", ""),
        "version": header.get("version", ""),
        "test_count": test_count
    }

# --- Models ---

@app.get("/api/models")
def list_models():
    """List available models"""
    from core.model_adapters import AVAILABLE_ADAPTERS
    models = []
    for model_id, adapter_cls in AVAILABLE_ADAPTERS.items():
        models.append({
            "id": model_id,
            "name": adapter_cls.__name__.replace("Adapter", ""),
            "requires_api_key": "api" in model_id.lower() or "openai" in model_id.lower() or "deepseek" in model_id.lower()
        })
    return {"models": models}

# --- Runs ---

@app.post("/api/run")
def run_suite(request: RunRequest, background_tasks: BackgroundTasks):
    """Start a suite run (async)"""
    runner = SuiteRunner()
    
    # Validate suite exists
    suite_path = Path(f"suites/{request.suite}.jsonl")
    if not suite_path.exists():
        raise HTTPException(404, f"Suite '{request.suite}' not found")
    
    # Run synchronously for now (can be made async)
    try:
        result = runner.run_suite(
            suite_path=str(suite_path),
            model_id=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return {
            "run_id": result.run_id,
            "suite": request.suite,
            "model": request.model,
            "accuracy": result.accuracy,
            "passed": result.passed,
            "failed": result.failed,
            "mean_latency_ms": result.mean_latency_ms
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/runs")
def list_runs(limit: int = 20):
    """List recent runs"""
    db = get_db()
    runs = db.get_runs(limit=limit)
    return {"runs": runs}

@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    """Get run details"""
    db = get_db()
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run '{run_id}' not found")
    
    results = db.get_results(run_id)
    return {
        "run": run,
        "results": results
    }

@app.get("/api/runs/{run_id}/results")
def get_run_results(run_id: str):
    """Get just the test results for a run"""
    db = get_db()
    results = db.get_results(run_id)
    if not results:
        raise HTTPException(404, f"No results for run '{run_id}'")
    return {"results": results}

# --- Compare ---

@app.post("/api/compare")
def compare_runs(request: CompareRequest):
    """Compare two runs"""
    runner = SuiteRunner()
    try:
        comparison = runner.compare_runs(request.run_a_id, request.run_b_id)
        return {
            "run_a": request.run_a_id,
            "run_b": request.run_b_id,
            "model_a": comparison.model_a,
            "model_b": comparison.model_b,
            "accuracy_delta": comparison.accuracy_delta,
            "regressions": len(comparison.regressions),
            "improvements": len(comparison.improvements),
            "regression_ids": comparison.regressions[:10],
            "improvement_ids": comparison.improvements[:10]
        }
    except Exception as e:
        raise HTTPException(500, str(e))

# --- Generate ---

@app.post("/api/generate")
def generate(request: GenerateRequest):
    """Single generation call"""
    try:
        adapter = get_adapter(request.model)
        config = GenerationConfig(
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        prompt = request.prompt
        if request.context:
            prompt = f"{request.context}\n\n{prompt}"
        
        import time
        start = time.time()
        result = adapter.generate(prompt, config)
        latency = (time.time() - start) * 1000
        
        return {
            "output": result.output,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "latency_ms": latency,
            "model": request.model
        }
    except Exception as e:
        raise HTTPException(500, str(e))


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    print(f"Starting Meridian API on http://{args.host}:{args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")
    start_server(args.host, args.port)
