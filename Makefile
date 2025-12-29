.PHONY: install run-ui run-suite check test clean demo help

# Default target
help:
	@echo "Meridian - Rigorous LLM Evaluation Framework"
	@echo ""
	@echo "Usage:"
	@echo "  make install      Install dependencies"
	@echo "  make run-ui       Start Streamlit UI"
	@echo "  make run-suite    Run a test suite (SUITE=name MODEL=id)"
	@echo "  make check        Run baseline checks"
	@echo "  make test         Run pytest tests"
	@echo "  make demo         Generate demo data"
	@echo "  make clean        Clean artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make run-suite SUITE=math_short MODEL=local_gpt2"
	@echo "  make check BASELINE=baselines/gpt2_baseline.json"

# Installation
install:
	pip install -e ".[dev]"

# Run Streamlit UI
run-ui:
	streamlit run ui/app.py

# Run a test suite
SUITE ?= math_short
MODEL ?= local_gpt2
run-suite:
	python -m core.cli run --suite $(SUITE) --model $(MODEL)

# Check against baseline
BASELINE ?= baselines/gpt2_baseline.json
check:
	python -m core.cli check --baseline $(BASELINE) --model $(MODEL)

# Compare two runs
compare:
	python -m core.cli compare --run-a $(RUN_A) --run-b $(RUN_B)

# Run tests
test:
	pytest tests/ -v

# Generate demo data
demo:
	python scripts/make_demo_data.py

# Clean artifacts
clean:
	rm -rf data/results/*
	rm -rf reports/output/*
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# List available suites and models
list-suites:
	python -m core.cli list-suites

list-models:
	python -m core.cli list-models
